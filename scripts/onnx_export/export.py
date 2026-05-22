#!/usr/bin/env python3
"""Merged ONNX export utilities for Qwen3-TTS.

Subcommands:
  all            export all isolated ONNX submodels
  tokenizer      export tokenizer12hz ONNX models
  consolidate    merge external-data shards into .onnx.data files
  patch-decoder  insert FP32 islands into tokenizer decoder ONNX
"""

"""打印 PyTorch 模型和 ONNX 输入输出 dtype 信息的小工具。

导出 FP16/FP32 子模型时，最容易出错的是 runtime 输入 dtype 和 ONNX 声明不一致。
这些打印函数帮助在导出阶段直接看到模型参数 dtype、ONNX IO dtype 和预期输入。
"""


from collections import Counter
from pathlib import Path

import onnx


def print_torch_dtype_summary(model, title="PyTorch model") -> None:
    """统计 PyTorch parameters/buffers 的 dtype 分布。"""
    param_dtypes = Counter(str(param.dtype).replace("torch.", "") for param in model.parameters())
    buffer_dtypes = Counter(str(buf.dtype).replace("torch.", "") for buf in model.buffers())
    print(f"\n[DType] {title}")
    print(f"  parameter dtypes: {dict(param_dtypes)}")
    print(f"  buffer dtypes:    {dict(buffer_dtypes)}")


def print_onnx_io_dtypes(onnx_path, title=None) -> None:
    """打印 ONNX graph input/output 的元素类型和形状。"""
    model = onnx.load(str(onnx_path), load_external_data=False)
    title = title or Path(onnx_path).name
    print(f"\n[DType] ONNX IO: {title}")
    for value in list(model.graph.input) + list(model.graph.output):
        tensor_type = value.type.tensor_type
        elem_type = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
        dims = []
        for dim in tensor_type.shape.dim:
            dims.append(dim.dim_param or dim.dim_value)
        print(f"  {value.name}: {elem_type} {dims}")


def print_expected_inputs(title, specs) -> None:
    """打印 runtime 期望喂给某个 ONNX 子图的输入 dtype/shape。"""
    print(f"\n[DType] Expected runtime inputs: {title}")
    for name, dtype, shape in specs:
        print(f"  {name}: {dtype} {shape}")

# ---- tokenizer export ----
"""导出 Qwen3-TTS 12Hz speech tokenizer 的 ONNX 模型。

这个文件从 convert_onnx.py 中拆分出来，只负责导出：
  - tokenizer12hz_encode.onnx
  - tokenizer12hz_decode.onnx

speech tokenizer 是 TTS 的“声学离散化器”：
encoder 把 24k waveform 变成 12Hz、每帧 16 个 codebook 的 codec codes；
decoder 把 codec codes 还原成 24k waveform。声音克隆主模型只负责生成
这些 codes，最终听到的声音由 decoder 负责合成。
"""

import argparse
import os
from typing import Any, Optional

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import TensorProto, helper
from transformers import AutoConfig, AutoProcessor

from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSProcessor
from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer



def parse_torch_dtype(dtype: str) -> torch.dtype:
    """把命令行 dtype 字符串转换为 torch dtype。"""
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype}")


def export_tokenizer_12hz_encode(
    speech_tokenizer: Any,
    output_dir: str,
    device: torch.device,
    opset_version: int = 14,
) -> None:
    """导出 tokenizer12hz_encode.onnx：音频 waveform -> RVQ codec tokens。

    输入：audio [1, num_samples] float32 - 24 kHz 音频
    输出：codes [1, T, 16] int64         - T 个 codec 帧

    Args:
        speech_tokenizer: 已加载的 Qwen3TTSTokenizer 实例，要求包含 .model。
        output_dir: ONNX 文件输出目录。
        device: 构造导出 dummy input 使用的 torch 设备。
        opset_version: encoder ONNX 导出使用的 opset 版本。
    """
    print("Exporting tokenizer12hz_encode.onnx ...")

    tokenizer_model = speech_tokenizer.model

    class TokenizerEncoder(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, audio):
            # 不直接 trace self.model.encode()，因为里面有 Python 列表推导
            # 和动态切片，容易被 torch.onnx.trace 固化。
            # 这里展开到底层 encoder.encode，返回纯 Tensor 路径：
            #   raw codes: [1, 16, T] -> [1, T, 16]
            encoded_frames = self.model.encoder.encode(
                input_values=audio.unsqueeze(1),
                return_dict=True,
            )
            codes = encoded_frames.audio_codes[:, :self.model.encoder_valid_num_quantizers]
            return codes.transpose(1, 2)

    wrapper = TokenizerEncoder(tokenizer_model)
    wrapper.eval()

    model_dtype = next(tokenizer_model.parameters()).dtype
    dummy_audio = torch.randn(1, 24000, device=device, dtype=model_dtype)

    torch.onnx.export(
        wrapper,
        (dummy_audio,),
        os.path.join(output_dir, "tokenizer12hz_encode.onnx"),
        input_names=["audio"],
        output_names=["codes"],
        dynamic_axes={
            "audio": {1: "num_samples"},
            "codes": {1: "num_frames"},
        },
        opset_version=opset_version,
    )
    patch_encoder_dynamic_reshape(os.path.join(output_dir, "tokenizer12hz_encode.onnx"))
    print("  Done: tokenizer12hz_encode.onnx")


def patch_encoder_dynamic_reshape(onnx_path: str) -> int:
    """修复 encoder ONNX 图里被 trace 固化的动态长度 Reshape。

    Args:
        onnx_path: tokenizer12hz_encode.onnx 文件路径，会原地覆盖保存。

    Returns:
        实际修复的节点数量。
    """
    model = onnx.load(onnx_path)
    patched = 0
    new_nodes = []

    for node in model.graph.node:
        # 这个 Reshape 来自 Mimi encoder transformer 的 causal mask 构造。
        # trace 1 秒音频时它会把 Range(seq) 固定 reshape 成 [25, 1]，
        # 导致 2 秒/3 秒输入时 Range 长度变成 50/75 后无法 reshape。
        # 正确做法是用动态 Unsqueeze(axis=1) 生成 [seq, 1]。
        if node.name == "/encoder_transformer/Reshape":
            axes_name = "/encoder_transformer/Unsqueeze_axis1_const_output_0"
            unsqueeze_out = "/encoder_transformer/Range_unsqueeze_axis1_output_0"
            new_nodes.append(
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=[axes_name],
                    name="/encoder_transformer/Unsqueeze_axis1_const",
                    value=helper.make_tensor("value", TensorProto.INT64, [1], [1]),
                )
            )
            new_nodes.append(
                helper.make_node(
                    "Unsqueeze",
                    inputs=["/encoder_transformer/Range_output_0", axes_name],
                    outputs=[unsqueeze_out],
                    name="/encoder_transformer/Range_unsqueeze_axis1",
                )
            )
            patched += 1
            continue

        if node.name == "/encoder_transformer/LessOrEqual":
            if node.input[1] == "/encoder_transformer/Reshape_output_0":
                node.input[1] = "/encoder_transformer/Range_unsqueeze_axis1_output_0"

        new_nodes.append(node)

    if patched:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
        onnx.checker.check_model(model)
        onnx.save(model, onnx_path)
        print(f"  Patched encoder dynamic reshape nodes: {patched}")

    return patched


def _register_diff_symbolic() -> None:
    """注册 decoder 导出需要的 aten::diff ONNX symbolic。

    Args:
        无。

    Returns:
        无返回值，注册结果会写入 PyTorch ONNX symbolic 全局表。
    """

    def _diff_symbolic(g, x, n, dim, prepend, append):
        # ONNX 没有直接等价的 torch.diff，这里用 Slice + Sub + Concat 模拟一阶差分。
        from torch.onnx.symbolic_helper import _get_const

        dim_val = _get_const(dim, "i", "dim")
        axes = g.op("Constant", value_t=torch.tensor([dim_val], dtype=torch.long))
        zero = g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))
        one = g.op("Constant", value_t=torch.tensor([1], dtype=torch.long))
        neg1 = g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long))
        big = g.op("Constant", value_t=torch.tensor([9223372036854775807], dtype=torch.long))

        a = g.op("Slice", x, zero, neg1, axes, one)
        b = g.op("Slice", x, one, big, axes, one)
        diff_result = g.op("Sub", b, a)

        first = g.op("Slice", x, zero, one, axes, one)
        zero_pad = g.op("Sub", first, first)

        return g.op("Concat", zero_pad, diff_result, axis_i=dim_val)

    torch.onnx.register_custom_op_symbolic("aten::diff", _diff_symbolic, 18)


def _fix_bool_cumsum(onnx_model: Any) -> int:
    """给 bool 输入的 CumSum 节点前插入 Cast(INT64)。

    Args:
        onnx_model: 已经加载到内存中的 ONNX ModelProto。

    Returns:
        实际插入的 Cast 节点数量。
    """
    name_to_node = {o: node for node in onnx_model.graph.node for o in node.output}
    cast_added = 0
    for i, node in enumerate(list(onnx_model.graph.node)):
        if node.op_type == "CumSum":
            data_input = node.input[0]
            src = name_to_node.get(data_input)
            if src and src.op_type in ("Not", "Equal", "Less", "Greater", "And", "Or"):
                cast_name = data_input + "_i64"
                cast_node = onnx.helper.make_node(
                    "Cast", inputs=[data_input], outputs=[cast_name], to=7
                )
                node.input[0] = cast_name
                onnx_model.graph.node.insert(i, cast_node)
                cast_added += 1
    return cast_added


def export_tokenizer_12hz_decode(
    speech_tokenizer: Any,
    output_dir: str,
    device: torch.device,
    opset_version: int = 18,
    trace_frames: int = 100,
) -> None:
    """导出 tokenizer12hz_decode*.onnx：RVQ codec tokens -> audio waveform。

    Args:
        speech_tokenizer: 已加载的 Qwen3TTSTokenizer 实例，要求包含 .model.decoder。
        output_dir: ONNX 文件输出目录。
        device: 构造导出 dummy input 使用的 torch 设备。
        opset_version: decoder ONNX 导出使用的 opset 版本，默认固定为 18。
        trace_frames: legacy ONNX tracer 使用的 dummy codec 帧数。当前 decoder
            waveform 输出实际会被这个长度限制，所以长音频导出时需要设大一些。
    """
    import io

    trace_T = int(trace_frames)
    if trace_T <= 0:
        raise ValueError("trace_frames must be positive")
    out_name = "tokenizer12hz_decode.onnx"
    label = "batch"

    print(f"Exporting {out_name}  [{label}, traced at T={trace_T}] ...")

    _register_diff_symbolic()

    speech_model = speech_tokenizer.model
    decode_upsample_rate = speech_model.decode_upsample_rate

    class DecoderForward(nn.Module):
        def __init__(self, decoder, upsample_rate):
            super().__init__()
            self.decoder = decoder
            self.upsample_rate = upsample_rate

        def forward(self, audio_codes):
            wav = self.decoder(audio_codes.transpose(1, 2))
            audio_values = wav.squeeze(1)
            lengths = (audio_codes[..., 0] >= 0).sum(dim=1) * self.upsample_rate
            # decoder 的主体会按 trace_T 产生偏长音频；这里按有效长度裁剪，
            # 让 ONNX 输出 shape 和原生 Python API 更一致。当前只支持 batch=1。
            audio_values = audio_values[:, :lengths[0]]
            return audio_values, lengths

    wrapper = DecoderForward(speech_model.decoder, decode_upsample_rate)
    wrapper.eval()

    dummy_codes = torch.randint(0, 1024, (1, trace_T, 16), device=device)

    buf = io.BytesIO()
    torch.onnx.export(
        wrapper,
        (dummy_codes,),
        buf,
        input_names=["audio_codes"],
        output_names=["audio_values", "lengths"],
        dynamic_axes={
            "audio_codes": {1: "codes_length"},
            "audio_values": {1: "audio_length"},
        },
        opset_version=opset_version,
        do_constant_folding=False,
        dynamo=False,
    )
    buf.seek(0)

    onnx_model = onnx.load_model_from_string(buf.getvalue())
    n_casts = _fix_bool_cumsum(onnx_model)
    print(f"  Post-processing: inserted {n_casts} Cast(INT64) before CumSum")

    out_path = os.path.join(output_dir, out_name)
    onnx.save(onnx_model, out_path)
    print(f"  Done: {out_name}")


def export_tokenizer_12hz_decode_chunk(
    speech_tokenizer: Any,
    output_dir: str,
    device: torch.device,
    opset_version: int = 18,
    trace_chunk_frames: int = 300,
    trace_context_frames: int = 25,
) -> None:
    """导出显式可选的 chunk decoder，用于 pipeline/chunk 实验。"""
    import io

    out_name = "tokenizer12hz_decode_chunk.onnx"
    trace_T = trace_context_frames + trace_chunk_frames
    print(
        f"Exporting {out_name} "
        f"[trace_chunk_frames={trace_chunk_frames}, "
        f"trace_context_frames={trace_context_frames}, trace_T={trace_T}] ..."
    )

    _register_diff_symbolic()

    speech_model = speech_tokenizer.model
    decode_upsample_rate = speech_model.decode_upsample_rate

    class ChunkDecoderForward(nn.Module):
        def __init__(self, decoder: nn.Module, upsample_rate: int):
            super().__init__()
            self.decoder = decoder
            self.upsample_rate = upsample_rate

        def forward(self, audio_codes: torch.Tensor, context_frames: torch.Tensor):
            wav = self.decoder(audio_codes.transpose(1, 2))
            audio_values = wav.squeeze(1)

            total_frames = (audio_codes[..., 0] >= 0).sum(dim=1)
            current_frames = total_frames - context_frames
            start_sample = context_frames * self.upsample_rate
            valid_samples = current_frames * self.upsample_rate

            audio_values = audio_values[:, start_sample : start_sample + valid_samples]
            return audio_values, valid_samples.unsqueeze(0)

    wrapper = ChunkDecoderForward(speech_model.decoder, decode_upsample_rate)
    wrapper.eval()

    dummy_codes = torch.randint(0, 1024, (1, trace_T, 16), device=device)
    dummy_context = torch.tensor(trace_context_frames, dtype=torch.long, device=device)

    buf = io.BytesIO()
    torch.onnx.export(
        wrapper,
        (dummy_codes, dummy_context),
        buf,
        input_names=["audio_codes", "context_frames"],
        output_names=["audio_values", "lengths"],
        dynamic_axes={
            "audio_codes": {1: "codes_length"},
            "audio_values": {1: "audio_length"},
        },
        opset_version=opset_version,
        do_constant_folding=False,
        dynamo=False,
    )
    buf.seek(0)

    onnx_model = onnx.load_model_from_string(buf.getvalue())
    n_casts = _fix_bool_cumsum(onnx_model)
    print(f"  Post-processing: inserted {n_casts} Cast(INT64) before CumSum")

    out_path = os.path.join(output_dir, out_name)
    onnx.save(onnx_model, out_path)
    print(f"  Done: {out_name}")


def _make_verify_audio(
    seconds: float,
    sample_rate: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """构造一段固定测试音频，避免随机输入导致每次校验结果不同。

    Args:
        seconds: 测试音频时长，单位秒。
        sample_rate: 测试音频采样率。
        device: 返回 Tensor 所在的 torch 设备。

    Returns:
        shape 为 [1, num_samples] 的 float32 Tensor。
    """
    num_samples = int(seconds * sample_rate)
    t = torch.arange(num_samples, device=device, dtype=dtype) / sample_rate

    # 用两个正弦波叠加，幅度保持较低，模拟一段稳定的语音状输入。
    audio = 0.20 * torch.sin(2 * torch.pi * 220 * t)
    audio += 0.05 * torch.sin(2 * torch.pi * 440 * t)
    return audio.unsqueeze(0)


def _make_verify_audio_numpy(seconds: float, sample_rate: int) -> np.ndarray:
    """构造原生 Python API 使用的 numpy 音频，内容和 Tensor 校验音频一致。

    Args:
        seconds: 测试音频时长，单位秒。
        sample_rate: 测试音频采样率。

    Returns:
        shape 为 [num_samples] 的 float32 numpy 数组。
    """
    num_samples = int(seconds * sample_rate)
    t = np.arange(num_samples, dtype=np.float32) / sample_rate
    audio = 0.20 * np.sin(2 * np.pi * 220 * t)
    audio += 0.05 * np.sin(2 * np.pi * 440 * t)
    return audio.astype(np.float32)


def _to_numpy(x: Any) -> np.ndarray:
    """把 PyTorch Tensor 或 ONNX 输出统一转成 numpy，方便比较。

    Args:
        x: PyTorch Tensor、numpy 数组或可被 np.asarray 接收的对象。

    Returns:
        numpy 数组。
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _create_onnx_session(path: str) -> Any:
    """创建 ONNX Runtime Session；校验阶段固定用 CPU，减少设备差异。

    Args:
        path: ONNX 文件路径。

    Returns:
        onnxruntime.InferenceSession 实例。
    """
    import onnxruntime as ort

    if not os.path.exists(path):
        raise FileNotFoundError(f"ONNX file not found: {path}")
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def _compare_audio(
    name: str,
    reference: Any,
    actual: Any,
    min_corr: float,
    max_abs_tol: float,
    mean_abs_tol: float,
) -> bool:
    """比较两段波形，输出长度、最大误差、平均误差和相关系数。

    Args:
        name: 打印日志时使用的比较项名称。
        reference: 参考波形，通常来自 PyTorch 或 batch ONNX。
        actual: 待校验波形，通常来自 ONNX。
        min_corr: 允许通过的最低相关系数。
        max_abs_tol: 允许通过的最大绝对误差阈值。
        mean_abs_tol: 允许通过的平均绝对误差阈值。

    Returns:
        所有指标是否通过阈值检查。
    """
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    got = np.asarray(actual, dtype=np.float64).reshape(-1)

    min_len = min(ref.shape[0], got.shape[0])
    ref_cmp = ref[:min_len]
    got_cmp = got[:min_len]
    diff = got_cmp - ref_cmp

    max_abs = float(np.max(np.abs(diff))) if min_len else float("inf")
    mean_abs = float(np.mean(np.abs(diff))) if min_len else float("inf")
    rmse = float(np.sqrt(np.mean(diff * diff))) if min_len else float("inf")

    if min_len and np.std(ref_cmp) > 0 and np.std(got_cmp) > 0:
        corr = float(np.corrcoef(ref_cmp, got_cmp)[0, 1])
    else:
        corr = float("nan")

    shape_ok = ref.shape == got.shape
    metric_ok = (
        shape_ok
        and corr >= min_corr
        and max_abs <= max_abs_tol
        and mean_abs <= mean_abs_tol
    )

    print(f"{name}:")
    print(f"  reference shape: {ref.shape}")
    print(f"  actual shape:    {got.shape}")
    print(f"  shape match:     {shape_ok}")
    print(f"  max_abs_diff:    {max_abs:.8f}")
    print(f"  mean_abs_diff:   {mean_abs:.8f}")
    print(f"  rmse:            {rmse:.8f}")
    print(f"  corr:            {corr:.8f}")
    print(f"  pass:            {metric_ok}")
    return metric_ok


def verify_tokenizer_12hz_encoder(
    speech_tokenizer: Any,
    output_dir: str,
    device: torch.device,
    audio: torch.Tensor,
) -> tuple[bool, torch.Tensor]:
    """校验 encoder：同一段 audio 下，PyTorch codes 必须和 ONNX codes 完全一致。

    Args:
        speech_tokenizer: 已加载的 Qwen3TTSTokenizer 实例。
        output_dir: 已导出的 ONNX 文件所在目录。
        device: PyTorch 推理设备。
        audio: shape 为 [1, num_samples] 的测试音频 Tensor。

    Returns:
        二元组：(是否通过, PyTorch encoder 生成的 codes Tensor)。
    """
    print("\n[Verify] tokenizer12hz_encode.onnx")
    encoder_path = os.path.join(output_dir, "tokenizer12hz_encode.onnx")
    session = _create_onnx_session(encoder_path)

    with torch.no_grad():
        # 和导出的 ONNX wrapper 保持完全一致：直接走底层 model.encode，
        # 并显式传全 1 padding_mask，避免 wrapper 额外预处理影响对齐判断。
        padding_mask = torch.ones_like(audio, dtype=torch.long)
        encoded = speech_tokenizer.model.encode(audio, padding_mask, return_dict=True)
        torch_codes = encoded.audio_codes[0].unsqueeze(0)

    onnx_codes = session.run(
        ["codes"],
        {"audio": _to_numpy(audio).astype(np.float32)},
    )[0]

    torch_codes_np = _to_numpy(torch_codes)
    onnx_codes_np = _to_numpy(onnx_codes)

    shape_ok = torch_codes_np.shape == onnx_codes_np.shape
    exact_ok = np.array_equal(torch_codes_np, onnx_codes_np)
    mismatch_count = 0
    if shape_ok:
        mismatch_count = int(np.sum(torch_codes_np != onnx_codes_np))

    print(f"  torch codes shape: {torch_codes_np.shape}")
    print(f"  onnx  codes shape: {onnx_codes_np.shape}")
    print(f"  shape match:       {shape_ok}")
    print(f"  exact match:       {exact_ok}")
    print(f"  mismatch count:    {mismatch_count}")

    return exact_ok, torch_codes


def verify_tokenizer_12hz_decoder(
    speech_tokenizer: Any,
    output_dir: str,
    device: torch.device,
    codes: torch.Tensor,
    min_corr: float,
    max_abs_tol: float,
    mean_abs_tol: float,
) -> tuple[bool, np.ndarray]:
    """校验 batch decoder：同一组 codec codes 下，PyTorch 和 ONNX 波形应高度一致。

    Args:
        speech_tokenizer: 已加载的 Qwen3TTSTokenizer 实例。
        output_dir: 已导出的 ONNX 文件所在目录。
        device: PyTorch 推理设备，当前函数保留该参数用于接口一致。
        codes: shape 为 [1, T, 16] 的 codec codes Tensor。
        min_corr: 允许通过的最低相关系数。
        max_abs_tol: 允许通过的最大绝对误差阈值。
        mean_abs_tol: 允许通过的平均绝对误差阈值。

    Returns:
        二元组：(是否通过, batch ONNX decoder 生成的音频 numpy 数组)。
    """
    print("\n[Verify] tokenizer12hz_decode.onnx")
    decoder_path = os.path.join(output_dir, "tokenizer12hz_decode.onnx")
    session = _create_onnx_session(decoder_path)

    with torch.no_grad():
        wav = speech_tokenizer.model.decoder(codes.transpose(1, 2))
        torch_audio = wav.squeeze(1)
        torch_lengths = (codes[..., 0] >= 0).sum(dim=1) * speech_tokenizer.model.decode_upsample_rate

    onnx_audio, onnx_lengths = session.run(
        ["audio_values", "lengths"],
        {"audio_codes": _to_numpy(codes).astype(np.int64)},
    )

    torch_lengths_np = _to_numpy(torch_lengths).astype(np.int64)
    onnx_lengths_np = _to_numpy(onnx_lengths).astype(np.int64)
    length_ok = np.array_equal(torch_lengths_np, onnx_lengths_np)

    print(f"  torch lengths: {torch_lengths_np}")
    print(f"  onnx  lengths: {onnx_lengths_np}")
    print(f"  length match:  {length_ok}")

    audio_ok = _compare_audio(
        "  audio compare",
        _to_numpy(torch_audio),
        onnx_audio,
        min_corr,
        max_abs_tol,
        mean_abs_tol,
    )
    return length_ok and audio_ok, onnx_audio


def verify_tokenizer_12hz_native_pipeline(
    speech_tokenizer: Any,
    output_dir: str,
    verify_seconds: float,
    min_corr: float,
    max_abs_tol: float,
    mean_abs_tol: float,
) -> bool:
    """校验原生 Python API 端到端行为。

    这一层校验模拟真实用户调用方式：
      1. 原生 Python: speech_tokenizer.encode(numpy_audio, sr=24000)
         再 speech_tokenizer.decode(encoded)
      2. ONNX pipeline: tokenizer12hz_encode.onnx
         再 tokenizer12hz_decode.onnx

    这样可以确认 ONNX 文件不仅底层 tensor 对齐，也和官方 Python wrapper
    的实际输入预处理、padding_mask、encode/decode 组合行为一致。

    Args:
        speech_tokenizer: 已加载的 Qwen3TTSTokenizer 实例。
        output_dir: 已导出的 ONNX 文件所在目录。
        verify_seconds: 用于校验的合成音频时长，单位秒。
        min_corr: 允许通过的最低相关系数。
        max_abs_tol: 允许通过的最大绝对误差阈值。
        mean_abs_tol: 允许通过的平均绝对误差阈值。

    Returns:
        原生 Python API 端到端输出是否和 ONNX pipeline 对齐。
    """
    print("\n[Verify] native Python API vs ONNX pipeline")

    audio_np = _make_verify_audio_numpy(verify_seconds, sample_rate=24000)

    # 原生 API 会调用 feature_extractor，生成 input_values/padding_mask，
    # 然后走底层 model.encode；这是最接近实际使用的参考结果。
    native_encoded = speech_tokenizer.encode(audio_np, sr=24000, return_dict=True)
    native_codes = native_encoded.audio_codes[0]
    native_wavs, native_sr = speech_tokenizer.decode(native_encoded)
    native_audio = native_wavs[0]

    encoder_session = _create_onnx_session(
        os.path.join(output_dir, "tokenizer12hz_encode.onnx")
    )
    decoder_session = _create_onnx_session(
        os.path.join(output_dir, "tokenizer12hz_decode.onnx")
    )

    # ONNX encoder 的输入是 [1, num_samples]，和导出时的 wrapper 一致。
    onnx_codes = encoder_session.run(
        ["codes"],
        {"audio": audio_np.reshape(1, -1).astype(np.float32)},
    )[0]

    # ONNX decoder 直接吃 encoder 输出的 [1, T, 16] codec codes。
    onnx_audio, onnx_lengths = decoder_session.run(
        ["audio_values", "lengths"],
        {"audio_codes": onnx_codes.astype(np.int64)},
    )

    native_codes_np = _to_numpy(native_codes)
    onnx_codes_np = _to_numpy(onnx_codes)
    if native_codes_np.ndim == 2:
        native_codes_np = native_codes_np[None, ...]

    codes_shape_ok = native_codes_np.shape == onnx_codes_np.shape
    codes_exact_ok = np.array_equal(native_codes_np, onnx_codes_np)
    mismatch_count = 0
    if codes_shape_ok:
        mismatch_count = int(np.sum(native_codes_np != onnx_codes_np))

    expected_length = onnx_codes_np.shape[1] * speech_tokenizer.model.decode_upsample_rate
    length_ok = int(onnx_lengths[0]) == expected_length
    sample_rate_ok = native_sr == 24000

    print("  codec codes:")
    print(f"    native shape:    {native_codes_np.shape}")
    print(f"    onnx shape:      {onnx_codes_np.shape}")
    print(f"    shape match:     {codes_shape_ok}")
    print(f"    exact match:     {codes_exact_ok}")
    print(f"    mismatch count:  {mismatch_count}")
    print("  metadata:")
    print(f"    native sr:       {native_sr}")
    print(f"    sample rate ok:  {sample_rate_ok}")
    print(f"    onnx length:     {onnx_lengths}")
    print(f"    expected length: {expected_length}")
    print(f"    length ok:       {length_ok}")

    audio_ok = _compare_audio(
        "  native decode vs onnx pipeline decode",
        native_audio,
        onnx_audio[0],
        min_corr,
        max_abs_tol,
        mean_abs_tol,
    )

    return codes_exact_ok and sample_rate_ok and length_ok and audio_ok


def verify_tokenizer_12hz(
    speech_tokenizer: Any,
    output_dir: str,
    device: torch.device,
    verify_seconds: float,
    min_corr: float,
    max_abs_tol: float,
    mean_abs_tol: float,
) -> bool:
    """完整校验入口：encoder、batch decoder、native pipeline 都在这里执行。

    Args:
        speech_tokenizer: 已加载的 Qwen3TTSTokenizer 实例。
        output_dir: 已导出的 ONNX 文件所在目录。
        device: PyTorch 推理设备。
        verify_seconds: 用于校验的合成音频时长，单位秒。
        min_corr: 允许通过的最低相关系数。
        max_abs_tol: 允许通过的最大绝对误差阈值。
        mean_abs_tol: 允许通过的平均绝对误差阈值。

    Returns:
        所有校验项是否全部通过。
    """
    print("\n" + "=" * 60)
    print("开始校验 12Hz speech tokenizer ONNX 导出结果")
    print("=" * 60)

    audio = _make_verify_audio(verify_seconds, sample_rate=24000, device=device)

    # 1. 底层 encoder 校验：离散 token 输出，正确标准是完全一致。
    encoder_ok, codes = verify_tokenizer_12hz_encoder(
        speech_tokenizer, output_dir, device, audio
    )

    # 2. 底层 batch decoder 校验：浮点波形输出，比较长度和误差指标。
    decoder_ok, batch_onnx_audio = verify_tokenizer_12hz_decoder(
        speech_tokenizer,
        output_dir,
        device,
        codes,
        min_corr,
        max_abs_tol,
        mean_abs_tol,
    )

    # 3. 原生 Python API 端到端校验：这是最终使用形态的对齐检查。
    native_pipeline_ok = verify_tokenizer_12hz_native_pipeline(
        speech_tokenizer,
        output_dir,
        verify_seconds,
        min_corr,
        max_abs_tol,
        mean_abs_tol,
    )

    all_ok = encoder_ok and decoder_ok and native_pipeline_ok
    print("\n" + "=" * 60)
    print(f"校验总结果: {'通过' if all_ok else '失败'}")
    print("=" * 60)
    return all_ok


def load_speech_tokenizer(model_path: str, device: torch.device, dtype: torch.dtype = torch.float32) -> Any:
    """加载 12Hz speech tokenizer，兼容不同 Qwen3-TTS 包版本的属性命名。

    有些版本会把 tokenizer 挂在 processor.speech_tokenizer；
    有些版本只有 processor.audio_tokenizer，甚至该属性为 None。
    对本地 ModelScope 目录，真正的 tokenizer 权重通常在 speech_tokenizer/ 子目录。

    Args:
        model_path: Qwen3-TTS 主模型目录或 HuggingFace 模型名。
        device: tokenizer 模型加载后移动到的 torch 设备。

    Returns:
        已加载并切到 eval 模式的 Qwen3TTSTokenizer 对象。
    """
    print(f"Loading processor: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, fix_mistral_regex=True)

    speech_tokenizer = getattr(processor, "speech_tokenizer", None)
    if speech_tokenizer is None:
        speech_tokenizer = getattr(processor, "audio_tokenizer", None)

    if speech_tokenizer is None:
        tokenizer_dir = os.path.join(model_path, "speech_tokenizer")
        if not os.path.isdir(tokenizer_dir):
            raise FileNotFoundError(
                "Cannot find speech tokenizer on processor or at "
                f"{tokenizer_dir}"
            )
        print(f"Processor has no loaded speech tokenizer; loading: {tokenizer_dir}")
        speech_tokenizer = Qwen3TTSTokenizer.from_pretrained(
            tokenizer_dir,
            device_map=str(device),
        )

    speech_tokenizer.model.to(device=device, dtype=dtype)
    speech_tokenizer.model.eval()
    return speech_tokenizer


def main_tokenizer() -> None:
    """命令行入口：负责解析参数、加载 tokenizer、执行导出和可选校验。

    Args:
        无，参数全部来自命令行。

    Returns:
        无返回值；校验失败时通过 SystemExit(1) 退出。
    """
    parser = argparse.ArgumentParser(
        description="Export only Qwen3-TTS 12Hz speech tokenizer ONNX models"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./qwen3-tts-0.6b-12hz",
        help="Output directory for ONNX models",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--opset-version", type=int, default=14)
    parser.add_argument(
        "--decode-trace-frames",
        type=int,
        default=100,
        help=(
            "Dummy codec frame count used when exporting tokenizer12hz_decode.onnx. "
            "With the current legacy exporter this also acts as the practical full-decoder output limit."
        ),
    )
    parser.add_argument(
        "--only-decode",
        action="store_true",
        help="Only export tokenizer12hz_decode.onnx; leave encoder untouched.",
    )
    parser.add_argument(
        "--export-chunk-decoder",
        action="store_true",
        help="Also export tokenizer12hz_decode_chunk.onnx for explicit chunk/pipeline experiments.",
    )
    parser.add_argument(
        "--only-chunk-decoder",
        action="store_true",
        help="Only export tokenizer12hz_decode_chunk.onnx; leave encoder and full decoder untouched.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=300,
        help="Current codec frames used when tracing tokenizer12hz_decode_chunk.onnx.",
    )
    parser.add_argument(
        "--left-context-size",
        type=int,
        default=25,
        help="Left-context codec frames used when tracing tokenizer12hz_decode_chunk.onnx.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip ONNX export and only run requested verification",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run PyTorch vs ONNX verification after export or with --skip-export",
    )
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument(
        "--verify-seconds",
        type=float,
        default=2.0,
        help="Seconds of synthetic 24 kHz audio used for verification",
    )
    parser.add_argument(
        "--verify-min-corr",
        type=float,
        default=0.999,
        help="Minimum acceptable waveform correlation for decoder verification",
    )
    parser.add_argument(
        "--verify-max-abs-tol",
        type=float,
        default=1e-2,
        help="Maximum acceptable absolute waveform difference",
    )
    parser.add_argument(
        "--verify-mean-abs-tol",
        type=float,
        default=1e-3,
        help="Maximum acceptable mean absolute waveform difference",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

    device = torch.device(args.device)
    dtype = parse_torch_dtype(args.dtype)

    speech_tokenizer = load_speech_tokenizer(args.model, device, dtype)
    print_torch_dtype_summary(speech_tokenizer.model, "Qwen3-TTS speech tokenizer")

    if not args.skip_export:
        if args.only_chunk_decoder:
            export_tokenizer_12hz_decode_chunk(
                speech_tokenizer,
                args.output_dir,
                device,
                opset_version=18,
                trace_chunk_frames=args.chunk_size,
                trace_context_frames=args.left_context_size,
            )
        else:
            if not args.only_decode:
                export_tokenizer_12hz_encode(
                    speech_tokenizer, args.output_dir, device, args.opset_version
                )
            export_tokenizer_12hz_decode(
                speech_tokenizer,
                args.output_dir,
                device,
                opset_version=18,
                trace_frames=args.decode_trace_frames,
            )
        if args.export_chunk_decoder and not args.only_chunk_decoder:
            export_tokenizer_12hz_decode_chunk(
                speech_tokenizer,
                args.output_dir,
                device,
                opset_version=18,
                trace_chunk_frames=args.chunk_size,
                trace_context_frames=args.left_context_size,
            )

    if not args.only_chunk_decoder:
        print_expected_inputs(
            "tokenizer12hz_encode.onnx",
            [("audio", "float32", "[1, num_samples]")],
        )
        print_onnx_io_dtypes(
            os.path.join(args.output_dir, "tokenizer12hz_encode.onnx"),
            "tokenizer12hz_encode.onnx",
        )
        print_expected_inputs(
            "tokenizer12hz_decode.onnx",
            [("audio_codes", "int64", "[1, codes_length, 16]")],
        )
        print_onnx_io_dtypes(
            os.path.join(args.output_dir, "tokenizer12hz_decode.onnx"),
            "tokenizer12hz_decode.onnx",
        )
    chunk_path = os.path.join(args.output_dir, "tokenizer12hz_decode_chunk.onnx")
    if os.path.exists(chunk_path):
        print_expected_inputs(
            "tokenizer12hz_decode_chunk.onnx",
            [
                ("audio_codes", "int64", "[1, codes_length, 16]"),
                ("context_frames", "int64", "scalar"),
            ],
        )
        print_onnx_io_dtypes(chunk_path, "tokenizer12hz_decode_chunk.onnx")

    if args.verify and not args.skip_verify:
        ok = verify_tokenizer_12hz(
            speech_tokenizer,
            args.output_dir,
            device,
            args.verify_seconds,
            args.verify_min_corr,
            args.verify_max_abs_tol,
            args.verify_mean_abs_tol,
        )
        if not ok:
            raise SystemExit(1)

    print("Tokenizer 12Hz ONNX export complete.")



# ---- external data consolidation ----
"""把每个 ONNX 模型的外部权重文件合并成一个 .data 文件。

大模型 ONNX 导出后经常会拆出很多 external data 文件。这个脚本把同一个
.onnx 旁边的外部权重合并为 `<model>.onnx.data`，目录更干净，也更方便发布。
"""

import argparse
from pathlib import Path

import onnx
from onnx import external_data_helper


def _external_locations(model):
    """收集 ONNX initializer 中引用的 external_data location。"""
    locations = set()
    for init in model.graph.initializer:
        if init.data_location == onnx.TensorProto.EXTERNAL:
            for entry in init.external_data:
                if entry.key == "location":
                    locations.add(entry.value)
    return locations


def consolidate_onnx(onnx_path: Path, remove_old: bool) -> tuple[int, int]:
    """合并单个 .onnx 的外部数据，返回 old/new 外部文件数量。"""
    model_dir = onnx_path.parent
    model = onnx.load(onnx_path, load_external_data=True)
    old_model = onnx.load(onnx_path, load_external_data=False)
    old_locations = _external_locations(old_model)

    if not old_locations:
        # 没有 external data 的小模型直接跳过。
        return 0, 0

    data_name = onnx_path.name + ".data"
    data_path = model_dir / data_name
    backup_data_path = model_dir / (data_name + ".bak_consolidate")
    tmp_onnx_path = model_dir / (onnx_path.name + ".tmp_consolidate")

    external_data_helper.convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=0,
        convert_attribute=False,
    )

    # 如果目标 .onnx.data 已经存在，onnx.save 可能会在旧文件后继续追加数据。
    # 先把旧文件挪走，再写新的 .data，避免已经合并过的模型越合并越大。
    if backup_data_path.exists():
        backup_data_path.unlink()
    if tmp_onnx_path.exists():
        tmp_onnx_path.unlink()
    if data_path.exists():
        data_path.replace(backup_data_path)

    try:
        onnx.save(model, tmp_onnx_path)
        tmp_onnx_path.replace(onnx_path)
        if backup_data_path.exists():
            backup_data_path.unlink()
    except Exception:
        if tmp_onnx_path.exists():
            tmp_onnx_path.unlink()
        if data_path.exists():
            data_path.unlink()
        if backup_data_path.exists():
            backup_data_path.replace(data_path)
        raise

    removed = 0
    if remove_old:
        for location in old_locations:
            if location == data_name:
                continue
            candidate = model_dir / location
            if candidate.exists() and candidate.is_file():
                candidate.unlink()
                removed += 1

    return len(old_locations), removed


def main_consolidate():
    # root 可以是 onnx_isolated 或 onnx_isolated_fp16，脚本会递归查找 .onnx。
    parser = argparse.ArgumentParser(description="Consolidate ONNX external data files")
    parser.add_argument("--root", type=str, default="./onnx_isolated")
    parser.add_argument("--remove-old", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    for onnx_path in sorted(root.glob("*/*.onnx")):
        external_count, removed = consolidate_onnx(onnx_path, args.remove_old)
        if external_count:
            print(
                f"{onnx_path}: consolidated {external_count} external file(s), "
                f"removed {removed}"
            )
        else:
            print(f"{onnx_path}: no external data")



# ---- decoder fp32 island patch ----
"""Patch tokenizer12hz decoder ONNX graphs with small FP32 compute islands.

The Qwen3-TTS tokenizer decoder can be sensitive to FP16/CUDA execution in a
few spots. This script inserts local Cast nodes around selected operations so
the rest of the model can stay FP16.
"""


import argparse
import shutil
from pathlib import Path
from typing import Iterable

import onnx
from onnx import TensorProto, helper


DECODER_REL_PATH = Path("tokenizer12hz") / "tokenizer12hz_decode.onnx"


class GraphPatcher:
    def __init__(self, model: onnx.ModelProto) -> None:
        self.model = model
        self.graph = model.graph
        self.nodes = list(self.graph.node)
        self.output_to_node = {out: node for node in self.nodes for out in node.output}
        self.used_names = {
            name
            for node in self.nodes
            for name in [node.name, *node.input, *node.output]
            if name
        }
        self.used_names.update(inp.name for inp in self.graph.input)
        self.used_names.update(out.name for out in self.graph.output)
        self.used_names.update(init.name for init in self.graph.initializer)

    def unique_name(self, base: str) -> str:
        clean = base.replace(":", "_")
        candidate = clean
        idx = 0
        while candidate in self.used_names:
            idx += 1
            candidate = f"{clean}_{idx}"
        self.used_names.add(candidate)
        return candidate

    def cast_node(self, src: str, dst_type: int, tag: str) -> tuple[onnx.NodeProto, str]:
        dst = self.unique_name(f"{src}_{tag}")
        node_name = self.unique_name(f"{src}_{tag}_Cast")
        node = helper.make_node("Cast", [src], [dst], name=node_name, to=dst_type)
        return node, dst

    def patch_node_to_float32(
        self,
        target: onnx.NodeProto,
        input_indices: Iterable[int],
        output_type: int,
        tag: str,
    ) -> tuple[list[onnx.NodeProto], list[onnx.NodeProto]]:
        before: list[onnx.NodeProto] = []
        after: list[onnx.NodeProto] = []

        for input_index in input_indices:
            cast, cast_output = self.cast_node(
                target.input[input_index], TensorProto.FLOAT, f"{tag}_to_fp32"
            )
            before.append(cast)
            target.input[input_index] = cast_output

        if len(target.output) != 1:
            raise ValueError(f"Expected one output for {target.name}, got {len(target.output)}")

        original_output = target.output[0]
        fp32_output = self.unique_name(f"{original_output}_{tag}_fp32")
        target.output[0] = fp32_output
        cast_back = helper.make_node(
            "Cast",
            [fp32_output],
            [original_output],
            name=self.unique_name(f"{original_output}_{tag}_back_Cast"),
            to=output_type,
        )
        after.append(cast_back)
        return before, after

    def replace_nodes(self, replacements: dict[str, tuple[list[onnx.NodeProto], list[onnx.NodeProto]]]) -> None:
        new_nodes: list[onnx.NodeProto] = []
        for node in self.nodes:
            before, after = replacements.get(node.name, ([], []))
            new_nodes.extend(before)
            new_nodes.append(node)
            new_nodes.extend(after)
        del self.graph.node[:]
        self.graph.node.extend(new_nodes)
        self.nodes = list(self.graph.node)
        self.output_to_node = {out: node for node in self.nodes for out in node.output}

    def find_swiglu_final_muls(self) -> list[onnx.NodeProto]:
        result: list[onnx.NodeProto] = []
        for node in self.nodes:
            if node.op_type != "Mul":
                continue
            name = node.name or ""
            if "/mlp/Mul" not in name or "/act_fn/" in name:
                continue
            if not any("/mlp/act_fn/Mul" in inp for inp in node.input):
                continue
            if not any("/mlp/up_proj/MatMul" in inp for inp in node.input):
                continue
            result.append(node)
        return result

    def find_down_proj_matmul(self, swiglu_mul: onnx.NodeProto) -> onnx.NodeProto | None:
        swiglu_output = swiglu_mul.output[0]
        consumers = [
            node
            for node in self.nodes
            if node.op_type == "MatMul" and swiglu_output in list(node.input)
        ]
        if len(consumers) != 1:
            return None
        consumer = consumers[0]
        if "/mlp/down_proj/MatMul" not in (consumer.name or ""):
            return None
        return consumer

    def patch_reduce_sum_lengths(self) -> int:
        replacements: dict[str, tuple[list[onnx.NodeProto], list[onnx.NodeProto]]] = {}
        patched = 0
        for node in self.nodes:
            if node.op_type != "ReduceSum":
                continue
            if len(node.input) < 1:
                continue
            if (node.name or "") != "/ReduceSum" and "length" not in " ".join(node.output).lower():
                continue
            before, after = self.patch_node_to_float32(
                node, input_indices=[0], output_type=TensorProto.INT64, tag="fp32_reduce"
            )
            replacements[node.name] = (before, after)
            patched += 1
        self.replace_nodes(replacements)
        return patched

    def patch_swiglu_mul(self) -> int:
        replacements: dict[str, tuple[list[onnx.NodeProto], list[onnx.NodeProto]]] = {}
        for node in self.find_swiglu_final_muls():
            before, after = self.patch_node_to_float32(
                node, input_indices=[0, 1], output_type=TensorProto.FLOAT16, tag="fp32_swiglu"
            )
            replacements[node.name] = (before, after)
        self.replace_nodes(replacements)
        return len(replacements)

    def patch_swiglu_down_proj(self) -> int:
        replacements: dict[str, tuple[list[onnx.NodeProto], list[onnx.NodeProto]]] = {}
        patched = 0

        for swiglu in self.find_swiglu_final_muls():
            down_proj = self.find_down_proj_matmul(swiglu)
            if down_proj is None:
                continue

            before: list[onnx.NodeProto] = []
            swiglu_after: list[onnx.NodeProto] = []
            for input_index in [0, 1]:
                cast, cast_output = self.cast_node(
                    swiglu.input[input_index], TensorProto.FLOAT, "fp32_swiglu_to_fp32"
                )
                before.append(cast)
                swiglu.input[input_index] = cast_output

            original_swiglu_output = swiglu.output[0]
            fp32_swiglu_output = self.unique_name(f"{original_swiglu_output}_fp32_swiglu")
            swiglu.output[0] = fp32_swiglu_output

            for input_index, input_name in enumerate(down_proj.input):
                if input_name == original_swiglu_output:
                    down_proj.input[input_index] = fp32_swiglu_output
                else:
                    cast, cast_output = self.cast_node(
                        input_name, TensorProto.FLOAT, "fp32_down_proj_to_fp32"
                    )
                    replacements.setdefault(down_proj.name, ([], []))[0].append(cast)
                    down_proj.input[input_index] = cast_output

            original_down_output = down_proj.output[0]
            fp32_down_output = self.unique_name(f"{original_down_output}_fp32_down_proj")
            down_proj.output[0] = fp32_down_output
            cast_back = helper.make_node(
                "Cast",
                [fp32_down_output],
                [original_down_output],
                name=self.unique_name(f"{original_down_output}_fp32_down_proj_back_Cast"),
                to=TensorProto.FLOAT16,
            )
            replacements.setdefault(down_proj.name, ([], []))[1].append(cast_back)
            replacements[swiglu.name] = (before, swiglu_after)
            patched += 1

        self.replace_nodes(replacements)
        return patched


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.onnx_root:
        input_root = Path(args.onnx_root)
        output_root = Path(args.output_root) if args.output_root else input_root.with_name(input_root.name + "_fp32_islands")
        return input_root / DECODER_REL_PATH, output_root / DECODER_REL_PATH

    if not args.input:
        raise SystemExit("Either --onnx-root or --input is required")
    input_model = Path(args.input)
    output_model = Path(args.output) if args.output else input_model.with_name(input_model.stem + "_fp32_islands.onnx")
    return input_model, output_model


def prepare_output_tree(args: argparse.Namespace, input_model: Path, output_model: Path) -> None:
    if args.onnx_root:
        input_root = Path(args.onnx_root)
        output_root = output_model.parents[1]
        if output_root.exists() and not args.overwrite:
            raise SystemExit(f"Output root exists: {output_root}. Use --overwrite to replace it.")
        if output_root.exists():
            shutil.rmtree(output_root)
        shutil.copytree(input_root, output_root)
    else:
        output_model.parent.mkdir(parents=True, exist_ok=True)
        if output_model.exists() and not args.overwrite:
            raise SystemExit(f"Output file exists: {output_model}. Use --overwrite to replace it.")


def patch_model(args: argparse.Namespace) -> None:
    input_model, output_model = resolve_paths(args)
    if not input_model.exists():
        raise SystemExit(f"Input model not found: {input_model}")

    prepare_output_tree(args, input_model, output_model)
    model = onnx.load(str(input_model), load_external_data=True)
    patcher = GraphPatcher(model)

    reduce_count = 0
    swiglu_count = 0
    if args.patch_reduce_sum:
        reduce_count = patcher.patch_reduce_sum_lengths()
    if args.patch_swiglu:
        if args.swiglu_mode == "mul":
            swiglu_count = patcher.patch_swiglu_mul()
        elif args.swiglu_mode == "down-proj":
            swiglu_count = patcher.patch_swiglu_down_proj()
        else:
            raise ValueError(args.swiglu_mode)

    onnx.checker.check_model(model)
    onnx.save_model(
        model,
        str(output_model),
        save_as_external_data=False,
    )

    print(f"wrote: {output_model}")
    print(f"patched ReduceSum nodes: {reduce_count}")
    print(f"patched SwiGLU blocks: {swiglu_count} ({args.swiglu_mode})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--onnx-root", help="ONNX root containing tokenizer12hz/tokenizer12hz_decode.onnx")
    source.add_argument("--input", help="Input tokenizer12hz_decode.onnx")
    parser.add_argument("--output-root", help="Output ONNX root. Defaults to <onnx-root>_fp32_islands")
    parser.add_argument("--output", help="Output ONNX model. Defaults to *_fp32_islands.onnx")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output path")
    parser.add_argument("--patch-reduce-sum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patch-swiglu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--swiglu-mode",
        choices=["mul", "down-proj"],
        default="down-proj",
        help="Patch only the SwiGLU Mul, or keep the SwiGLU product in FP32 through down_proj MatMul",
    )
    return parser


def main_patch_decoder() -> None:
    patch_model(build_parser().parse_args())



# ---- all-submodel export orchestrator ----
"""把所有独立 ONNX 子模型导出到隔离目录。

这个脚本是“总控导出器”：它顺序调用其它 verify/export 脚本，把 Qwen3-TTS
拆成 runtime 需要的多个子模型目录。默认 float32 会导出并校验，float16
通常跳过校验以节省时间。
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_MODEL = "/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def run_step(name: str, cmd: list[str]) -> None:
    """打印并执行一个导出步骤；失败时 subprocess.run(check=True) 会中断全流程。"""
    print("\n" + "=" * 80, flush=True)
    print(f"[{name}]", flush=True)
    print(" ".join(cmd), flush=True)
    print("=" * 80, flush=True)
    subprocess.run(cmd, check=True)


def main_all() -> None:
    # common 是所有导出脚本共享的模型目录、设备和 dtype 参数。
    parser = argparse.ArgumentParser(description="Export all ONNX models into isolated directories")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=str, default="./onnx_isolated")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--clean", action="store_true", help="Delete output-root before exporting")
    parser.add_argument(
        "--skip-speaker-encoder",
        action="store_true",
        help="Skip speaker_encoder.onnx export. CustomVoice/VoiceDesign models do not use the Base voice-clone speaker encoder.",
    )
    parser.add_argument(
        "--with-chunk-decoder",
        action="store_true",
        help="Also export tokenizer12hz_decode_chunk.onnx for chunk/pipeline runtime.",
    )
    parser.add_argument(
        "--decode-trace-frames",
        type=int,
        default=100,
        help="Codec frame count used when tracing tokenizer12hz_decode.onnx. This is the practical full-decoder output limit.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=300,
        help="Codec frame count used when tracing tokenizer12hz_decode_chunk.onnx.",
    )
    parser.add_argument(
        "--left-context-size",
        type=int,
        default=25,
        help="Left-context codec frame count used when tracing tokenizer12hz_decode_chunk.onnx.",
    )
    args = parser.parse_args()

    root = Path(args.output_root)
    if args.clean and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    py = sys.executable
    common = ["--model", args.model, "--device", args.device, "--dtype", args.dtype]
    verify_args = [] if args.dtype == "float32" else ["--skip-verify"]
    talker_verify_args = ["--atol", "2e-3"] if args.dtype == "float32" else []

    run_step(
        "tokenizer12hz",
        [
            py,
            str(script_dir / "export.py"),
            "tokenizer",
            *common,
            "--output-dir",
            str(root / "tokenizer12hz"),
            "--decode-trace-frames",
            str(args.decode_trace_frames),
            *(["--verify"] if args.dtype == "float32" else []),
        ],
    )

    if args.with_chunk_decoder:
        run_step(
            "tokenizer12hz_chunk_decoder",
            [
                py,
                str(script_dir / "export.py"),
            "tokenizer",
                *common,
                "--output-dir",
                str(root / "tokenizer12hz"),
                "--only-chunk-decoder",
                "--chunk-size",
                str(args.chunk_size),
                "--left-context-size",
                str(args.left_context_size),
            ],
        )

    run_step(
        "text_project",
        [
            py,
            str(script_dir / "verify.py"),
            "embed",
            *common,
            "--output-dir",
            str(root / "text_project"),
            "--only",
            "text_project",
            *verify_args,
        ],
    )

    run_step(
        "codec_embed",
        [
            py,
            str(script_dir / "verify.py"),
            "embed",
            *common,
            "--output-dir",
            str(root / "codec_embed"),
            "--only",
            "codec_embed",
            *verify_args,
        ],
    )

    run_step(
        "code_predictor_embed",
        [
            py,
            str(script_dir / "verify.py"),
            "code-predictor-embed",
            *common,
            "--output-dir",
            str(root / "code_predictor_embed"),
            *verify_args,
        ],
    )

    run_step(
        "code_predictor",
        [
            py,
            str(script_dir / "verify.py"),
            "code-predictor",
            *common,
            "--output-dir",
            str(root / "code_predictor"),
            *verify_args,
        ],
    )

    run_step(
        "talker_prefill",
        [
            py,
            str(script_dir / "verify.py"),
            "talker-prefill",
            *common,
            "--output-dir",
            str(root / "talker_prefill"),
            "--seq-len",
            "1",
            "--extra-seq-lens",
            "2,3,4,5,7,8,9,12,16,20,24,32,48,64",
            *talker_verify_args,
            *verify_args,
        ],
    )

    run_step(
        "talker_decode",
        [
            py,
            str(script_dir / "verify.py"),
            "talker-decode",
            *common,
            "--output-dir",
            str(root / "talker_decode"),
            "--past-len",
            "1",
            "--extra-past-lens",
            "2,3,4,5,7,8,9,12,16,20,24,32,48,64",
            *talker_verify_args,
            *verify_args,
        ],
    )

    if not args.skip_speaker_encoder:
        run_step(
            "speaker_encoder",
            [
                py,
                str(script_dir / "verify.py"),
            "speaker-encoder",
                *common,
                "--output-dir",
                str(root / "speaker_encoder"),
                *verify_args,
            ],
        )

    print("\nAll isolated ONNX exports and verifications completed.", flush=True)
    print(f"DType: {args.dtype}", flush=True)
    print(f"Output root: {root.resolve()}", flush=True)




def main() -> None:
    import sys
    commands = {
        "all": main_all,
        "tokenizer": main_tokenizer,
        "consolidate": main_consolidate,
        "patch-decoder": main_patch_decoder,
    }
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        available = ", ".join(sorted(commands))
        print(f"usage: {Path(sys.argv[0]).name} <{available}> [args...]", file=sys.stderr)
        raise SystemExit(2)
    command = sys.argv[1]
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    commands[command]()


if __name__ == "__main__":
    main()
