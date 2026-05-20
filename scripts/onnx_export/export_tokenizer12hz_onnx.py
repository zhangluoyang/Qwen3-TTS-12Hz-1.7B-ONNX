#!/usr/bin/env python3
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

from onnx_dtype_utils import print_expected_inputs, print_onnx_io_dtypes, print_torch_dtype_summary


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
    trace_chunk_frames: int = 50,
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


def main() -> None:
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
        default=50,
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


if __name__ == "__main__":
    main()
