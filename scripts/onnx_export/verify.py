#!/usr/bin/env python3
"""Merged ONNX export-time verification utilities for Qwen3-TTS."""

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

# ---- talker prefill verification ----
"""导出并校验 talker_prefill.onnx。

talker_prefill 负责把完整 prompt 一次性送进 talker，产出：
logits、last_hidden，以及每一层的 past_key/past_value。
后续 talker_decode 会复用这些 KV cache 做逐帧自回归生成。
"""

import argparse
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoProcessor

from qwen_tts.core.models import (
    Qwen3TTSConfig,
    Qwen3TTSForConditionalGeneration,
    Qwen3TTSProcessor,
)


def parse_torch_dtype(dtype: str) -> torch.dtype:
    """把命令行 dtype 字符串转换成 torch dtype。"""
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype}")


def _get_talker_head(talker: nn.Module) -> nn.Module:
    """兼容不同版本 Qwen3-TTS talker 的输出 head 命名。"""
    if hasattr(talker, "lm_head"):
        return talker.lm_head
    if hasattr(talker, "codec_head"):
        return talker.codec_head
    raise AttributeError("talker has neither lm_head nor codec_head")


def export_talker_prefill_compat(model, output_dir, opset_version=14):
    """把 PyTorch talker prefill 包装成 ONNX 子图。"""
    print("Exporting talker_prefill.onnx ...")
    talker = model.talker
    num_layers = talker.config.num_hidden_layers
    d_model = talker.config.hidden_size
    head = _get_talker_head(talker)

    class TalkerPrefill(nn.Module):
        def __init__(self, talker_module: nn.Module, head_module: nn.Module):
            super().__init__()
            self.talker = talker_module
            self.head = head_module

        def forward(self, inputs_embeds, attention_mask):
            out = self.talker.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            hidden = out.last_hidden_state
            logits = self.head(hidden[:, -1:, :])
            pkv = out.past_key_values
            return (logits, hidden) + tuple(t for kv in pkv for t in kv)

    wrapper = TalkerPrefill(talker, head).eval()
    t = 8
    model_dtype = next(talker.parameters()).dtype
    dummy_embeds = torch.randn(1, t, d_model, dtype=model_dtype, device=model.device)
    dummy_mask = torch.ones(1, t, dtype=torch.long, device=model.device)

    kv_names = []
    for i in range(num_layers):
        kv_names += [f"past_key_{i}", f"past_value_{i}"]
    kv_dynamic = {name: {2: "seq_len"} for name in kv_names}

    torch.onnx.export(
        wrapper,
        (dummy_embeds, dummy_mask),
        os.path.join(output_dir, "talker_prefill.onnx"),
        input_names=["inputs_embeds", "attention_mask"],
        output_names=["logits", "last_hidden"] + kv_names,
        dynamic_axes={
            "inputs_embeds": {1: "seq_len"},
            "attention_mask": {1: "seq_len"},
            "last_hidden": {1: "seq_len"},
            **kv_dynamic,
        },
        opset_version=opset_version,
        do_constant_folding=False,
        dynamo=False,
    )
    print(f"  Done: talker_prefill.onnx ({num_layers} layers, {len(kv_names)} KV tensors)")


def patch_talker_prefill_dynamic_reshape(onnx_path: str) -> int:
    """把固定的 Range->Reshape([trace_len,1]) 修补成动态 Unsqueeze(axis=1)。"""
    # 导出时某些位置会把 trace seq_len 固化到 reshape；这里改成依赖 Range 的动态长度。
    model = onnx.load(onnx_path, load_external_data=False)
    patched = 0
    new_nodes = []

    for node in model.graph.node:
        if node.name == "/model/Reshape_2":
            axes_name = "/model/Unsqueeze_axis1_const_output_0"
            unsqueeze_out = "/model/Range_unsqueeze_axis1_output_0"
            new_nodes.append(
                onnx.helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=[axes_name],
                    name="/model/Unsqueeze_axis1_const",
                    value=onnx.helper.make_tensor("value", onnx.TensorProto.INT64, [1], [1]),
                )
            )
            new_nodes.append(
                onnx.helper.make_node(
                    "Unsqueeze",
                    inputs=["/model/Range_output_0", axes_name],
                    outputs=[unsqueeze_out],
                    name="/model/Range_unsqueeze_axis1",
                )
            )
            patched += 1
            continue

        if node.name == "/model/LessOrEqual":
            if len(node.input) > 1 and node.input[1] == "/model/Reshape_2_output_0":
                node.input[1] = "/model/Range_unsqueeze_axis1_output_0"

        new_nodes.append(node)

    if patched:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
        # 带外部权重的大模型在 checker 序列化时可能超过 protobuf 限制；
        # 这里跳过 checker，依赖后面的运行时校验。
        onnx.save(model, onnx_path)
        print(f"  Patched talker_prefill dynamic reshape nodes: {patched}")
    else:
        already_patched = any(
            node.name == "/model/Range_unsqueeze_axis1" for node in model.graph.node
        )
        if already_patched:
            print("  Patch skipped: talker_prefill dynamic reshape already patched")
        else:
            print("  Patch skipped: target node /model/Reshape_2 not found")

    return patched


def compare_tensor(name, reference, actual, max_abs_tol, mean_abs_tol):
    """打印并判断 PyTorch reference 与 ONNX output 的误差。"""
    ref = np.asarray(reference, dtype=np.float64)
    got = np.asarray(actual, dtype=np.float64)
    shape_ok = ref.shape == got.shape

    if shape_ok and ref.size:
        diff = got - ref
        max_abs = float(np.max(np.abs(diff)))
        mean_abs = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff * diff)))
    else:
        max_abs = float("inf")
        mean_abs = float("inf")
        rmse = float("inf")

    ok = shape_ok and max_abs <= max_abs_tol and mean_abs <= mean_abs_tol
    print(f"  {name}:")
    print(f"    reference shape: {ref.shape}")
    print(f"    actual shape:    {got.shape}")
    print(f"    shape match:     {shape_ok}")
    print(f"    max_abs_diff:    {max_abs:.8f}")
    print(f"    mean_abs_diff:   {mean_abs:.8f}")
    print(f"    rmse:            {rmse:.8f}")
    print(f"    pass:            {ok}")
    return ok, max_abs, mean_abs


def verify_talker_prefill(model, session, onnx_path, seq_len=12, atol=5e-4, rtol=1e-3):
    """随机生成 inputs_embeds/mask，比较 PyTorch 和 ONNX prefill 输出。"""
    talker = model.talker
    d_model = talker.config.hidden_size
    num_layers = talker.config.num_hidden_layers

    inputs_embeds = torch.randn(1, seq_len, d_model, dtype=torch.float32, device=model.device)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=model.device)

    with torch.no_grad():
        pt_out = talker.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        pt_hidden = pt_out.last_hidden_state
        pt_logits = _get_talker_head(talker)(pt_hidden[:, -1:, :])
        pt_kv_flat = tuple(t for kv in pt_out.past_key_values for t in kv)

    try:
        ort_outputs = session.run(
            None,
            {
                "inputs_embeds": inputs_embeds.detach().cpu().numpy(),
                "attention_mask": attention_mask.detach().cpu().numpy().astype(np.int64),
            },
        )
    except Exception as exc:
        print(f"\n[Verify] talker_prefill.onnx seq_len={seq_len}")
        print(f"  onnx path: {onnx_path}")
        print(f"  onnxruntime error: {exc}")
        print("  result: FAIL")
        return False

    pt_logits_np = pt_logits.detach().cpu().numpy()
    pt_hidden_np = pt_hidden.detach().cpu().numpy()

    print(f"\n[Verify] talker_prefill.onnx seq_len={seq_len}")
    print(f"  onnx path: {onnx_path}")
    print(f"  layers: {num_layers}")
    logits_ok, _, _ = compare_tensor("logits compare", pt_logits_np, ort_outputs[0], atol, atol)
    hidden_ok, _, _ = compare_tensor("hidden compare", pt_hidden_np, ort_outputs[1], atol, atol)

    kv_shape_ok = True
    kv_value_ok = True
    kv_max_abs = 0.0
    kv_mean_abs = 0.0

    for i, pt_tensor in enumerate(pt_kv_flat):
        ort_tensor = ort_outputs[2 + i]
        pt_np = pt_tensor.detach().cpu().numpy()

        if tuple(pt_np.shape) != tuple(ort_tensor.shape):
            kv_shape_ok = False

        abs_diff = np.abs(pt_np.astype(np.float64) - ort_tensor.astype(np.float64))
        max_abs = float(np.max(abs_diff))
        mean_abs = float(np.mean(abs_diff))
        kv_max_abs = max(kv_max_abs, max_abs)
        kv_mean_abs = max(kv_mean_abs, mean_abs)

        if max_abs > atol or mean_abs > atol:
            kv_value_ok = False

    ok = logits_ok and hidden_ok and kv_shape_ok and kv_value_ok

    print(f"  kv shape ok: {kv_shape_ok}")
    print(f"  kv value close: {kv_value_ok}")
    print(f"  kv max_abs_diff: {kv_max_abs:.8f}")
    print(f"  kv max_mean_abs_diff: {kv_mean_abs:.8f}")
    print(f"  result: {'PASS' if ok else 'FAIL'}")

    return ok


def main_talker_prefill():
    parser = argparse.ArgumentParser(description="Standalone talker_prefill ONNX export/verify")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--output-dir", type=str, default="./qwen3-tts-0.6b-12hz")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--opset-version", type=int, default=14)
    parser.add_argument("--skip-export", action="store_true", help="Skip exporting talker_prefill.onnx")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument(
        "--skip-patch",
        action="store_true",
        help="Skip ONNX graph patch for dynamic Range->Unsqueeze",
    )
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument(
        "--extra-seq-lens",
        type=str,
        default="",
        help="Comma-separated extra seq lens for verification, e.g. 8,12,20",
    )
    parser.add_argument("--atol", type=float, default=5e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

    model = AutoModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=parse_torch_dtype(args.dtype),
    )
    model.eval()
    print_torch_dtype_summary(model, "Qwen3-TTS")

    if not args.skip_export:
        export_talker_prefill_compat(model, args.output_dir, args.opset_version)

    onnx_path = os.path.join(args.output_dir, "talker_prefill.onnx")
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(
            f"Cannot verify: missing {onnx_path}. Remove --skip-export or provide a valid output dir."
        )

    if not args.skip_patch:
        patch_talker_prefill_dynamic_reshape(onnx_path)

    if args.skip_verify:
        print_onnx_io_dtypes(onnx_path, "talker_prefill.onnx")
        return

    seq_lens = [args.seq_len]
    if args.extra_seq_lens.strip():
        for x in args.extra_seq_lens.split(","):
            x = x.strip()
            if x:
                seq_lens.append(int(x))
    seq_lens = list(dict.fromkeys(seq_lens))

    print("\n开始校验 talker_prefill.onnx")
    print(f"  seq_lens: {seq_lens}")
    print(f"  atol={args.atol}, rtol={args.rtol}")
    print_expected_inputs(
        "talker_prefill.onnx",
        [
            ("inputs_embeds", "float32", f"[1, seq_len, {model.talker.config.hidden_size}]"),
            ("attention_mask", "int64", "[1, seq_len]"),
        ],
    )
    print_onnx_io_dtypes(onnx_path, "talker_prefill.onnx")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    all_ok = True
    for cur_len in seq_lens:
        ok = verify_talker_prefill(
            model,
            session,
            onnx_path,
            seq_len=cur_len,
            atol=args.atol,
            rtol=args.rtol,
        )
        all_ok = all_ok and ok
    print(f"\n校验总结果: {'通过' if all_ok else '失败'}")
    if not all_ok:
        raise RuntimeError("talker_prefill verification failed")



# ---- talker decode verification ----
"""导出并校验 talker_decode.onnx。

talker_decode 是自回归主循环的单步子图：输入当前 1 帧 embedding、
attention_mask、cache_position 和旧 KV cache，输出下一步 logits、
last_hidden 以及更新后的 KV cache。
"""

import argparse
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoProcessor
from transformers.cache_utils import DynamicCache

from qwen_tts.core.models import (
    Qwen3TTSConfig,
    Qwen3TTSForConditionalGeneration,
    Qwen3TTSProcessor,
)



def _flatten_cache(cache, num_layers):
    """把 Transformers DynamicCache 展平成 ONNX 输出需要的 key/value 列表。"""
    return tuple(t for i in range(num_layers) for t in (cache.layers[i].keys, cache.layers[i].values))


def _legacy_cache_from_flat(past_kv_flat, num_layers):
    """把 ONNX flat past_key/value 列表还原为 Transformers DynamicCache。"""
    legacy_cache = tuple(
        (past_kv_flat[2 * i], past_kv_flat[2 * i + 1]) for i in range(num_layers)
    )
    return DynamicCache.from_legacy_cache(legacy_cache)


def export_talker_decode_compat(model, output_dir, opset_version=14, trace_past_len=8):
    """导出单步 decode 子图；trace_past_len 只用于 tracing 样例长度。"""
    print("Exporting talker_decode.onnx ...")
    talker = model.talker
    num_layers = talker.config.num_hidden_layers
    d_model = talker.config.hidden_size
    num_kv_heads = talker.config.num_key_value_heads
    head_dim = getattr(talker.config, "head_dim", d_model // talker.config.num_attention_heads)
    head = _get_talker_head(talker)

    class TalkerDecode(nn.Module):
        def __init__(self, talker_module, head_module, layer_count):
            super().__init__()
            self.talker = talker_module
            self.head = head_module
            self.layer_count = layer_count

        def forward(self, inputs_embeds, attention_mask, cache_position, *past_kv_flat):
            past_key_values = _legacy_cache_from_flat(past_kv_flat, self.layer_count)
            out = self.talker.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )
            hidden = out.last_hidden_state
            logits = self.head(hidden)
            valid_cache_len = cache_position[-1] + 1
            new_kv = tuple(
                tensor[:, :, :valid_cache_len, :]
                for tensor in _flatten_cache(out.past_key_values, self.layer_count)
            )
            return (logits, hidden) + new_kv

    wrapper = TalkerDecode(talker, head, num_layers).eval()

    model_dtype = next(talker.parameters()).dtype
    dummy_embeds = torch.randn(1, 1, d_model, dtype=model_dtype, device=model.device)
    dummy_mask = torch.ones(1, trace_past_len + 2, dtype=torch.long, device=model.device)
    dummy_cache_position = torch.arange(
        trace_past_len, trace_past_len + 1, dtype=torch.long, device=model.device
    )
    dummy_pkv = [
        torch.randn(1, num_kv_heads, trace_past_len, head_dim, dtype=model_dtype, device=model.device)
        for _ in range(num_layers * 2)
    ]

    in_kv_names = []
    out_kv_names = []
    for i in range(num_layers):
        in_kv_names += [f"past_key_{i}", f"past_value_{i}"]
        out_kv_names += [f"new_past_key_{i}", f"new_past_value_{i}"]

    in_kv_dynamic = {name: {2: "past_len"} for name in in_kv_names}
    out_kv_dynamic = {name: {2: "new_len"} for name in out_kv_names}

    torch.onnx.export(
        wrapper,
        (dummy_embeds, dummy_mask, dummy_cache_position, *dummy_pkv),
        os.path.join(output_dir, "talker_decode.onnx"),
        input_names=["inputs_embeds", "attention_mask", "cache_position"] + in_kv_names,
        output_names=["logits", "last_hidden"] + out_kv_names,
        dynamic_axes={
            "attention_mask": {1: "full_len"},
            "cache_position": {0: "decode_len"},
            **in_kv_dynamic,
            **out_kv_dynamic,
        },
        opset_version=opset_version,
        do_constant_folding=False,
        dynamo=False,
    )
    print(f"  Done: talker_decode.onnx ({num_layers} layers, trace_past_len={trace_past_len})")


def patch_talker_decode_dynamic_reshape(onnx_path):
    """把 Range->Reshape([trace_len,1]) 掩码修补成动态 Unsqueeze(axis=1)。"""
    # 导出时某些 attention mask 相关 reshape 会固化 trace_past_len；
    # 这里把 Range 输出改成动态 Unsqueeze，支持多种 past_len 校验。
    model = onnx.load(onnx_path, load_external_data=False)
    new_nodes = []
    patched = 0

    for node in model.graph.node:
        if node.op_type == "Reshape" and node.name.startswith("/model/Reshape"):
            if len(node.input) >= 2 and node.input[0].startswith("/model/Range"):
                axes_name = f"{node.name}_unsqueeze_axis1_const_output_0"
                unsqueeze_out = f"{node.name}_range_unsqueeze_axis1_output_0"
                original_out = node.output[0]
                new_nodes.append(
                    onnx.helper.make_node(
                        "Constant",
                        inputs=[],
                        outputs=[axes_name],
                        name=f"{node.name}_unsqueeze_axis1_const",
                        value=onnx.helper.make_tensor("value", onnx.TensorProto.INT64, [1], [1]),
                    )
                )
                new_nodes.append(
                    onnx.helper.make_node(
                        "Unsqueeze",
                        inputs=[node.input[0], axes_name],
                        outputs=[unsqueeze_out],
                        name=f"{node.name}_range_unsqueeze_axis1",
                    )
                )
                for consumer in model.graph.node:
                    for idx, input_name in enumerate(consumer.input):
                        if input_name == original_out:
                            consumer.input[idx] = unsqueeze_out
                patched += 1
                continue

        new_nodes.append(node)

    if patched:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
        onnx.save(model, onnx_path)
        print(f"  Patched talker_decode dynamic reshape nodes: {patched}")
    else:
        print("  Patch skipped: no Range->Reshape mask nodes found")

    return patched


def _make_reference_inputs(model, past_len):
    """构造一组 PyTorch/ONNX 共用的随机 decode 输入。"""
    talker = model.talker
    d_model = talker.config.hidden_size
    inputs_prefill = torch.randn(1, past_len, d_model, dtype=torch.float32, device=model.device)
    prefill_mask = torch.ones(1, past_len, dtype=torch.long, device=model.device)
    decode_embed = torch.randn(1, 1, d_model, dtype=torch.float32, device=model.device)
    decode_mask = torch.ones(1, past_len + 2, dtype=torch.long, device=model.device)
    cache_position = torch.arange(past_len, past_len + 1, dtype=torch.long, device=model.device)

    with torch.no_grad():
        prefill_out = talker.model(
            inputs_embeds=inputs_prefill,
            attention_mask=prefill_mask,
            use_cache=True,
            return_dict=True,
        )
        decode_out = talker.model(
            inputs_embeds=decode_embed,
            attention_mask=decode_mask,
            past_key_values=prefill_out.past_key_values,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
        hidden = decode_out.last_hidden_state
        logits = _get_talker_head(talker)(hidden)

    return (
        decode_embed,
        decode_mask,
        cache_position,
        prefill_out.past_key_values,
        logits,
        hidden,
        decode_out.past_key_values,
    )


def verify_talker_decode(model, session, onnx_path, past_len, atol=5e-4):
    """比较一个 past_len 下 PyTorch decode 和 ONNX decode 输出。"""
    talker = model.talker
    num_layers = talker.config.num_hidden_layers
    decode_embed, decode_mask, cache_position, past_cache, pt_logits, pt_hidden, pt_new_cache = _make_reference_inputs(
        model, past_len
    )
    past_kv_flat = _flatten_cache(past_cache, num_layers)
    pt_new_kv_flat = _flatten_cache(pt_new_cache, num_layers)

    feed = {
        "inputs_embeds": decode_embed.detach().cpu().numpy(),
        "attention_mask": decode_mask.detach().cpu().numpy().astype(np.int64),
        "cache_position": cache_position.detach().cpu().numpy().astype(np.int64),
    }
    for i, tensor in enumerate(past_kv_flat):
        layer_idx = i // 2
        kind = "key" if i % 2 == 0 else "value"
        feed[f"past_{kind}_{layer_idx}"] = tensor.detach().cpu().numpy()

    try:
        ort_outputs = session.run(None, feed)
    except Exception as exc:
        print(f"\n[Verify] talker_decode.onnx past_len={past_len}")
        print(f"  onnx path: {onnx_path}")
        print(f"  onnxruntime error: {exc}")
        print("  result: FAIL")
        return False

    print(f"\n[Verify] talker_decode.onnx past_len={past_len}")
    print(f"  onnx path: {onnx_path}")
    print(f"  layers: {num_layers}")
    logits_ok, _, _ = compare_tensor(
        "logits compare", pt_logits.detach().cpu().numpy(), ort_outputs[0], atol, atol
    )
    hidden_ok, _, _ = compare_tensor(
        "hidden compare", pt_hidden.detach().cpu().numpy(), ort_outputs[1], atol, atol
    )

    kv_shape_ok = True
    kv_value_ok = True
    kv_max_abs = 0.0
    kv_mean_abs = 0.0
    for i, pt_tensor in enumerate(pt_new_kv_flat):
        ort_tensor = ort_outputs[2 + i]
        pt_np = pt_tensor.detach().cpu().numpy()
        if tuple(pt_np.shape) != tuple(ort_tensor.shape):
            kv_shape_ok = False
            continue
        abs_diff = np.abs(pt_np.astype(np.float64) - ort_tensor.astype(np.float64))
        max_abs = float(np.max(abs_diff))
        mean_abs = float(np.mean(abs_diff))
        kv_max_abs = max(kv_max_abs, max_abs)
        kv_mean_abs = max(kv_mean_abs, mean_abs)
        if max_abs > atol or mean_abs > atol:
            kv_value_ok = False

    ok = logits_ok and hidden_ok and kv_shape_ok and kv_value_ok
    print(f"  kv shape ok: {kv_shape_ok}")
    print(f"  kv value close: {kv_value_ok}")
    print(f"  kv max_abs_diff: {kv_max_abs:.8f}")
    print(f"  kv max_mean_abs_diff: {kv_mean_abs:.8f}")
    print(f"  result: {'PASS' if ok else 'FAIL'}")
    return ok


def _parse_ints(primary, extra):
    values = [primary]
    if extra.strip():
        values.extend(int(x.strip()) for x in extra.split(",") if x.strip())
    return list(dict.fromkeys(values))


def main_talker_decode():
    parser = argparse.ArgumentParser(description="Standalone talker_decode ONNX export/verify")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--output-dir", type=str, default="./qwen3-tts-0.6b-12hz")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--opset-version", type=int, default=14)
    parser.add_argument("--trace-past-len", type=int, default=8)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--skip-patch", action="store_true")
    parser.add_argument("--past-len", type=int, default=8)
    parser.add_argument("--extra-past-lens", type=str, default="")
    parser.add_argument("--atol", type=float, default=5e-4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

    model = AutoModel.from_pretrained(args.model, device_map=args.device, dtype=parse_torch_dtype(args.dtype))
    model.eval()
    print_torch_dtype_summary(model, "Qwen3-TTS")

    if not args.skip_export:
        export_talker_decode_compat(
            model,
            args.output_dir,
            opset_version=args.opset_version,
            trace_past_len=args.trace_past_len,
        )

    onnx_path = os.path.join(args.output_dir, "talker_decode.onnx")
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"Cannot verify: missing {onnx_path}")

    if not args.skip_patch:
        patch_talker_decode_dynamic_reshape(onnx_path)

    if args.skip_verify:
        print_onnx_io_dtypes(onnx_path, "talker_decode.onnx")
        return

    past_lens = _parse_ints(args.past_len, args.extra_past_lens)
    print("\n开始校验 talker_decode.onnx")
    print(f"  past_lens: {past_lens}")
    print(f"  atol={args.atol}")
    print_expected_inputs(
        "talker_decode.onnx",
        [
            ("inputs_embeds", "float32", f"[1, 1, {model.talker.config.hidden_size}]"),
            ("attention_mask", "int64", "[1, past_len + 2]"),
            ("cache_position", "int64", "[1]"),
            (
                "past_key_i",
                "float32",
                f"[1, {model.talker.config.num_key_value_heads}, past_len, "
                f"{getattr(model.talker.config, 'head_dim', model.talker.config.hidden_size // model.talker.config.num_attention_heads)}]",
            ),
            (
                "past_value_i",
                "float32",
                f"[1, {model.talker.config.num_key_value_heads}, past_len, "
                f"{getattr(model.talker.config, 'head_dim', model.talker.config.hidden_size // model.talker.config.num_attention_heads)}]",
            ),
        ],
    )
    print_onnx_io_dtypes(onnx_path, "talker_decode.onnx")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    all_ok = True
    for past_len in past_lens:
        ok = verify_talker_decode(model, session, onnx_path, past_len=past_len, atol=args.atol)
        all_ok = all_ok and ok

    print(f"\n校验总结果: {'通过' if all_ok else '失败'}")
    if not all_ok:
        raise RuntimeError("talker_decode verification failed")



# ---- embedding verification ----
"""导出并校验 text_project.onnx 和 codec_embed.onnx。

text_project 把文本 token ids 映射到 talker hidden size；
codec_embed 把第 0 个 codec codebook token 映射到同一 hidden size。
这两个小子图是构造 talker prompt 和 decode_embed 的基础。
"""

import argparse
import os

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoProcessor

from qwen_tts.core.models import (
    Qwen3TTSConfig,
    Qwen3TTSForConditionalGeneration,
    Qwen3TTSProcessor,
)



def _text_embedding(talker):
    """兼容不同版本 talker 文本 embedding 层命名。"""
    if hasattr(talker.model, "text_embed_tokens"):
        return talker.model.text_embed_tokens
    if hasattr(talker.model, "text_embedding"):
        return talker.model.text_embedding
    raise AttributeError("talker.model has neither text_embed_tokens nor text_embedding")


def _codec_embedding(talker):
    """兼容不同版本 talker codec embedding 层命名。"""
    if hasattr(talker.model, "embed_tokens"):
        return talker.model.embed_tokens
    if hasattr(talker.model, "codec_embedding"):
        return talker.model.codec_embedding
    raise AttributeError("talker.model has neither embed_tokens nor codec_embedding")


def export_text_project(model, output_dir, opset_version=14):
    """导出 text_project.onnx: input_ids -> text_embed。"""
    print("Exporting text_project.onnx ...")

    class TextProject(nn.Module):
        def __init__(self, talker):
            super().__init__()
            self.text_embed = _text_embedding(talker)
            self.text_projection = talker.text_projection

        def forward(self, input_ids):
            return self.text_projection(self.text_embed(input_ids))

    wrapper = TextProject(model.talker).eval()
    dummy_input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long, device=model.device)

    torch.onnx.export(
        wrapper,
        (dummy_input_ids,),
        os.path.join(output_dir, "text_project.onnx"),
        input_names=["input_ids"],
        output_names=["text_embed"],
        dynamic_shapes={"input_ids": {1: torch.export.Dim("seq_len")}},
        opset_version=opset_version,
        dynamo=True,
        external_data=True,
    )
    print("  Done: text_project.onnx")


def export_codec_embed(model, output_dir, opset_version=14):
    """导出 codec_embed.onnx: token_ids -> embed。"""
    print("Exporting codec_embed.onnx ...")

    class CodecEmbed(nn.Module):
        def __init__(self, talker):
            super().__init__()
            self.embed_tokens = _codec_embedding(talker)

        def forward(self, token_ids):
            return self.embed_tokens(token_ids)

    wrapper = CodecEmbed(model.talker).eval()
    dummy_ids = torch.tensor([[100, 101, 102, 103, 104]], dtype=torch.long, device=model.device)

    torch.onnx.export(
        wrapper,
        (dummy_ids,),
        os.path.join(output_dir, "codec_embed.onnx"),
        input_names=["token_ids"],
        output_names=["embed"],
        dynamic_shapes={"token_ids": {1: torch.export.Dim("seq_len")}},
        opset_version=opset_version,
        dynamo=True,
        external_data=True,
    )
    print("  Done: codec_embed.onnx")


def _make_ids(vocab_size, seq_len, device):
    """生成覆盖首尾 id 的确定性 token ids，避免随机性影响验证。"""
    base = torch.arange(seq_len, dtype=torch.long, device=device)
    ids = (base * 37 + 1) % vocab_size
    if seq_len >= 2:
        ids[0] = 0
        ids[-1] = vocab_size - 1
    return ids.unsqueeze(0)


def verify_text_project(model, session, onnx_path, seq_len, atol):
    """比较 text_project PyTorch 与 ONNX 输出。"""
    ids = _make_ids(model.talker.config.text_vocab_size, seq_len, model.device)
    with torch.no_grad():
        pt = model.talker.text_projection(_text_embedding(model.talker)(ids))
    ort_out = session.run(["text_embed"], {"input_ids": ids.cpu().numpy().astype(np.int64)})[0]

    print(f"\n[Verify] text_project.onnx seq_len={seq_len}")
    print(f"  onnx path: {onnx_path}")
    ok, _, _ = compare_tensor("text_embed compare", pt.cpu().numpy(), ort_out, atol, atol)
    print(f"  result: {'PASS' if ok else 'FAIL'}")
    return ok


def verify_codec_embed(model, session, onnx_path, seq_len, atol):
    """比较 codec_embed PyTorch 与 ONNX 输出。"""
    ids = _make_ids(model.talker.config.vocab_size, seq_len, model.device)
    with torch.no_grad():
        pt = _codec_embedding(model.talker)(ids)
    ort_out = session.run(["embed"], {"token_ids": ids.cpu().numpy().astype(np.int64)})[0]

    print(f"\n[Verify] codec_embed.onnx seq_len={seq_len}")
    print(f"  onnx path: {onnx_path}")
    ok, _, _ = compare_tensor("embed compare", pt.cpu().numpy(), ort_out, atol, atol)
    print(f"  result: {'PASS' if ok else 'FAIL'}")
    return ok


def _parse_ints(primary, extra):
    values = [primary]
    if extra.strip():
        values.extend(int(x.strip()) for x in extra.split(",") if x.strip())
    return list(dict.fromkeys(values))


def main_embed():
    parser = argparse.ArgumentParser(description="Export/verify text_project and codec_embed ONNX")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--output-dir", type=str, default="./qwen3-tts-0.6b-12hz")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--opset-version", type=int, default=14)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument(
        "--only",
        choices=["all", "text_project", "codec_embed"],
        default="all",
        help="Select which embedding ONNX model to export and verify",
    )
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument("--extra-seq-lens", type=str, default="2,3,4,5,8,12,16,32,64")
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

    model = AutoModel.from_pretrained(args.model, device_map=args.device, dtype=parse_torch_dtype(args.dtype))
    model.eval()
    print_torch_dtype_summary(model, "Qwen3-TTS")

    export_text = args.only in ("all", "text_project")
    export_codec = args.only in ("all", "codec_embed")

    if not args.skip_export and export_text:
        export_text_project(model, args.output_dir, args.opset_version)
    if not args.skip_export and export_codec:
        export_codec_embed(model, args.output_dir, args.opset_version)

    text_path = os.path.join(args.output_dir, "text_project.onnx")
    codec_path = os.path.join(args.output_dir, "codec_embed.onnx")
    if export_text and not os.path.isfile(text_path):
        raise FileNotFoundError(f"Cannot verify: missing {text_path}")
    if export_codec and not os.path.isfile(codec_path):
        raise FileNotFoundError(f"Cannot verify: missing {codec_path}")

    seq_lens = _parse_ints(args.seq_len, args.extra_seq_lens)
    print("\n开始校验 embedding ONNX")
    print(f"  seq_lens: {seq_lens}")
    print(f"  atol={args.atol}")
    if export_text:
        print_expected_inputs(
            "text_project.onnx",
            [("input_ids", "int64", "[1, seq_len]")],
        )
        print_onnx_io_dtypes(text_path, "text_project.onnx")
    if export_codec:
        print_expected_inputs(
            "codec_embed.onnx",
            [("token_ids", "int64", "[1, seq_len]")],
        )
        print_onnx_io_dtypes(codec_path, "codec_embed.onnx")

    if args.skip_verify:
        return

    text_session = (
        ort.InferenceSession(text_path, providers=["CPUExecutionProvider"])
        if export_text
        else None
    )
    codec_session = (
        ort.InferenceSession(codec_path, providers=["CPUExecutionProvider"])
        if export_codec
        else None
    )

    all_ok = True
    for seq_len in seq_lens:
        if export_text:
            all_ok = verify_text_project(model, text_session, text_path, seq_len, args.atol) and all_ok
        if export_codec:
            all_ok = verify_codec_embed(model, codec_session, codec_path, seq_len, args.atol) and all_ok

    print(f"\n校验总结果: {'通过' if all_ok else '失败'}")
    if not all_ok:
        raise RuntimeError("embedding ONNX verification failed")



# ---- code predictor embedding verification ----
"""导出并校验 code_predictor_embed.onnx。

每个 residual codebook 都有自己的 embedding 表。这个子图输入 token_id
和 layer_idx，输出对应 residual codebook 的 embedding。
"""

import argparse
import os

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoProcessor

from qwen_tts.core.models import (
    Qwen3TTSConfig,
    Qwen3TTSForConditionalGeneration,
    Qwen3TTSProcessor,
)



def export_code_predictor_embed(model, output_dir, opset_version=14):
    """导出 code_predictor residual embedding 查询子图。"""
    print("Exporting code_predictor_embed.onnx ...")
    embed_layers = list(model.talker.code_predictor.get_input_embeddings())

    class CodePredictorEmbed(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.embed_layers = nn.ModuleList(layers)

        def forward(self, token_id, layer_idx):
            embeds = []
            for layer in self.embed_layers:
                embeds.append(layer(token_id))
            stacked = torch.stack(embeds, dim=0)
            return stacked[layer_idx]

    wrapper = CodePredictorEmbed(embed_layers).eval()
    dummy_token = torch.tensor([[100, 101, 102, 103, 104]], dtype=torch.long, device=model.device)
    dummy_layer = torch.tensor(0, dtype=torch.long, device=model.device)

    torch.onnx.export(
        wrapper,
        (dummy_token, dummy_layer),
        os.path.join(output_dir, "code_predictor_embed.onnx"),
        input_names=["token_id", "layer_idx"],
        output_names=["embed"],
        dynamic_axes={
            "token_id": {1: "seq_len"},
            "embed": {1: "seq_len"},
        },
        opset_version=opset_version,
        do_constant_folding=False,
        dynamo=False,
    )
    print(f"  Done: code_predictor_embed.onnx ({len(embed_layers)} layers)")


def _make_token_ids(vocab_size, seq_len, device):
    """生成确定性 token ids，覆盖 0 和 vocab_size-1 两个边界。"""
    ids = (torch.arange(seq_len, dtype=torch.long, device=device) * 43 + 7) % vocab_size
    if seq_len >= 2:
        ids[0] = 0
        ids[-1] = vocab_size - 1
    return ids.unsqueeze(0)


def verify_code_predictor_embed(model, session, onnx_path, seq_len, layer_idx, atol):
    """比较某个 residual layer 的 embedding 查询结果。"""
    code_predictor = model.talker.code_predictor
    vocab_size = code_predictor.config.vocab_size
    ids = _make_token_ids(vocab_size, seq_len, model.device)
    layer_idx_tensor = torch.tensor(layer_idx, dtype=torch.long, device=model.device)

    with torch.no_grad():
        pt = list(code_predictor.get_input_embeddings())[layer_idx](ids)

    ort_out = session.run(
        ["embed"],
        {
            "token_id": ids.cpu().numpy().astype(np.int64),
            "layer_idx": layer_idx_tensor.cpu().numpy().astype(np.int64),
        },
    )[0]

    print(f"\n[Verify] code_predictor_embed.onnx seq_len={seq_len} layer_idx={layer_idx}")
    print(f"  onnx path: {onnx_path}")
    ok, _, _ = compare_tensor("embed compare", pt.cpu().numpy(), ort_out, atol, atol)
    print(f"  result: {'PASS' if ok else 'FAIL'}")
    return ok


def _parse_ints(primary, extra):
    values = [primary]
    if extra.strip():
        values.extend(int(x.strip()) for x in extra.split(",") if x.strip())
    return list(dict.fromkeys(values))


def main_code_predictor_embed():
    parser = argparse.ArgumentParser(description="Export/verify code_predictor_embed ONNX")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--output-dir", type=str, default="./qwen3-tts-0.6b-12hz")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--opset-version", type=int, default=14)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument("--extra-seq-lens", type=str, default="2,3,5,8,16,32,64")
    parser.add_argument(
        "--layer-indices",
        type=str,
        default="0,1,2,7,14",
        help="Comma-separated code predictor embedding layer indices to verify",
    )
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

    model = AutoModel.from_pretrained(args.model, device_map=args.device, dtype=parse_torch_dtype(args.dtype))
    model.eval()
    print_torch_dtype_summary(model, "Qwen3-TTS")

    if not args.skip_export:
        export_code_predictor_embed(model, args.output_dir, args.opset_version)

    onnx_path = os.path.join(args.output_dir, "code_predictor_embed.onnx")
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"Cannot verify: missing {onnx_path}")

    seq_lens = _parse_ints(args.seq_len, args.extra_seq_lens)
    layer_indices = [int(x.strip()) for x in args.layer_indices.split(",") if x.strip()]

    print("\n开始校验 code_predictor_embed.onnx")
    print(f"  seq_lens: {seq_lens}")
    print(f"  layer_indices: {layer_indices}")
    print(f"  atol={args.atol}")
    print_expected_inputs(
        "code_predictor_embed.onnx",
        [
            ("token_id", "int64", "[1, seq_len]"),
            ("layer_idx", "int64", "scalar, 0..14"),
        ],
    )
    print_onnx_io_dtypes(onnx_path, "code_predictor_embed.onnx")

    if args.skip_verify:
        return

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    all_ok = True
    for seq_len in seq_lens:
        for layer_idx in layer_indices:
            ok = verify_code_predictor_embed(model, session, onnx_path, seq_len, layer_idx, args.atol)
            all_ok = ok and all_ok

    print(f"\n校验总结果: {'通过' if all_ok else '失败'}")
    if not all_ok:
        raise RuntimeError("code_predictor_embed ONNX verification failed")



# ---- code predictor verification ----
"""导出并校验 code_predictor.onnx。

code_predictor 负责在 talker 给出每帧第 0 个 codec token 后，继续预测
剩余 residual codebook token。输入 context 是
[last_hidden, main_codec_embed, 已生成 residual embeds...]。
"""

import argparse
import os

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoProcessor

from qwen_tts.core.models import (
    Qwen3TTSConfig,
    Qwen3TTSForConditionalGeneration,
    Qwen3TTSProcessor,
)



def export_code_predictor(model, output_dir, opset_version=14, trace_ctx_len=8):
    """导出 residual code predictor 子图。"""
    print("Exporting code_predictor.onnx ...")
    code_predictor = model.talker.code_predictor
    in_dim = model.talker.config.hidden_size
    num_heads = len(code_predictor.lm_head)

    class CodePredictor(nn.Module):
        def __init__(self, predictor, head_count):
            super().__init__()
            self.predictor = predictor
            self.head_count = head_count

        def forward(self, context, gen_step):
            hidden_in = self.predictor.small_to_mtp_projection(context)
            out = self.predictor.model(
                input_ids=None,
                inputs_embeds=hidden_in,
                use_cache=False,
                return_dict=True,
            )
            hidden = out.last_hidden_state[:, -1:, :]
            logits = []
            for i in range(self.head_count):
                logits.append(self.predictor.lm_head[i](hidden))
            stacked = torch.stack(logits, dim=0)
            return stacked[gen_step]

    wrapper = CodePredictor(code_predictor, num_heads).eval()
    model_dtype = next(code_predictor.parameters()).dtype
    dummy_context = torch.randn(1, trace_ctx_len, in_dim, dtype=model_dtype, device=model.device)
    dummy_step = torch.tensor(0, dtype=torch.long, device=model.device)

    torch.onnx.export(
        wrapper,
        (dummy_context, dummy_step),
        os.path.join(output_dir, "code_predictor.onnx"),
        input_names=["context", "gen_step"],
        output_names=["logits"],
        dynamic_axes={
            "context": {1: "ctx_len"},
        },
        opset_version=opset_version,
        do_constant_folding=False,
        dynamo=False,
    )
    print(f"  Done: code_predictor.onnx ({num_heads} heads, trace_ctx_len={trace_ctx_len})")


def patch_code_predictor_dynamic_reshape(onnx_path):
    """修补导出图中可能固化 trace_ctx_len 的 Range->Reshape。"""
    model = onnx.load(onnx_path, load_external_data=False)
    new_nodes = []
    patched = 0

    for node in model.graph.node:
        if node.op_type == "Reshape" and node.name.startswith("/model/Reshape"):
            if len(node.input) >= 2 and node.input[0].startswith("/model/Range"):
                axes_name = f"{node.name}_unsqueeze_axis1_const_output_0"
                unsqueeze_out = f"{node.name}_range_unsqueeze_axis1_output_0"
                original_out = node.output[0]
                new_nodes.append(
                    onnx.helper.make_node(
                        "Constant",
                        inputs=[],
                        outputs=[axes_name],
                        name=f"{node.name}_unsqueeze_axis1_const",
                        value=onnx.helper.make_tensor("value", onnx.TensorProto.INT64, [1], [1]),
                    )
                )
                new_nodes.append(
                    onnx.helper.make_node(
                        "Unsqueeze",
                        inputs=[node.input[0], axes_name],
                        outputs=[unsqueeze_out],
                        name=f"{node.name}_range_unsqueeze_axis1",
                    )
                )
                for consumer in model.graph.node:
                    for idx, input_name in enumerate(consumer.input):
                        if input_name == original_out:
                            consumer.input[idx] = unsqueeze_out
                patched += 1
                continue

        new_nodes.append(node)

    if patched:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
        onnx.save(model, onnx_path)
        print(f"  Patched code_predictor dynamic reshape nodes: {patched}")
    else:
        print("  Patch skipped: no Range->Reshape mask nodes found")

    return patched


def _make_context(seq_len, hidden_size, device):
    """构造 [1, ctx_len, hidden] 的随机 context。"""
    return torch.randn(1, seq_len, hidden_size, dtype=torch.float32, device=device)


def _reference_logits(model, context, gen_step):
    """用 PyTorch 原始 code_predictor 得到某个 residual step 的 logits。"""
    code_predictor = model.talker.code_predictor
    with torch.no_grad():
        hidden_in = code_predictor.small_to_mtp_projection(context)
        out = code_predictor.model(
            input_ids=None,
            inputs_embeds=hidden_in,
            use_cache=False,
            return_dict=True,
        )
        hidden = out.last_hidden_state[:, -1:, :]
        return code_predictor.lm_head[gen_step](hidden)


def verify_code_predictor(model, session, onnx_path, ctx_len, gen_step, atol):
    """比较指定 ctx_len/gen_step 下 PyTorch 与 ONNX logits。"""
    hidden_size = model.talker.config.hidden_size
    context = _make_context(ctx_len, hidden_size, model.device)
    gen_step_tensor = torch.tensor(gen_step, dtype=torch.long, device=model.device)
    pt = _reference_logits(model, context, gen_step)

    try:
        ort_out = session.run(
            ["logits"],
            {
                "context": context.cpu().numpy().astype(np.float32),
                "gen_step": gen_step_tensor.cpu().numpy().astype(np.int64),
            },
        )[0]
    except Exception as exc:
        print(f"\n[Verify] code_predictor.onnx ctx_len={ctx_len} gen_step={gen_step}")
        print(f"  onnx path: {onnx_path}")
        print(f"  onnxruntime error: {exc}")
        print("  result: FAIL")
        return False

    print(f"\n[Verify] code_predictor.onnx ctx_len={ctx_len} gen_step={gen_step}")
    print(f"  onnx path: {onnx_path}")
    ok, _, _ = compare_tensor("logits compare", pt.cpu().numpy(), ort_out, atol, atol)
    print(f"  result: {'PASS' if ok else 'FAIL'}")
    return ok


def _parse_ints(primary, extra):
    values = [primary]
    if extra.strip():
        values.extend(int(x.strip()) for x in extra.split(",") if x.strip())
    return list(dict.fromkeys(values))


def main_code_predictor():
    parser = argparse.ArgumentParser(description="Export/verify code_predictor ONNX")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--output-dir", type=str, default="./qwen3-tts-0.6b-12hz")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--opset-version", type=int, default=14)
    parser.add_argument("--trace-ctx-len", type=int, default=8)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--skip-patch", action="store_true")
    parser.add_argument("--ctx-len", type=int, default=2)
    parser.add_argument("--extra-ctx-lens", type=str, default="3,4,5,8,12,16,32,64")
    parser.add_argument("--gen-steps", type=str, default="0,1,2,7,14")
    parser.add_argument("--atol", type=float, default=5e-4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

    model = AutoModel.from_pretrained(args.model, device_map=args.device, dtype=parse_torch_dtype(args.dtype))
    model.eval()
    print_torch_dtype_summary(model, "Qwen3-TTS")

    if not args.skip_export:
        export_code_predictor(
            model,
            args.output_dir,
            opset_version=args.opset_version,
            trace_ctx_len=args.trace_ctx_len,
        )

    onnx_path = os.path.join(args.output_dir, "code_predictor.onnx")
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"Cannot verify: missing {onnx_path}")

    if not args.skip_patch:
        patch_code_predictor_dynamic_reshape(onnx_path)

    if args.skip_verify:
        print_onnx_io_dtypes(onnx_path, "code_predictor.onnx")
        return

    ctx_lens = _parse_ints(args.ctx_len, args.extra_ctx_lens)
    gen_steps = [int(x.strip()) for x in args.gen_steps.split(",") if x.strip()]

    print("\n开始校验 code_predictor.onnx")
    print(f"  ctx_lens: {ctx_lens}")
    print(f"  gen_steps: {gen_steps}")
    print(f"  atol={args.atol}")
    print_expected_inputs(
        "code_predictor.onnx",
        [
            ("context", "float32", f"[1, ctx_len, {model.talker.config.hidden_size}]"),
            ("gen_step", "int64", f"scalar, 0..{model.talker.config.num_code_groups - 2}"),
        ],
    )
    print_onnx_io_dtypes(onnx_path, "code_predictor.onnx")

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    all_ok = True
    for ctx_len in ctx_lens:
        for gen_step in gen_steps:
            ok = verify_code_predictor(model, session, onnx_path, ctx_len, gen_step, args.atol)
            all_ok = ok and all_ok

    print(f"\n校验总结果: {'通过' if all_ok else '失败'}")
    if not all_ok:
        raise RuntimeError("code_predictor ONNX verification failed")



# ---- speaker encoder verification ----
"""导出并校验 speaker_encoder.onnx。

speaker_encoder 接收 mel [1, frames, mel_dim]，输出说话人 embedding。
声音克隆时这个 embedding 会被插入 codec prompt，作为音色条件。
"""

import argparse
import os

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoProcessor

from qwen_tts.core.models import (
    Qwen3TTSConfig,
    Qwen3TTSForConditionalGeneration,
    Qwen3TTSProcessor,
)



def _speaker_encoder_output(output):
    """兼容 speaker_encoder 返回 tensor 或 tuple/list 的版本差异。"""
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def export_speaker_encoder(model, output_dir, opset_version=14, trace_frames=100):
    """导出 speaker_encoder.onnx，time 维保持动态。"""
    print("Exporting speaker_encoder.onnx ...")
    if model.speaker_encoder is None:
        raise RuntimeError("model has no speaker_encoder")

    class SpeakerEncoderWrapper(nn.Module):
        def __init__(self, speaker_encoder):
            super().__init__()
            self.encoder = speaker_encoder

        def forward(self, mel):
            return _speaker_encoder_output(self.encoder(mel))

    wrapper = SpeakerEncoderWrapper(model.speaker_encoder).eval()
    mel_dim = model.config.speaker_encoder_config.mel_dim
    model_dtype = next(model.speaker_encoder.parameters()).dtype
    dummy_mel = torch.randn(1, trace_frames, mel_dim, device=model.device, dtype=model_dtype)

    torch.onnx.export(
        wrapper,
        (dummy_mel,),
        os.path.join(output_dir, "speaker_encoder.onnx"),
        input_names=["mel"],
        output_names=["speaker_embedding"],
        dynamic_axes={
            "mel": {1: "time"},
        },
        opset_version=opset_version,
        do_constant_folding=False,
        dynamo=False,
    )
    print(f"  Done: speaker_encoder.onnx (trace_frames={trace_frames}, mel_dim={mel_dim})")


def _make_mel(frames, mel_dim, device):
    """构造平滑伪 mel，用于稳定验证动态 frames。"""
    t = torch.linspace(0.0, 1.0, frames, dtype=torch.float32, device=device).unsqueeze(1)
    bins = torch.linspace(0.2, 2.0, mel_dim, dtype=torch.float32, device=device).unsqueeze(0)
    mel = torch.sin(2.0 * torch.pi * t * bins) + 0.1 * torch.cos(torch.pi * t * bins)
    return mel.unsqueeze(0)


def verify_speaker_encoder(model, session, onnx_path, frames, atol):
    """比较指定 frames 下 PyTorch 与 ONNX speaker embedding。"""
    mel_dim = model.config.speaker_encoder_config.mel_dim
    mel = _make_mel(frames, mel_dim, model.device)
    with torch.no_grad():
        pt = _speaker_encoder_output(model.speaker_encoder(mel))

    try:
        ort_out = session.run(
            ["speaker_embedding"],
            {"mel": mel.cpu().numpy().astype(np.float32)},
        )[0]
    except Exception as exc:
        print(f"\n[Verify] speaker_encoder.onnx frames={frames}")
        print(f"  onnx path: {onnx_path}")
        print(f"  onnxruntime error: {exc}")
        print("  result: FAIL")
        return False

    print(f"\n[Verify] speaker_encoder.onnx frames={frames}")
    print(f"  onnx path: {onnx_path}")
    ok, _, _ = compare_tensor("speaker_embedding compare", pt.cpu().numpy(), ort_out, atol, atol)
    print(f"  result: {'PASS' if ok else 'FAIL'}")
    return ok


def _parse_ints(primary, extra):
    values = [primary]
    if extra.strip():
        values.extend(int(x.strip()) for x in extra.split(",") if x.strip())
    return list(dict.fromkeys(values))


def main_speaker_encoder():
    parser = argparse.ArgumentParser(description="Export/verify speaker_encoder ONNX")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--output-dir", type=str, default="./qwen3-tts-0.6b-12hz")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--opset-version", type=int, default=14)
    parser.add_argument("--trace-frames", type=int, default=100)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--extra-frames", type=str, default="32,64,100,128,256,512")
    parser.add_argument("--atol", type=float, default=5e-4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

    model = AutoModel.from_pretrained(args.model, device_map=args.device, dtype=parse_torch_dtype(args.dtype))
    model.eval()
    print_torch_dtype_summary(model, "Qwen3-TTS")

    if not args.skip_export:
        export_speaker_encoder(
            model,
            args.output_dir,
            opset_version=args.opset_version,
            trace_frames=args.trace_frames,
        )

    onnx_path = os.path.join(args.output_dir, "speaker_encoder.onnx")
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"Cannot verify: missing {onnx_path}")

    frame_lengths = _parse_ints(args.frames, args.extra_frames)
    print("\n开始校验 speaker_encoder.onnx")
    print(f"  frames: {frame_lengths}")
    print(f"  atol={args.atol}")
    print_expected_inputs(
        "speaker_encoder.onnx",
        [("mel", "float32", "[1, frames, 128]")],
    )
    print_onnx_io_dtypes(onnx_path, "speaker_encoder.onnx")

    if args.skip_verify:
        return

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    all_ok = True
    for frames in frame_lengths:
        ok = verify_speaker_encoder(model, session, onnx_path, frames, args.atol)
        all_ok = ok and all_ok

    print(f"\n校验总结果: {'通过' if all_ok else '失败'}")
    if not all_ok:
        raise RuntimeError("speaker_encoder ONNX verification failed")




def main() -> None:
    import sys
    from pathlib import Path
    commands = {
        "talker-prefill": main_talker_prefill,
        "talker-decode": main_talker_decode,
        "embed": main_embed,
        "code-predictor-embed": main_code_predictor_embed,
        "code-predictor": main_code_predictor,
        "speaker-encoder": main_speaker_encoder,
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
