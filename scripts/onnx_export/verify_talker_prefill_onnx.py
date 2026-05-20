#!/usr/bin/env python3
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

from onnx_dtype_utils import print_expected_inputs, print_onnx_io_dtypes, print_torch_dtype_summary

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


def main():
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


if __name__ == "__main__":
    main()
