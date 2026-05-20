#!/usr/bin/env python3
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

from onnx_dtype_utils import print_expected_inputs, print_onnx_io_dtypes, print_torch_dtype_summary
from verify_talker_prefill_onnx import _get_talker_head, compare_tensor, parse_torch_dtype


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


def main():
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


if __name__ == "__main__":
    main()
