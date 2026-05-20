#!/usr/bin/env python3
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

from onnx_dtype_utils import print_expected_inputs, print_onnx_io_dtypes, print_torch_dtype_summary
from verify_talker_prefill_onnx import compare_tensor, parse_torch_dtype


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


def main():
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


if __name__ == "__main__":
    main()
