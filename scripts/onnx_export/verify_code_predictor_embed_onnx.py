#!/usr/bin/env python3
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

from onnx_dtype_utils import print_expected_inputs, print_onnx_io_dtypes, print_torch_dtype_summary
from verify_talker_prefill_onnx import parse_torch_dtype
from verify_talker_prefill_onnx import compare_tensor


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


def main():
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


if __name__ == "__main__":
    main()
