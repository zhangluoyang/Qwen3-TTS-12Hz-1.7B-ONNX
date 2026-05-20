#!/usr/bin/env python3
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

from onnx_dtype_utils import print_expected_inputs, print_onnx_io_dtypes, print_torch_dtype_summary
from verify_talker_prefill_onnx import parse_torch_dtype
from verify_talker_prefill_onnx import compare_tensor


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


def main():
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


if __name__ == "__main__":
    main()
