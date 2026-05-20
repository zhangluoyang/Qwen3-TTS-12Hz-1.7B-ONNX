#!/usr/bin/env python3
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

from onnx_dtype_utils import print_expected_inputs, print_onnx_io_dtypes, print_torch_dtype_summary
from verify_talker_prefill_onnx import compare_tensor, parse_torch_dtype


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


def main():
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


if __name__ == "__main__":
    main()
