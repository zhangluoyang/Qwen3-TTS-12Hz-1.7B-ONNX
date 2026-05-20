#!/usr/bin/env python3
"""在 greedy 模式下比较 PyTorch 与 ONNX Runtime 的 Qwen3-TTS 声音克隆输出。

greedy 会关闭随机采样，适合排查导出后的 ONNX 子图和原始 PyTorch wrapper
是否在主流程上对齐。最终音频可能仍有微小差异，但 token/codes 应尽量一致。
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from voice_clone_ort import Qwen3TTSVoiceCloneORT


def corrcoef(a, b):
    """计算两段波形的相关系数，长度不同时只比较共同前缀。"""
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    x = np.asarray(a[:n], dtype=np.float64)
    y = np.asarray(b[:n], dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    den = np.sqrt(np.sum(x * x) * np.sum(y * y))
    return float(np.sum(x * y) / den) if den > 0 else float("nan")


def summarize_audio(name, wav, sr):
    """打印音频基本统计，快速发现全零、爆音、采样率错误等问题。"""
    print(
        f"{name}: sr={sr}, samples={wav.shape[0]}, dur={wav.shape[0] / sr:.3f}s, "
        f"min={float(np.min(wav)):.6f}, max={float(np.max(wav)):.6f}, mean={float(np.mean(wav)):.6f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Greedy PyTorch vs ORT voice-clone comparison")
    parser.add_argument("--model", default="/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--onnx-root", default="./onnx_isolated")
    parser.add_argument("--text", default="我和我的祖国，一刻也不能分割")
    parser.add_argument("--ref-audio", default="./data/林志玲.mp3")
    parser.add_argument("--ref-text", default="告诉自己，不要怕")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--output-dir", default="compare_outputs")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # PyTorch 路径是对照组，直接使用官方 wrapper，不经过 ONNX 子图。
    print("Loading PyTorch model...")
    pt = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=torch.float32,
    )

    print("Running PyTorch greedy generate_voice_clone...")
    with torch.no_grad():
        pt_wavs, pt_sr = pt.generate_voice_clone(
            text=args.text,
            language=args.language,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            subtalker_dosample=False,
        )
    pt_wav = np.asarray(pt_wavs[0], dtype=np.float32)
    sf.write(out_dir / "pt_greedy.wav", pt_wav, pt_sr)
    summarize_audio("PyTorch", pt_wav, pt_sr)

    # ONNX 路径使用导出的拆分子模型，采样参数与 PyTorch 都设为 greedy。
    print("Running ORT greedy voice_clone...")
    ort_runner = Qwen3TTSVoiceCloneORT(
        args.model,
        args.onnx_root,
        providers=[args.provider],
        seed=1234,
    )
    ort_wav, ort_sr, ort_codes = ort_runner.generate_voice_clone(
        text=args.text,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        language=args.language,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        subtalker_dosample=False,
    )
    sf.write(out_dir / "ort_greedy.wav", ort_wav, ort_sr)
    np.save(out_dir / "ort_codes.npy", ort_codes)
    summarize_audio("ORT", ort_wav, ort_sr)

    n = min(pt_wav.shape[0], ort_wav.shape[0])
    diff = pt_wav[:n] - ort_wav[:n]
    print("\nWaveform compare on common prefix:")
    print(f"  common_samples={n}")
    print(f"  length_delta={pt_wav.shape[0] - ort_wav.shape[0]}")
    print(f"  corr={corrcoef(pt_wav, ort_wav):.6f}")
    print(f"  max_abs={float(np.max(np.abs(diff))):.6f}")
    print(f"  mean_abs={float(np.mean(np.abs(diff))):.6f}")
    print(f"\nWrote: {out_dir / 'pt_greedy.wav'}")
    print(f"Wrote: {out_dir / 'ort_greedy.wav'}")
    print(f"Wrote: {out_dir / 'ort_codes.npy'}")


if __name__ == "__main__":
    main()
