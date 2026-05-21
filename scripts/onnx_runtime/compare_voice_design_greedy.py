#!/usr/bin/env python3
"""Greedy compare for PyTorch vs ONNX Runtime Qwen3-TTS VoiceDesign."""

import argparse
import gc
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from voice_design_ort import Qwen3TTSVoiceDesignORT


def corrcoef(a, b):
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
    print(
        f"{name}: sr={sr}, samples={wav.shape[0]}, dur={wav.shape[0] / sr:.3f}s, "
        f"min={float(np.min(wav)):.6f}, max={float(np.max(wav)):.6f}, mean={float(np.mean(wav)):.6f}"
    )


def torch_dtype_from_arg(name):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def run_pytorch(args):
    print("Loading PyTorch VoiceDesign model...")
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=torch_dtype_from_arg(args.dtype),
        local_files_only=True,
    )

    input_ids = model._tokenize_texts([model._build_assistant_text(args.text)])
    instruct_ids = [None]
    if args.instruct:
        instruct_ids = [model._tokenize_texts([model._build_instruct_text(args.instruct)])[0]]

    gen_kwargs = model._merge_generate_kwargs(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        subtalker_dosample=False,
    )
    with torch.no_grad():
        codes_list, _ = model.model.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=[args.language],
            non_streaming_mode=True,
            **gen_kwargs,
        )
        wavs, sr = model.model.speech_tokenizer.decode([{"audio_codes": c} for c in codes_list])

    codes = codes_list[0].detach().cpu().numpy().astype(np.int64)
    wav = np.asarray(wavs[0], dtype=np.float32)
    del model, codes_list, wavs, input_ids, instruct_ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return wav, sr, codes


def main():
    parser = argparse.ArgumentParser(description="Greedy PyTorch vs ORT VoiceDesign comparison")
    parser.add_argument("--model", default="/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    parser.add_argument("--onnx-root", default="./onnx_voice_design_1p7b_fp16")
    parser.add_argument("--text", default="你好，这是 Qwen 三音色设计的 GPU 贪心对齐测试。")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--instruct", default="一个年轻女性的声音，语气温柔自然，语速适中，发音清晰。")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--provider", default="CUDAExecutionProvider")
    parser.add_argument("--output-dir", default="compare_voice_design_outputs")
    parser.add_argument("--no-iobinding", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pt_wav, pt_sr, pt_codes = run_pytorch(args)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sf.write(out_dir / "pt_voice_design_greedy.wav", pt_wav, pt_sr)
    np.save(out_dir / "pt_voice_design_codes.npy", pt_codes)
    summarize_audio("PyTorch", pt_wav, pt_sr)

    print("Running ORT VoiceDesign...")
    ort_runner = Qwen3TTSVoiceDesignORT(
        model_dir=args.model,
        onnx_root=args.onnx_root,
        providers=[args.provider],
        seed=1234,
        use_iobinding=not args.no_iobinding,
    )
    ort_wav, ort_sr, ort_codes = ort_runner.generate_voice_design(
        text=args.text,
        instruct=args.instruct,
        language=args.language,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        subtalker_dosample=False,
        verbose=True,
    )
    sf.write(out_dir / "ort_voice_design_greedy.wav", ort_wav, ort_sr)
    np.save(out_dir / "ort_voice_design_codes.npy", ort_codes)
    summarize_audio("ORT", ort_wav, ort_sr)

    common_frames = min(pt_codes.shape[0], ort_codes.shape[0])
    frame_equal = np.array_equal(pt_codes[:common_frames], ort_codes[:common_frames])
    mismatch = int(np.count_nonzero(pt_codes[:common_frames] != ort_codes[:common_frames]))
    print("\nCodec codes compare:")
    print(f"  pt_shape={pt_codes.shape}")
    print(f"  ort_shape={ort_codes.shape}")
    print(f"  common_frames={common_frames}")
    print(f"  common_prefix_exact={frame_equal}")
    print(f"  mismatch_count={mismatch}")
    if mismatch:
        first = np.argwhere(pt_codes[:common_frames] != ort_codes[:common_frames])[0]
        frame, group = int(first[0]), int(first[1])
        print(f"  first_mismatch=frame {frame}, group {group}, pt={pt_codes[frame, group]}, ort={ort_codes[frame, group]}")

    n = min(pt_wav.shape[0], ort_wav.shape[0])
    diff = pt_wav[:n] - ort_wav[:n]
    print("\nWaveform compare on common prefix:")
    print(f"  common_samples={n}")
    print(f"  length_delta={pt_wav.shape[0] - ort_wav.shape[0]}")
    print(f"  corr={corrcoef(pt_wav, ort_wav):.6f}")
    print(f"  max_abs={float(np.max(np.abs(diff))):.6f}")
    print(f"  mean_abs={float(np.mean(np.abs(diff))):.6f}")
    print(f"\nWrote outputs under: {out_dir}")


if __name__ == "__main__":
    main()
