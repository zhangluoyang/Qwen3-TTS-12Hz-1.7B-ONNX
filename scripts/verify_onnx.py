#!/usr/bin/env python3
"""User-facing ONNX verification entrypoint for Qwen3-TTS."""

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from export_onnx import REPO_ROOT, default_output_root, infer_model_type, load_model_config, parse_dtype


DEFAULT_MODEL = "/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base"
RUNTIME_DIR = REPO_ROOT / "scripts" / "onnx_runtime"
sys.path.insert(0, str(RUNTIME_DIR))


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
    import torch

    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def default_torch_dtype(model_type: str, export_dtype: str) -> str:
    if model_type == "voice_design":
        return "bfloat16"
    return "float16" if export_dtype == "float16" else "float32"


def cleanup_torch():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_pytorch_base(args):
    import torch
    from qwen_tts import Qwen3TTSModel

    print("Loading PyTorch Base model...")
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=torch_dtype_from_arg(args.torch_dtype),
        local_files_only=True,
    )
    print("Running PyTorch greedy voice clone...")
    with torch.no_grad():
        wavs, sr = model.generate_voice_clone(
            text=args.text,
            language=args.language,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            subtalker_dosample=False,
        )
    wav = np.asarray(wavs[0], dtype=np.float32)
    del model, wavs
    cleanup_torch()
    return wav, sr, None


def run_pytorch_custom_voice(args):
    import torch
    from qwen_tts import Qwen3TTSModel

    print("Loading PyTorch CustomVoice model...")
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=torch_dtype_from_arg(args.torch_dtype),
        local_files_only=True,
    )
    input_ids = model._tokenize_texts([model._build_assistant_text(args.text)])
    instruct_ids = [None]
    if args.instruct and model.model.tts_model_size not in "0b6":
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
            speakers=[args.speaker],
            non_streaming_mode=True,
            **gen_kwargs,
        )
        wavs, sr = model.model.speech_tokenizer.decode([{"audio_codes": c} for c in codes_list])
    codes = codes_list[0].detach().cpu().numpy().astype(np.int64)
    wav = np.asarray(wavs[0], dtype=np.float32)
    del model, codes_list, wavs, input_ids, instruct_ids
    cleanup_torch()
    return wav, sr, codes


def run_pytorch_voice_design(args):
    import torch
    from qwen_tts import Qwen3TTSModel

    print("Loading PyTorch VoiceDesign model...")
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=torch_dtype_from_arg(args.torch_dtype),
        local_files_only=True,
    )
    input_ids = model._tokenize_texts([model._build_assistant_text(args.text)])
    instruct_ids = [model._tokenize_texts([model._build_instruct_text(args.instruct)])[0]] if args.instruct else [None]
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
    cleanup_torch()
    return wav, sr, codes


def run_ort(args, model_type: str, onnx_root: Path):
    providers = [args.provider]
    if model_type == "base":
        from voice_clone_ort import Qwen3TTSVoiceCloneORT

        runner = Qwen3TTSVoiceCloneORT(
            args.model,
            onnx_root,
            providers=providers,
            seed=1234,
            use_iobinding=not args.no_iobinding,
        )
        return runner.generate_voice_clone(
            text=args.text,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            language=args.language,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            subtalker_dosample=False,
        )
    if model_type == "custom_voice":
        from custom_voice_ort import Qwen3TTSCustomVoiceORT

        runner = Qwen3TTSCustomVoiceORT(
            model_dir=args.model,
            onnx_root=onnx_root,
            providers=providers,
            seed=1234,
            use_iobinding=not args.no_iobinding,
        )
        return runner.generate_custom_voice(
            text=args.text,
            speaker=args.speaker,
            language=args.language,
            instruct=args.instruct or None,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            subtalker_dosample=False,
            verbose=True,
        )
    if model_type == "voice_design":
        from voice_design_ort import Qwen3TTSVoiceDesignORT

        runner = Qwen3TTSVoiceDesignORT(
            model_dir=args.model,
            onnx_root=onnx_root,
            providers=providers,
            seed=1234,
            use_iobinding=not args.no_iobinding,
        )
        return runner.generate_voice_design(
            text=args.text,
            instruct=args.instruct,
            language=args.language,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            subtalker_dosample=False,
            verbose=True,
        )
    raise ValueError(f"unsupported tts_model_type={model_type!r}")


def compare_codes(pt_codes, ort_codes):
    if pt_codes is None:
        return
    common_frames = min(pt_codes.shape[0], ort_codes.shape[0])
    mismatch = int(np.count_nonzero(pt_codes[:common_frames] != ort_codes[:common_frames]))
    print("\nCodec codes compare:")
    print(f"  pt_shape={pt_codes.shape}")
    print(f"  ort_shape={ort_codes.shape}")
    print(f"  common_frames={common_frames}")
    print(f"  common_prefix_exact={np.array_equal(pt_codes[:common_frames], ort_codes[:common_frames])}")
    print(f"  mismatch_count={mismatch}")
    if mismatch:
        first = np.argwhere(pt_codes[:common_frames] != ort_codes[:common_frames])[0]
        frame, group = int(first[0]), int(first[1])
        print(f"  first_mismatch=frame {frame}, group {group}, pt={pt_codes[frame, group]}, ort={ort_codes[frame, group]}")


def compare_audio(pt_wav, ort_wav):
    n = min(pt_wav.shape[0], ort_wav.shape[0])
    diff = pt_wav[:n] - ort_wav[:n]
    print("\nWaveform compare on common prefix:")
    print(f"  common_samples={n}")
    print(f"  length_delta={pt_wav.shape[0] - ort_wav.shape[0]}")
    print(f"  corr={corrcoef(pt_wav, ort_wav):.6f}")
    print(f"  max_abs={float(np.max(np.abs(diff))):.6f}")
    print(f"  mean_abs={float(np.mean(np.abs(diff))):.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify exported Qwen3-TTS ONNX models.")
    parser.add_argument("--model", default=os.environ.get("QWEN3_TTS_MODEL_DIR", DEFAULT_MODEL))
    parser.add_argument("--onnx-root", default=None)
    parser.add_argument("--dtype", type=parse_dtype, default="float16", help="Export dtype used by the ONNX root.")
    parser.add_argument("--provider", default="CUDAExecutionProvider")
    parser.add_argument("--device", default="cuda:0", help="PyTorch reference device.")
    parser.add_argument("--text", default="你好，这是 Qwen 三 ONNX Runtime 贪心对齐测试。")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-iobinding", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ref-audio", default="./data/ref_from_mp3_24k_mono.wav")
    parser.add_argument("--ref-text", default="告诉自己，不要怕")
    parser.add_argument("--speaker", default="Vivian")
    parser.add_argument("--instruct", default="一个年轻女性的声音，语气温柔自然，语速适中，发音清晰。")
    parser.add_argument("--torch-dtype", default=None, choices=("float32", "float16", "bfloat16"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_model_config(args.model)
    model_type = infer_model_type(args.model, config)
    args.torch_dtype = args.torch_dtype or default_torch_dtype(model_type, args.dtype)

    onnx_root = Path(args.onnx_root) if args.onnx_root else default_output_root(args.model, model_type, args.dtype)
    onnx_root = onnx_root.expanduser()
    if not onnx_root.is_absolute():
        onnx_root = REPO_ROOT / onnx_root

    print("Qwen3-TTS ONNX verify")
    print(f"  model:       {args.model}")
    print(f"  model-type:  {model_type}")
    print(f"  onnx-root:   {onnx_root}")
    print(f"  provider:    {args.provider}")
    print(f"  torch-dtype: {args.torch_dtype}")
    if args.dry_run:
        print("\nDry run only; no model was loaded.")
        return

    out_dir = Path(args.output_dir or f"verify_{model_type}_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    pytorch_runners = {
        "base": run_pytorch_base,
        "custom_voice": run_pytorch_custom_voice,
        "voice_design": run_pytorch_voice_design,
    }
    if model_type not in pytorch_runners:
        raise SystemExit(f"unsupported tts_model_type={model_type!r}")

    pt_wav, pt_sr, pt_codes = pytorch_runners[model_type](args)
    sf.write(out_dir / "pytorch_greedy.wav", pt_wav, pt_sr)
    if pt_codes is not None:
        np.save(out_dir / "pytorch_codes.npy", pt_codes)
    summarize_audio("PyTorch", pt_wav, pt_sr)

    print("Running ONNX Runtime greedy inference...")
    ort_wav, ort_sr, ort_codes = run_ort(args, model_type, onnx_root)
    sf.write(out_dir / "onnx_greedy.wav", ort_wav, ort_sr)
    np.save(out_dir / "onnx_codes.npy", ort_codes)
    summarize_audio("ONNX", ort_wav, ort_sr)

    compare_codes(pt_codes, ort_codes)
    compare_audio(pt_wav, ort_wav)
    print(f"\nWrote outputs under: {out_dir}")


if __name__ == "__main__":
    main()
