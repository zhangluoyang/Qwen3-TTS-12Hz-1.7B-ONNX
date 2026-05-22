#!/usr/bin/env python3
"""Single ONNX Runtime example for Qwen3-TTS Base / CustomVoice / VoiceDesign."""

import argparse
import sys
from pathlib import Path

import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from export_onnx import default_output_root, infer_model_type, load_model_config, parse_dtype  # noqa: E402


DEFAULT_MODEL = "/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_TEXT = "你好，这是 Qwen 三 ONNX Runtime 统一样例。"


def ensure_chunk_decoder(onnx_root: Path, enabled: bool) -> None:
    if not enabled:
        return
    path = onnx_root / "tokenizer12hz" / "tokenizer12hz_decode_chunk.onnx"
    if not path.exists():
        raise FileNotFoundError(
            f"Chunk decoder not found: {path}. Export with scripts/export_onnx.py first."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one Qwen3-TTS ONNX Runtime example.")
    parser.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--onnx-root", type=Path, default=None)
    parser.add_argument("--dtype", type=parse_dtype, default="float16", help="Used only to infer default ONNX root.")
    parser.add_argument("--provider", default="CUDAExecutionProvider")
    parser.add_argument("--output", type=Path, default=Path("output_onnx_example.wav"))
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--max-new-tokens", type=int, default=600)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--no-iobinding", action="store_true")

    parser.add_argument("--ref-audio", type=Path, default=Path("data/ref_from_mp3_24k_mono.wav"))
    parser.add_argument("--ref-text", default="告诉自己，不要怕")
    parser.add_argument("--speaker", default="Vivian")
    parser.add_argument("--instruct", default="一个年轻女性的声音，语气温柔自然，语速适中，发音清晰。")

    parser.add_argument("--legacy-full-decoder", action="store_true")
    parser.add_argument("--chunk-frames", type=int, default=300)
    parser.add_argument("--left-context-frames", type=int, default=25)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_model_config(str(args.model))
    model_type = infer_model_type(str(args.model), config)

    onnx_root = args.onnx_root or default_output_root(str(args.model), model_type, args.dtype)
    onnx_root = onnx_root.expanduser()
    if not onnx_root.is_absolute():
        onnx_root = REPO_ROOT / onnx_root

    providers = [args.provider]
    do_sample = False if args.greedy else None
    use_chunk_decoder = not args.legacy_full_decoder

    print(f"model_type={model_type}")
    print(f"onnx_root={onnx_root}")

    if model_type == "base":
        from voice_clone_ort import Qwen3TTSVoiceCloneORT

        runner = Qwen3TTSVoiceCloneORT(
            model_dir=args.model,
            onnx_root=onnx_root,
            providers=providers,
            seed=args.seed,
            use_iobinding=not args.no_iobinding,
        )
        wav, sample_rate, codes = runner.generate_voice_clone(
            text=args.text,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            language=args.language,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            subtalker_dosample=do_sample,
            verbose=True,
        )
    elif model_type == "custom_voice":
        from custom_voice_ort import Qwen3TTSCustomVoiceORT

        ensure_chunk_decoder(onnx_root, use_chunk_decoder)
        runner = Qwen3TTSCustomVoiceORT(
            model_dir=args.model,
            onnx_root=onnx_root,
            providers=providers,
            seed=args.seed,
            use_iobinding=not args.no_iobinding,
        )
        wav, sample_rate, codes = runner.generate_custom_voice(
            text=args.text,
            language=args.language,
            speaker=args.speaker,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            subtalker_dosample=do_sample,
            use_chunk_decoder=use_chunk_decoder,
            chunk_frames=args.chunk_frames,
            left_context_frames=args.left_context_frames,
            verbose=True,
        )
    elif model_type == "voice_design":
        from voice_design_ort import Qwen3TTSVoiceDesignORT

        ensure_chunk_decoder(onnx_root, use_chunk_decoder)
        runner = Qwen3TTSVoiceDesignORT(
            model_dir=args.model,
            onnx_root=onnx_root,
            providers=providers,
            seed=args.seed,
            use_iobinding=not args.no_iobinding,
        )
        wav, sample_rate, codes = runner.generate_voice_design(
            text=args.text,
            language=args.language,
            instruct=args.instruct,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            subtalker_dosample=do_sample,
            use_chunk_decoder=use_chunk_decoder,
            chunk_frames=args.chunk_frames,
            left_context_frames=args.left_context_frames,
            verbose=True,
        )
    else:
        raise ValueError(f"Unsupported tts_model_type={model_type!r}")

    sf.write(args.output, wav, sample_rate)
    print(f"wrote {args.output}: samples={wav.shape[0]}, sr={sample_rate}, generated_frames={codes.shape[0]}")


if __name__ == "__main__":
    main()
