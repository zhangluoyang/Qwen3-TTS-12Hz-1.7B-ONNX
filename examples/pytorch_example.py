#!/usr/bin/env python3
"""Single PyTorch example for Qwen3-TTS Base / CustomVoice / VoiceDesign."""

import argparse
import json
from pathlib import Path

import soundfile as sf


DEFAULT_MODEL = "/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_TEXT = "你好，这是 Qwen 三 PyTorch 统一样例。"


def load_model_config(model: Path) -> dict:
    config_path = model.expanduser() / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def infer_model_type(model: Path, config: dict) -> str:
    model_type = config.get("tts_model_type")
    if model_type:
        return str(model_type)
    name = model.name.lower()
    if "customvoice" in name or "custom_voice" in name:
        return "custom_voice"
    if "voicedesign" in name or "voice_design" in name:
        return "voice_design"
    return "base"


def torch_dtype(name: str):
    import torch

    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def default_dtype(model_type: str) -> str:
    import torch

    if not torch.cuda.is_available():
        return "float32"
    return "bfloat16" if model_type in {"custom_voice", "voice_design"} else "float16"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one Qwen3-TTS PyTorch example.")
    parser.add_argument("--model", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--output", type=Path, default=Path("output_pytorch_example.wav"))
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--max-new-tokens", type=int, default=600)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default=None)

    parser.add_argument("--ref-audio", type=Path, default=Path("data/ref_from_mp3_24k_mono.wav"))
    parser.add_argument("--ref-text", default="告诉自己，不要怕")
    parser.add_argument("--speaker", default="Vivian")
    parser.add_argument("--instruct", default="一个年轻女性的声音，语气温柔自然，语速适中，发音清晰。")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from qwen_tts import Qwen3TTSModel

    config = load_model_config(args.model)
    model_type = infer_model_type(args.model, config)

    device_map = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype_name = args.dtype or default_dtype(model_type)

    print(f"model_type={model_type}")
    print(f"device={device_map}")
    print(f"dtype={dtype_name}")

    model = Qwen3TTSModel.from_pretrained(
        str(args.model),
        dtype=torch_dtype(dtype_name),
        device_map=device_map,
        local_files_only=True,
    )

    if model_type == "base":
        wavs, sample_rate = model.generate_voice_clone(
            text=args.text,
            language=args.language,
            ref_audio=str(args.ref_audio),
            ref_text=args.ref_text,
            max_new_tokens=args.max_new_tokens,
        )
    elif model_type == "custom_voice":
        wavs, sample_rate = model.generate_custom_voice(
            text=args.text,
            language=args.language,
            speaker=args.speaker,
            instruct=args.instruct,
            max_new_tokens=args.max_new_tokens,
        )
    elif model_type == "voice_design":
        wavs, sample_rate = model.generate_voice_design(
            text=args.text,
            language=args.language,
            instruct=args.instruct,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        raise ValueError(f"Unsupported tts_model_type={model_type!r}")

    wav = wavs[0]
    sf.write(args.output, wav, sample_rate)
    print(f"wrote {args.output}: samples={wav.shape[0]}, sr={sample_rate}")


if __name__ == "__main__":
    main()
