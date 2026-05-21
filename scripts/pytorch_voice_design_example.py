#!/usr/bin/env python3
"""Qwen3-TTS official PyTorch VoiceDesign minimal example."""

from pathlib import Path

import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel


MODEL_DIR = Path("/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
OUTPUT_WAV = Path("output_pytorch_voice_design_example.wav")


def main():
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = Qwen3TTSModel.from_pretrained(
        str(MODEL_DIR),
        dtype=dtype,
        device_map=device_map,
        local_files_only=True,
    )

    wavs, sample_rate = model.generate_voice_design(
        text="你好，这是 Qwen 三音色设计的 PyTorch 测试。",
        language="Chinese",
        instruct="一个年轻女性的声音，语气温柔自然，语速适中，发音清晰。",
        max_new_tokens=120,
    )

    sf.write(OUTPUT_WAV, wavs[0], sample_rate)
    print(f"wrote {OUTPUT_WAV}: samples={wavs[0].shape[0]}, sr={sample_rate}")


if __name__ == "__main__":
    main()
