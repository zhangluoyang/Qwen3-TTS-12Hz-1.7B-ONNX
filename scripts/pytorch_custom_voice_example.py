#!/usr/bin/env python3
"""Qwen3-TTS 官方 PyTorch CustomVoice 最小示例。"""

from pathlib import Path

import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel


MODEL_DIR = Path("/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
OUTPUT_WAV = Path("output_pytorch_custom_voice_example.wav")


def main():
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = Qwen3TTSModel.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=dtype,
        device_map=device_map,
        local_files_only=True,
    )

    wavs, sample_rate = model.generate_custom_voice(
        text="你好，这是 Qwen 三自定义音色的 PyTorch 测试。",
        language="Chinese",
        speaker="Vivian",
        instruct="用自然、清晰的语气说",
        max_new_tokens=120,
    )

    sf.write(OUTPUT_WAV, wavs[0], sample_rate)
    print(f"wrote {OUTPUT_WAV}: samples={wavs[0].shape[0]}, sr={sample_rate}")


if __name__ == "__main__":
    main()
