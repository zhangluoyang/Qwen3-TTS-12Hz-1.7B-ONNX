#!/usr/bin/env python3
"""Minimal Python example for exported Qwen3-TTS VoiceDesign ONNX models."""

from pathlib import Path

import soundfile as sf

from voice_design_ort import Qwen3TTSVoiceDesignORT


MODEL_DIR = Path("/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
ONNX_ROOT = Path("onnx_voice_design_1p7b_fp16")
OUTPUT_WAV = Path("output_python_voice_design_example.wav")


def main():
    runner = Qwen3TTSVoiceDesignORT(
        model_dir=MODEL_DIR,
        onnx_root=ONNX_ROOT,
        providers=["CUDAExecutionProvider"],
        seed=1234,
    )

    wav, sample_rate, codes = runner.generate_voice_design(
        text="你好，这是 Qwen 三音色设计的 ONNX Runtime 测试。",
        language="Chinese",
        instruct="一个年轻女性的声音，语气温柔自然，语速适中，发音清晰。",
        max_new_tokens=120,
        verbose=True,
    )

    sf.write(OUTPUT_WAV, wav, sample_rate)
    print(f"wrote {OUTPUT_WAV}: samples={wav.shape[0]}, sr={sample_rate}, generated_frames={codes.shape[0]}")


if __name__ == "__main__":
    main()
