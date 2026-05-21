#!/usr/bin/env python3
"""导出的 Qwen3-TTS CustomVoice ONNX 模型的最小 Python 示例。"""

from pathlib import Path

import soundfile as sf

from custom_voice_ort import Qwen3TTSCustomVoiceORT


MODEL_DIR = Path("/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
ONNX_ROOT = Path("onnx_custom_voice_0p6b_fp32")
OUTPUT_WAV = Path("output_python_custom_voice_example.wav")


def main():
    runner = Qwen3TTSCustomVoiceORT(
        model_dir=MODEL_DIR,
        onnx_root=ONNX_ROOT,
        providers=["CUDAExecutionProvider"],
        seed=1234,
    )

    wav, sample_rate, codes = runner.generate_custom_voice(
        text="你好，这是 Qwen 三自定义音色的 ONNX Runtime 测试。",
        language="Chinese",
        speaker="Vivian",
        max_new_tokens=120,
        verbose=True,
    )

    sf.write(OUTPUT_WAV, wav, sample_rate)
    print(f"wrote {OUTPUT_WAV}: samples={wav.shape[0]}, sr={sample_rate}, generated_frames={codes.shape[0]}")


if __name__ == "__main__":
    main()
