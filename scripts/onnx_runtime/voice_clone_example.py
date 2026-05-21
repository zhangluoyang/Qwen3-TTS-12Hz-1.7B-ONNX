#!/usr/bin/env python3
"""导出的 Qwen3-TTS ONNX 模型的最小 Python 声音克隆示例。

它只展示“如何实例化 Qwen3TTSVoiceCloneORT 并调用 generate_voice_clone()”。
要看完整推理细节，请读 voice_clone_ort.py。
"""

from pathlib import Path

import soundfile as sf

from voice_clone_ort import Qwen3TTSVoiceCloneORT


MODEL_DIR = Path("/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base")
ONNX_ROOT = Path("onnx_isolated_fp16")
REFERENCE_AUDIO = Path("data/ref_from_mp3_24k_mono.wav")
OUTPUT_WAV = Path("output_python_voice_clone_fp16_example.wav")


def main():
    # providers 可以换成 CPUExecutionProvider；FP16 ONNX 通常建议 CUDA。
    runner = Qwen3TTSVoiceCloneORT(
        model_dir=MODEL_DIR,
        onnx_root=ONNX_ROOT,
        providers=["CUDAExecutionProvider"],
        seed=1234,
    )

    wav, sample_rate, codes = runner.generate_voice_clone(
        text="你好，这是使用 MP3 参考音频进行声音克隆的测试。",
        ref_audio=REFERENCE_AUDIO,
        ref_text="告诉自己，不要怕",
        language="auto",
        max_new_tokens=80,
        do_sample=True,
        subtalker_dosample=True,
        temperature=0.01,
        subtalker_temperature=0.01,
        verbose=True,
    )

    sf.write(OUTPUT_WAV, wav, sample_rate)
    print(f"wrote {OUTPUT_WAV}: samples={wav.shape[0]}, sr={sample_rate}, generated_frames={codes.shape[0]}")


if __name__ == "__main__":
    main()
