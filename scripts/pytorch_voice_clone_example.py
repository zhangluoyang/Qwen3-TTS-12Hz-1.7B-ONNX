#!/usr/bin/env python3
"""Qwen3-TTS 官方 PyTorch 声音克隆示例。

这个脚本直接使用 qwen_tts 的 PyTorch/HuggingFace wrapper，不使用导出的 ONNX 模型。
"""

from pathlib import Path

import soundfile as sf
import torch

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


MODEL_DIR = Path("/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base")
REFERENCE_AUDIO = Path("data/ref_from_mp3_24k_mono.wav")
OUTPUT_WAV = Path("output_pytorch_voice_clone_example.wav")


def main():
    # 这个脚本用于和 ONNX 结果做听感/流程对比：同一段文本、同一段参考音频，
    # 但走官方 PyTorch wrapper，不走导出的 ONNX 子图。
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = Qwen3TTSModel.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=dtype,
        device_map=device_map,
        local_files_only=True,
    )

    wavs, sample_rate = model.generate_voice_clone(
        text="你好呀！很高兴认识你！👋、你、我是DeepSeek，由深度求索公司创造的AI助手。简单来说，我就是你的智能小伙伴，随时准备帮你解答问题、处理任务或者陪你聊天！我正在做什么呢？此刻，我正在认真阅读你的消息，思考怎么给你最好的回答。我的日常就是：倾听你的问题 → 快速检索知识 → 给出清晰有用的回复。不管是学习、工作、生活中的疑惑，还是想找人聊聊想法，我都乐意奉陪！有什么我可以帮你的吗？尽管说，别客气～😊",
        language="auto",
        ref_audio=str(REFERENCE_AUDIO),
        ref_text="告诉自己，不要怕",
        x_vector_only_mode=False,
        max_new_tokens=512,
        do_sample=True,
        top_k=50,
        top_p=1.0,
        temperature=0.01,
        subtalker_dosample=True,
        subtalker_top_k=50,
        subtalker_top_p=1.0,
        subtalker_temperature=0.01
    )

    sf.write(OUTPUT_WAV, wavs[0], sample_rate)
    print(f"wrote {OUTPUT_WAV}: samples={wavs[0].shape[0]}, sr={sample_rate}")


if __name__ == "__main__":
    main()
