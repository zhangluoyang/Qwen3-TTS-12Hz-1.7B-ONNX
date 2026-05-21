#!/usr/bin/env python3
"""非流式 Qwen3-TTS ONNX 声音克隆 Gradio UI。

页面背后只调用 `Qwen3TTSVoiceCloneORT.generate_voice_clone()`：
先生成完整 codec 序列，再一次性解码成 wav。因此它适合验证音质和功能，
不用于测试 chunk 延迟。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort

from voice_clone_ort import DEFAULT_MODEL, Qwen3TTSVoiceCloneORT


DEFAULT_TEXT = "你好，这是使用 ONNX Runtime 进行非流式声音克隆的测试。"
DEFAULT_REF_AUDIO = "./data/ref_from_mp3_24k_mono.wav"
DEFAULT_REF_TEXT = "告诉自己，不要怕"


def build_runner(model_dir: str, onnx_root: str, provider: str, use_iobinding: bool) -> Qwen3TTSVoiceCloneORT:
    """创建一个可复用 runner；Gradio 每次点击不重新加载 ONNX session。"""
    ort.set_default_logger_severity(3)
    return Qwen3TTSVoiceCloneORT(
        model_dir=model_dir,
        onnx_root=onnx_root,
        providers=[provider],
        print_timing=False,
        use_iobinding=use_iobinding,
    )


def make_infer_fn(runner: Qwen3TTSVoiceCloneORT):
    """把 runner 包成 Gradio 回调函数。"""
    def infer(
        text: str,
        ref_audio,
        ref_text: str,
        max_new_tokens: int,
        greedy: bool,
    ):
        if not text.strip():
            raise gr.Error("请输入要合成的文本。")

        ref_audio_path = ref_audio if isinstance(ref_audio, str) else DEFAULT_REF_AUDIO
        # Gradio 的 Audio 组件可能返回路径，也可能为空；统一转换成可读文件路径。
        if not ref_audio_path or not Path(ref_audio_path).exists():
            raise gr.Error(f"参考音频不存在: {ref_audio_path}")

        wav, sample_rate, codes = runner.generate_voice_clone(
            text=text,
            ref_audio=ref_audio_path,
            ref_text=ref_text,
            max_new_tokens=int(max_new_tokens),
            do_sample=not greedy,
            subtalker_dosample=not greedy,
            verbose=True,
        )
        print(
            f"[gradio] done frames={codes.shape[0]} "
            f"samples={wav.shape[0]} sr={sample_rate} seconds={wav.shape[0] / float(sample_rate):.2f}"
        )
        return sample_rate, wav.astype(np.float32, copy=False)

    return infer


def main():
    # CLI 参数主要控制模型目录、ONNX 目录、provider 和是否启用 I/O Binding。
    parser = argparse.ArgumentParser(description="Launch a full-output Gradio UI for Qwen3-TTS ONNX voice clone.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--onnx-root", default="./onnx_isolated_fp16")
    parser.add_argument("--provider", default="CUDAExecutionProvider")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-iobinding", action="store_true", help="Disable Python ONNX Runtime I/O Binding")
    args = parser.parse_args()

    runner = build_runner(args.model, args.onnx_root, args.provider, not args.no_iobinding)
    infer = make_infer_fn(runner)

    with gr.Blocks(title="Qwen3-TTS ONNX Voice Clone") as demo:
        gr.Markdown("## Qwen3-TTS ONNX 非流式声音克隆")
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="输入文本", value=DEFAULT_TEXT, lines=5)
                ref_audio = gr.Audio(label="参考音频", value=DEFAULT_REF_AUDIO, type="filepath")
                ref_text = gr.Textbox(label="参考文本", value=DEFAULT_REF_TEXT)
                with gr.Row():
                    max_new_tokens = gr.Slider(16, 2048, value=120, step=1, label="max_new_tokens")
                    greedy = gr.Checkbox(value=True, label="greedy")
                generate_btn = gr.Button("生成", variant="primary")
            with gr.Column():
                output_audio = gr.Audio(label="输出音频", autoplay=True)

        generate_btn.click(
            infer,
            inputs=[
                text,
                ref_audio,
                ref_text,
                max_new_tokens,
                greedy,
            ],
            outputs=output_audio,
        )

    demo.queue()
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
