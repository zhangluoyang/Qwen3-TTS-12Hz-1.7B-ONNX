#!/usr/bin/env python3
"""chunk/pipeline Qwen3-TTS ONNX 声音克隆 Gradio UI。

它调用 `iter_voice_clone_chunked()` 收集多段音频，再拼成最终输出。
UI 里暴露 chunk_frames、left_context_frames、crossfade_ms，方便学习它们
对延迟和边界连续性的影响。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort

from audio_postprocess import concatenate_audio_chunks, milliseconds_to_samples
from voice_clone_ort import DEFAULT_MODEL, Qwen3TTSVoiceCloneORT


DEFAULT_TEXT = "你好，这是使用 ONNX Runtime 进行 chunk 流水线声音克隆的测试。"
DEFAULT_REF_AUDIO = "./data/ref_from_mp3_24k_mono.wav"
DEFAULT_REF_TEXT = "告诉自己，不要怕"


def build_runner(model_dir: str, onnx_root: str, provider: str, use_iobinding: bool) -> Qwen3TTSVoiceCloneORT:
    """创建一个长生命周期 runner，避免每次推理重新加载 ONNX 模型。"""
    ort.set_default_logger_severity(3)
    return Qwen3TTSVoiceCloneORT(
        model_dir=model_dir,
        onnx_root=onnx_root,
        providers=[provider],
        print_timing=False,
        use_iobinding=use_iobinding,
    )


def make_infer_fn(runner: Qwen3TTSVoiceCloneORT):
    """把 chunk generator 包装成 Gradio 的流式输出回调。"""
    def infer(
        text: str,
        ref_audio,
        ref_text: str,
        max_new_tokens: int,
        chunk_frames: int,
        left_context_frames: int,
        crossfade_ms: float,
        greedy: bool,
    ):
        if not text.strip():
            raise gr.Error("请输入要合成的文本。")

        ref_audio_path = ref_audio if isinstance(ref_audio, str) else DEFAULT_REF_AUDIO
        if not ref_audio_path or not Path(ref_audio_path).exists():
            raise gr.Error(f"参考音频不存在: {ref_audio_path}")

        chunks = []
        sample_rate = 24000
        # generator 每 yield 一段 PipelineAudioChunk，这里逐步拼成当前已有音频返回给 Gradio。
        for item in runner.iter_voice_clone_chunked(
            text=text,
            ref_audio=ref_audio_path,
            ref_text=ref_text,
            max_new_tokens=int(max_new_tokens),
            chunk_frames=int(chunk_frames),
            left_context_frames=int(left_context_frames),
            do_sample=not greedy,
            subtalker_dosample=not greedy,
            verbose=True,
        ):
            sample_rate = item.sample_rate
            chunks.append(item.audio.astype(np.float32, copy=False))
            wav_so_far = concatenate_audio_chunks(
                chunks,
                milliseconds_to_samples(crossfade_ms, sample_rate),
            )
            print(
                f"[gradio-chunk] frames={item.start_frame}:{item.end_frame} "
                f"generated={item.generated_frames} samples={wav_so_far.shape[0]}"
            )
            yield sample_rate, wav_so_far

    return infer


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a chunk/pipeline Gradio UI for Qwen3-TTS ONNX voice clone.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--onnx-root", default="./onnx_isolated_fp16")
    parser.add_argument("--provider", default="CUDAExecutionProvider")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--no-iobinding", action="store_true", help="Disable Python ONNX Runtime I/O Binding")
    args = parser.parse_args()

    runner = build_runner(args.model, args.onnx_root, args.provider, not args.no_iobinding)
    infer = make_infer_fn(runner)

    with gr.Blocks(title="Qwen3-TTS ONNX Chunk Voice Clone") as demo:
        gr.Markdown("## Qwen3-TTS ONNX Chunk 流水线声音克隆")
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="输入文本", value=DEFAULT_TEXT, lines=5)
                ref_audio = gr.Audio(label="参考音频", value=DEFAULT_REF_AUDIO, type="filepath")
                ref_text = gr.Textbox(label="参考文本", value=DEFAULT_REF_TEXT)
                with gr.Row():
                    max_new_tokens = gr.Slider(16, 2048, value=160, step=1, label="max_new_tokens")
                    chunk_frames = gr.Slider(5, 300, value=50, step=1, label="chunk_frames")
                    left_context_frames = gr.Slider(0, 100, value=25, step=1, label="left_context_frames")
                    crossfade_ms = gr.Slider(0, 120, value=0, step=1, label="crossfade_ms")
                greedy = gr.Checkbox(value=True, label="greedy")
                generate_btn = gr.Button("生成", variant="primary")
            with gr.Column():
                output_audio = gr.Audio(label="Chunk 输出音频", autoplay=True)

        generate_btn.click(
            infer,
            inputs=[
                text,
                ref_audio,
                ref_text,
                max_new_tokens,
                chunk_frames,
                left_context_frames,
                crossfade_ms,
                greedy,
            ],
            outputs=output_audio,
        )

    demo.queue()
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
