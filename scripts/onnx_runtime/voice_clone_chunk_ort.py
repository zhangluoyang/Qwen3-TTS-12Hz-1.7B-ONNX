#!/usr/bin/env python3
"""Chunk/pipeline Qwen3-TTS ONNX voice clone CLI.

This script keeps the original full-output CLI untouched. It consumes the new
generator-style chunk API and writes the concatenated waveform at the end.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

from audio_postprocess import concatenate_audio_chunks, milliseconds_to_samples
from voice_clone_ort import DEFAULT_MODEL, Qwen3TTSVoiceCloneORT, Timer


def main() -> None:
    # 这个 CLI 是 chunk/pipeline 的最小命令行验证入口。
    # 完整非流式基线在 voice_clone_ort.py 的 main() 中。
    parser = argparse.ArgumentParser(description="Run chunk/pipeline Qwen3-TTS ONNX voice clone.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--onnx-root", default="./onnx_isolated_fp16")
    parser.add_argument("--text", default="你好，这是 Python chunk 流水线解码测试。")
    parser.add_argument("--ref-audio", default="./data/ref_from_mp3_24k_mono.wav")
    parser.add_argument("--ref-text", default="告诉自己，不要怕")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--output", default="output_voice_clone_chunk_ort.wav")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--chunk-frames", type=int, default=50)
    parser.add_argument("--left-context-frames", type=int, default=25)
    parser.add_argument("--crossfade-ms", type=float, default=0.0, help="Optional crossfade between decoded chunks")
    parser.add_argument("--provider", default="CUDAExecutionProvider")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--greedy", action="store_true", help="Disable sampling for the main talker and code predictor")
    parser.add_argument("--no-timing", action="store_true", help="Disable timing summary output")
    parser.add_argument("--timing-json", default=None, help="Write detailed timing records to a JSON file")
    parser.add_argument("--no-iobinding", action="store_true", help="Disable Python ONNX Runtime I/O Binding")
    parser.add_argument("--dump-dir", default=None, help="Dump intermediate tensors for Python/C++ comparison")
    args = parser.parse_args()

    ort.set_default_logger_severity(3)

    providers = [args.provider]
    total_timer = Timer()
    with total_timer.measure("total.init_runner"):
        runner = Qwen3TTSVoiceCloneORT(
            args.model,
            args.onnx_root,
            providers=providers,
            seed=args.seed,
            print_timing=not args.no_timing,
            use_iobinding=not args.no_iobinding,
        )

    chunks = []
    sample_rate = 24000
    generated_frames = 0
    with total_timer.measure("total.generate_voice_clone_chunked"):
        # iter_voice_clone_chunked 是 generator：每完成一个 chunk decoder 就返回一段音频。
        for item in runner.iter_voice_clone_chunked(
            text=args.text,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            language=args.language,
            max_new_tokens=args.max_new_tokens,
            chunk_frames=args.chunk_frames,
            left_context_frames=args.left_context_frames,
            do_sample=not args.greedy,
            subtalker_dosample=not args.greedy,
            dump_dir=args.dump_dir,
            verbose=True,
        ):
            sample_rate = item.sample_rate
            generated_frames = item.generated_frames
            chunks.append(item.audio.astype(np.float32, copy=False))

    if not chunks:
        raise RuntimeError("No audio chunks were produced")

    # 可选 crossfade 只影响最终拼接，不改变每个 chunk decoder 的原始输出。
    wav = concatenate_audio_chunks(chunks, milliseconds_to_samples(args.crossfade_ms, sample_rate))
    if args.dump_dir:
        dump_dir = Path(args.dump_dir)
        np.save(dump_dir / "waveform.npy", wav.astype(np.float32))
    with total_timer.measure("total.write_wav"):
        sf.write(args.output, wav, sample_rate)
    print(
        f"wrote {args.output}: samples={wav.shape[0]}, sr={sample_rate}, "
        f"generated_frames={generated_frames}, chunks={len(chunks)}"
    )
    if not args.no_timing:
        total_timer.print_summary("[Timing] Overall")
        runner.timer.print_summary("[Timing] Detail")
    if args.timing_json:
        detail_path = Path(args.timing_json)
        total_timer.write_json(
            detail_path,
            extra={
                "kind": "voice_clone_chunk",
                "output": args.output,
                "generated_frames": int(generated_frames),
                "chunks": len(chunks),
                "samples": int(wav.shape[0]),
                "sample_rate": int(sample_rate),
                "chunk_frames": int(args.chunk_frames),
                "left_context_frames": int(args.left_context_frames),
                "crossfade_ms": float(args.crossfade_ms),
            },
        )
        runner.timer.write_json(
            detail_path.with_suffix(".detail.json"),
            extra={
                "kind": "voice_clone_chunk_detail",
                "output": args.output,
                "generated_frames": int(generated_frames),
                "chunks": len(chunks),
            },
        )


if __name__ == "__main__":
    main()
