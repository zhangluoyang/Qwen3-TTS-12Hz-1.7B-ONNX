#!/usr/bin/env python3
"""生成原生 Python 与 ONNX Runtime tokenizer 音频，方便人工听感对比。

这个脚本读取一个音频文件，并分别运行两种实现：
  1. 原生 Python Qwen3TTSTokenizer encode/decode
  2. ONNX Runtime tokenizer12hz_encode + tokenizer12hz_decode_chunk

脚本会写出两份 wav 文件，方便直接试听和比较。
"""

import argparse
import os
import time
from typing import Any

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from transformers import AutoConfig, AutoProcessor

from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSProcessor
from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer


def load_speech_tokenizer(model_path: str, device: torch.device) -> Any:
    """加载原生 Python 版本的 12Hz speech tokenizer。

    Args:
        model_path: Qwen3-TTS 主模型目录。
        device: tokenizer 模型运行设备。

    Returns:
        已加载并切到 eval 模式的 Qwen3TTSTokenizer 对象。
    """
    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

    processor = AutoProcessor.from_pretrained(model_path, fix_mistral_regex=True)
    speech_tokenizer = getattr(processor, "speech_tokenizer", None)
    if speech_tokenizer is None:
        speech_tokenizer = getattr(processor, "audio_tokenizer", None)

    if speech_tokenizer is None:
        tokenizer_dir = os.path.join(model_path, "speech_tokenizer")
        speech_tokenizer = Qwen3TTSTokenizer.from_pretrained(
            tokenizer_dir,
            device_map=str(device),
        )

    speech_tokenizer.model.to(device)
    speech_tokenizer.model.eval()
    return speech_tokenizer


def load_audio(path: str, target_sr: int) -> np.ndarray:
    """读取音频并转成 mono 24k waveform。

    Args:
        path: 输入音频路径。
        target_sr: 目标采样率，Qwen3-TTS 12Hz tokenizer 使用 24000。

    Returns:
        shape 为 [num_samples] 的 float32 numpy 音频。
    """
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


def run_native(speech_tokenizer: Any, audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """运行原生 Python tokenizer encode/decode。

    Args:
        speech_tokenizer: 已加载的原生 tokenizer。
        audio: 输入 waveform，shape 为 [num_samples]。
        sample_rate: 输入音频采样率。

    Returns:
        原生 Python decode 得到的 waveform。
    """
    encoded = speech_tokenizer.encode(audio, sr=sample_rate, return_dict=True)
    wavs, out_sr = speech_tokenizer.decode(encoded)
    if out_sr != sample_rate:
        raise ValueError(f"Unexpected native output sample rate: {out_sr}")
    return wavs[0].astype(np.float32)


def run_onnx_chunked(
    encoder: ort.InferenceSession,
    decoder: ort.InferenceSession,
    audio: np.ndarray,
    chunk_size: int,
    left_context_size: int,
) -> np.ndarray:
    """运行 ONNX Runtime encode + chunk decode。

    Args:
        encoder: tokenizer12hz_encode.onnx 的 InferenceSession。
        decoder: tokenizer12hz_decode_chunk.onnx 的 InferenceSession。
        audio: 输入 waveform，shape 为 [num_samples]。
        chunk_size: 每个 current chunk 的 codec 帧数。
        left_context_size: 每个 chunk 携带的左上下文 codec 帧数。

    Returns:
        ONNX chunked decode 拼接后的 waveform。
    """
    codes = encoder.run(
        ["codes"],
        {"audio": audio.reshape(1, -1).astype(np.float32)},
    )[0].astype(np.int64)

    total_frames = codes.shape[1]
    chunks = []
    start_index = 0
    while start_index < total_frames:
        end_index = min(start_index + chunk_size, total_frames)
        context_size = (
            left_context_size
            if start_index - left_context_size > 0
            else start_index
        )
        codes_chunk = codes[:, start_index - context_size:end_index, :]
        wav_chunk, lengths = decoder.run(
            ["audio_values", "lengths"],
            {
                "audio_codes": codes_chunk,
                "context_frames": np.asarray(context_size, dtype=np.int64),
            },
        )

        expected_samples = (end_index - start_index) * 1920
        valid_samples = min(expected_samples, int(lengths.reshape(-1)[0]))
        chunks.append(wav_chunk[:, :valid_samples])
        start_index = end_index

    return np.concatenate(chunks, axis=1).reshape(-1).astype(np.float32)


def create_onnx_sessions(onnx_dir: str, use_cuda: bool) -> tuple[ort.InferenceSession, ort.InferenceSession]:
    """创建 ONNX Runtime sessions，加载耗时不计入纯推理 benchmark。

    Args:
        onnx_dir: 存放 tokenizer12hz_encode.onnx 和 tokenizer12hz_decode_chunk.onnx 的目录。
        use_cuda: 是否优先使用 CUDAExecutionProvider。

    Returns:
        二元组：(encoder session, decoder session)。
    """
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_cuda
        else ["CPUExecutionProvider"]
    )
    encoder = ort.InferenceSession(
        os.path.join(onnx_dir, "tokenizer12hz_encode.onnx"),
        providers=providers,
    )
    decoder = ort.InferenceSession(
        os.path.join(onnx_dir, "tokenizer12hz_decode_chunk.onnx"),
        providers=providers,
    )
    print(f"  ONNX encoder providers: {encoder.get_providers()}")
    print(f"  ONNX decoder providers: {decoder.get_providers()}")
    return encoder, decoder


def benchmark_native(
    speech_tokenizer: Any,
    audio: np.ndarray,
    sample_rate: int,
    device: torch.device,
    warmup_runs: int,
    benchmark_runs: int,
) -> tuple[np.ndarray, list[float]]:
    """对原生 Python tokenizer 做 warmup 和多轮纯推理计时。

    Args:
        speech_tokenizer: 已加载的原生 tokenizer。
        audio: 输入 waveform。
        sample_rate: 输入采样率。
        device: PyTorch 设备，用于 CUDA synchronize。
        warmup_runs: 预热次数，不计入耗时统计。
        benchmark_runs: 正式计时次数。

    Returns:
        二元组：(最后一次输出音频, 每轮耗时列表)。
    """
    for _ in range(warmup_runs):
        _ = run_native(speech_tokenizer, audio, sample_rate)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    outputs = None
    times = []
    for _ in range(benchmark_runs):
        start = time.perf_counter()
        outputs = run_native(speech_tokenizer, audio, sample_rate)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - start)

    return outputs, times


def benchmark_onnx(
    encoder: ort.InferenceSession,
    decoder: ort.InferenceSession,
    audio: np.ndarray,
    chunk_size: int,
    left_context_size: int,
    warmup_runs: int,
    benchmark_runs: int,
) -> tuple[np.ndarray, list[float]]:
    """对 ONNX Runtime tokenizer 做 warmup 和多轮纯推理计时。

    Args:
        encoder: tokenizer12hz_encode.onnx 的 InferenceSession。
        decoder: tokenizer12hz_decode_chunk.onnx 的 InferenceSession。
        audio: 输入 waveform。
        chunk_size: 每个 current chunk 的 codec 帧数。
        left_context_size: 每个 chunk 携带的左上下文 codec 帧数。
        warmup_runs: 预热次数，不计入耗时统计。
        benchmark_runs: 正式计时次数。

    Returns:
        二元组：(最后一次输出音频, 每轮耗时列表)。
    """
    for _ in range(warmup_runs):
        _ = run_onnx_chunked(encoder, decoder, audio, chunk_size, left_context_size)

    outputs = None
    times = []
    for _ in range(benchmark_runs):
        start = time.perf_counter()
        outputs = run_onnx_chunked(encoder, decoder, audio, chunk_size, left_context_size)
        times.append(time.perf_counter() - start)

    return outputs, times


def format_times(times: list[float]) -> str:
    """格式化多轮耗时统计。"""
    arr = np.asarray(times, dtype=np.float64)
    return (
        f"avg={arr.mean():.3f}s, min={arr.min():.3f}s, "
        f"max={arr.max():.3f}s, runs={[round(x, 3) for x in times]}"
    )


def main() -> None:
    """命令行入口：生成两份 wav 音频用于人工听感对比。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    )
    parser.add_argument("--onnx-dir", type=str, default="./tokenizer12hz_onnx_chunk")
    parser.add_argument("--input", type=str, default="./tokenizer_demo_1.wav")
    parser.add_argument("--output-dir", type=str, default="./compare_outputs")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--left-context-size", type=int, default=25)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--benchmark-runs", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sample_rate = 24000
    audio = load_audio(args.input, target_sr=sample_rate)
    stem = os.path.splitext(os.path.basename(args.input))[0]

    print(f"Input: {args.input}, samples={audio.shape[0]}, sr={sample_rate}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda:0, but CUDA is not available.")

    print("Loading native Python tokenizer...")
    speech_tokenizer = load_speech_tokenizer(args.model, device)

    print("Loading ONNX Runtime sessions...")
    encoder_session, decoder_session = create_onnx_sessions(
        args.onnx_dir,
        use_cuda=device.type == "cuda",
    )

    print(f"Benchmark warmup_runs={args.warmup_runs}, benchmark_runs={args.benchmark_runs}")
    print("Running native Python tokenizer benchmark...")
    native_audio, native_times = benchmark_native(
        speech_tokenizer,
        audio,
        sample_rate,
        device,
        args.warmup_runs,
        args.benchmark_runs,
    )

    print("Running ONNX Runtime tokenizer benchmark...")
    onnx_audio, onnx_times = benchmark_onnx(
        encoder_session,
        decoder_session,
        audio,
        args.chunk_size,
        args.left_context_size,
        args.warmup_runs,
        args.benchmark_runs,
    )

    native_path = os.path.join(args.output_dir, f"{stem}_native_python.wav")
    onnx_path = os.path.join(args.output_dir, f"{stem}_onnxruntime.wav")
    sf.write(native_path, native_audio, sample_rate)
    sf.write(onnx_path, onnx_audio, sample_rate)

    print(f"Saved native Python audio: {native_path}")
    print(f"Saved ONNX Runtime audio: {onnx_path}")
    print("Timing:")
    print(f"  Native Python ({args.device}): {format_times(native_times)}")
    print(f"  ONNX Runtime ({'CUDAExecutionProvider' if device.type == 'cuda' else 'CPUExecutionProvider'}): {format_times(onnx_times)}")


if __name__ == "__main__":
    main()
