"""Python ONNX Runtime demos 共用的音频后处理工具。

chunk/pipeline 解码会返回多段 mono float32 PCM。这里负责把它们拼成一段，
并可选做短 crossfade，降低 chunk 边界的突兀感。
"""

from __future__ import annotations

import numpy as np


def concatenate_audio_chunks(chunks, crossfade_samples=0):
    """拼接 mono audio chunks，并可选做 equal-power crossfade。

    Args:
        chunks: 每段 shape [samples] 的 numpy 数组。
        crossfade_samples: 相邻 chunk 重叠混合的采样点数；0 表示直接拼接。
    """
    chunks = [chunk.astype(np.float32, copy=False).reshape(-1) for chunk in chunks if chunk is not None and len(chunk)]
    if not chunks:
        return np.zeros((0,), dtype=np.float32)

    crossfade_samples = max(int(crossfade_samples), 0)
    output = chunks[0].copy()
    for chunk in chunks[1:]:
        fade = min(crossfade_samples, output.shape[0], chunk.shape[0])
        if fade <= 0:
            output = np.concatenate([output, chunk]).astype(np.float32, copy=False)
            continue

        t = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
        # equal-power 曲线比线性淡入淡出更不容易在中间点听起来变小。
        fade_out = np.cos(t * np.pi * 0.5)
        fade_in = np.sin(t * np.pi * 0.5)
        mixed = output[-fade:] * fade_out + chunk[:fade] * fade_in
        output = np.concatenate([output[:-fade], mixed, chunk[fade:]]).astype(np.float32, copy=False)
    return output.astype(np.float32, copy=False)


def milliseconds_to_samples(milliseconds, sample_rate):
    """把毫秒转换成采样点数，供 UI/CLI 用更直观的 crossfade-ms 参数。"""
    return int(round(max(float(milliseconds), 0.0) * float(sample_rate) / 1000.0))
