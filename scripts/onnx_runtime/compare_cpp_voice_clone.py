#!/usr/bin/env python3
"""比较 Python ONNX Runtime 与 C++ Qwen3-TTS 声音克隆导出的调试张量。

典型用法：
先用 Python/C++ 分别指定 --dump-dir 生成 .npy，再运行本脚本。
整数张量要求完全一致；浮点张量输出 max/mean abs diff，帮助定位差异来自
前端、prompt、prefill、decode 还是 vocoder。
"""

import argparse
from pathlib import Path

import numpy as np


TENSORS = [
    "assistant_text_ids",
    "reference_text_ids",
    "audio_24k",
    "mel",
    "reference_codes",
    "speaker_embedding",
    "generated_codes",
    "waveform",
]


def compare_one(name: str, py_dir: Path, cpp_dir: Path) -> bool:
    """比较同名 .npy；返回 True 表示形状正确且整数完全一致。"""
    py_path = py_dir / f"{name}.npy"
    cpp_path = cpp_dir / f"{name}.npy"
    if not py_path.exists() or not cpp_path.exists():
        print(f"{name:20s} missing py={py_path.exists()} cpp={cpp_path.exists()}")
        return False

    py = np.load(py_path)
    cpp = np.load(cpp_path)
    ok_shape = py.shape == cpp.shape
    if not ok_shape:
        print(f"{name:20s} SHAPE py={py.shape} cpp={cpp.shape}")
        return False

    if np.issubdtype(py.dtype, np.integer):
        # token ids / codec codes 必须完全一致，否则后续浮点对齐没有意义。
        same = np.array_equal(py, cpp)
        if same:
            print(f"{name:20s} OK int shape={py.shape}")
            return True
        diff = np.flatnonzero(py.reshape(-1) != cpp.reshape(-1))
        i = int(diff[0])
        print(f"{name:20s} DIFF int shape={py.shape} first={i} py={py.reshape(-1)[i]} cpp={cpp.reshape(-1)[i]}")
        return False

    # 浮点张量允许细小误差，尤其 FP16/CUDA 路径会有舍入差异。
    d = py.astype(np.float64) - cpp.astype(np.float64)
    max_abs = float(np.max(np.abs(d))) if d.size else 0.0
    mean_abs = float(np.mean(np.abs(d))) if d.size else 0.0
    denom = np.maximum(np.abs(py.astype(np.float64)), 1e-8)
    max_rel = float(np.max(np.abs(d) / denom)) if d.size else 0.0
    print(f"{name:20s} shape={py.shape} max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} max_rel={max_rel:.6g}")
    return max_abs < 1e-3 or mean_abs < 1e-4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-dir", required=True)
    parser.add_argument("--cpp-dir", required=True)
    args = parser.parse_args()

    py_dir = Path(args.python_dir)
    cpp_dir = Path(args.cpp_dir)
    all_ok = True
    for name in TENSORS:
        all_ok = compare_one(name, py_dir, cpp_dir) and all_ok
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
