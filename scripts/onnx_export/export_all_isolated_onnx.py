#!/usr/bin/env python3
"""把所有独立 ONNX 子模型导出到隔离目录。

这个脚本是“总控导出器”：它顺序调用其它 verify/export 脚本，把 Qwen3-TTS
拆成 runtime 需要的多个子模型目录。默认 float32 会导出并校验，float16
通常跳过校验以节省时间。
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_MODEL = "/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def run_step(name: str, cmd: list[str]) -> None:
    """打印并执行一个导出步骤；失败时 subprocess.run(check=True) 会中断全流程。"""
    print("\n" + "=" * 80, flush=True)
    print(f"[{name}]", flush=True)
    print(" ".join(cmd), flush=True)
    print("=" * 80, flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    # common 是所有导出脚本共享的模型目录、设备和 dtype 参数。
    parser = argparse.ArgumentParser(description="Export all ONNX models into isolated directories")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=str, default="./onnx_isolated")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--clean", action="store_true", help="Delete output-root before exporting")
    args = parser.parse_args()

    root = Path(args.output_root)
    if args.clean and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    py = sys.executable
    common = ["--model", args.model, "--device", args.device, "--dtype", args.dtype]
    verify_args = [] if args.dtype == "float32" else ["--skip-verify"]
    talker_verify_args = ["--atol", "2e-3"] if args.dtype == "float32" else []

    run_step(
        "tokenizer12hz",
        [
            py,
            str(script_dir / "export_tokenizer12hz_onnx.py"),
            *common,
            "--output-dir",
            str(root / "tokenizer12hz"),
            *(["--verify"] if args.dtype == "float32" else []),
        ],
    )

    run_step(
        "text_project",
        [
            py,
            str(script_dir / "verify_embed_onnx.py"),
            *common,
            "--output-dir",
            str(root / "text_project"),
            "--only",
            "text_project",
            *verify_args,
        ],
    )

    run_step(
        "codec_embed",
        [
            py,
            str(script_dir / "verify_embed_onnx.py"),
            *common,
            "--output-dir",
            str(root / "codec_embed"),
            "--only",
            "codec_embed",
            *verify_args,
        ],
    )

    run_step(
        "code_predictor_embed",
        [
            py,
            str(script_dir / "verify_code_predictor_embed_onnx.py"),
            *common,
            "--output-dir",
            str(root / "code_predictor_embed"),
            *verify_args,
        ],
    )

    run_step(
        "code_predictor",
        [
            py,
            str(script_dir / "verify_code_predictor_onnx.py"),
            *common,
            "--output-dir",
            str(root / "code_predictor"),
            *verify_args,
        ],
    )

    run_step(
        "talker_prefill",
        [
            py,
            str(script_dir / "verify_talker_prefill_onnx.py"),
            *common,
            "--output-dir",
            str(root / "talker_prefill"),
            "--seq-len",
            "1",
            "--extra-seq-lens",
            "2,3,4,5,7,8,9,12,16,20,24,32,48,64",
            *talker_verify_args,
            *verify_args,
        ],
    )

    run_step(
        "talker_decode",
        [
            py,
            str(script_dir / "verify_talker_decode_onnx.py"),
            *common,
            "--output-dir",
            str(root / "talker_decode"),
            "--past-len",
            "1",
            "--extra-past-lens",
            "2,3,4,5,7,8,9,12,16,20,24,32,48,64",
            *talker_verify_args,
            *verify_args,
        ],
    )

    run_step(
        "speaker_encoder",
        [
            py,
            str(script_dir / "verify_speaker_encoder_onnx.py"),
            *common,
            "--output-dir",
            str(root / "speaker_encoder"),
            *verify_args,
        ],
    )

    print("\nAll isolated ONNX exports and verifications completed.", flush=True)
    print(f"DType: {args.dtype}", flush=True)
    print(f"Output root: {root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
