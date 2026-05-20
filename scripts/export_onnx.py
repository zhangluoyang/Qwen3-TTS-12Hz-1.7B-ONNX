#!/usr/bin/env python3
"""User-facing ONNX export entrypoint for Qwen3-TTS.

This script wraps the lower-level exporters under scripts/onnx_export/ and keeps
the normal export flow in one command:

  1. export all isolated ONNX submodels;
  2. optionally export tokenizer12hz_decode_chunk.onnx;
  3. optionally consolidate external data files.
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ONNX_EXPORT_DIR = REPO_ROOT / "scripts" / "onnx_export"
DEFAULT_MODEL = "/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def parse_dtype(value: str) -> str:
    """Accept both float16/float32 and fp16/fp32 spellings."""
    normalized = value.lower()
    aliases = {
        "fp16": "float16",
        "float16": "float16",
        "half": "float16",
        "fp32": "float32",
        "float32": "float32",
    }
    if normalized not in aliases:
        raise argparse.ArgumentTypeError("dtype must be float32/fp32 or float16/fp16")
    return aliases[normalized]


def default_output_root(dtype: str) -> Path:
    if dtype == "float16":
        return REPO_ROOT / "onnx_isolated_fp16"
    return REPO_ROOT / "onnx_isolated"


def default_device(dtype: str) -> str:
    if dtype == "float16":
        return "cuda"
    return "cpu"


def run_command(cmd: list[str], dry_run: bool) -> None:
    print("\n$ " + shlex.join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default=os.environ.get("QWEN3_TTS_MODEL_DIR", DEFAULT_MODEL),
        help="HuggingFace model id or local model directory. Can also be set with QWEN3_TTS_MODEL_DIR.",
    )
    parser.add_argument(
        "--dtype",
        type=parse_dtype,
        default="float32",
        help="Export dtype: float32/fp32 or float16/fp16. Default: float32.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for export. Use auto for cpu on float32 and cuda on float16.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Directory that will contain tokenizer12hz/, text_project/, talker_decode/, etc.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete output-root before exporting. Passed through to export_all_isolated_onnx.py.",
    )
    parser.add_argument(
        "--skip-consolidate",
        action="store_true",
        help="Do not merge ONNX external data files after export.",
    )
    parser.add_argument(
        "--keep-old-external-data",
        action="store_true",
        help="When consolidating, keep old split external-data files instead of removing them.",
    )
    parser.add_argument(
        "--with-chunk-decoder",
        action="store_true",
        help="Also export tokenizer12hz_decode_chunk.onnx for chunk/pipeline runtime.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Codec frame count used when tracing tokenizer12hz_decode_chunk.onnx.",
    )
    parser.add_argument(
        "--left-context-size",
        type=int,
        default=25,
        help="Left-context codec frame count used when tracing tokenizer12hz_decode_chunk.onnx.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export Qwen3-TTS ONNX submodels with the repository's isolated export flow. "
            "Examples: --dtype fp32 --clean, or --dtype fp16 --device cuda --clean."
        )
    )
    add_common_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    dtype = args.dtype
    device = default_device(dtype) if args.device == "auto" else args.device
    output_root = Path(args.output_root) if args.output_root else default_output_root(dtype)
    output_root = output_root.expanduser()
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root

    py = sys.executable
    export_all = ONNX_EXPORT_DIR / "export_all_isolated_onnx.py"
    tokenizer_export = ONNX_EXPORT_DIR / "export_tokenizer12hz_onnx.py"
    consolidate = ONNX_EXPORT_DIR / "consolidate_external_data.py"

    missing = [path for path in (export_all, tokenizer_export, consolidate) if not path.exists()]
    if missing:
        for path in missing:
            print(f"missing required script: {path}", file=sys.stderr)
        raise SystemExit(2)

    model_path = Path(args.model).expanduser()
    if model_path.is_absolute() and not model_path.exists():
        print(f"warning: local model path does not exist: {model_path}", file=sys.stderr)

    print("Qwen3-TTS ONNX export")
    print(f"  model:       {args.model}")
    print(f"  dtype:       {dtype}")
    print(f"  device:      {device}")
    print(f"  output-root: {output_root}")

    export_cmd = [
        py,
        str(export_all),
        "--model",
        args.model,
        "--output-root",
        str(output_root),
        "--device",
        device,
        "--dtype",
        dtype,
    ]
    if args.clean:
        export_cmd.append("--clean")
    run_command(export_cmd, args.dry_run)

    if args.with_chunk_decoder:
        chunk_cmd = [
            py,
            str(tokenizer_export),
            "--model",
            args.model,
            "--output-dir",
            str(output_root / "tokenizer12hz"),
            "--device",
            device,
            "--dtype",
            dtype,
            "--only-chunk-decoder",
            "--chunk-size",
            str(args.chunk_size),
            "--left-context-size",
            str(args.left_context_size),
        ]
        run_command(chunk_cmd, args.dry_run)

    if not args.skip_consolidate:
        consolidate_cmd = [
            py,
            str(consolidate),
            "--root",
            str(output_root),
        ]
        if not args.keep_old_external_data:
            consolidate_cmd.append("--remove-old")
        run_command(consolidate_cmd, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
