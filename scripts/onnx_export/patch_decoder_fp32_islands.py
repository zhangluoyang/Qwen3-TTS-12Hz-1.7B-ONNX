#!/usr/bin/env python3
"""Patch tokenizer12hz decoder ONNX graphs with small FP32 compute islands.

The Qwen3-TTS tokenizer decoder can be sensitive to FP16/CUDA execution in a
few spots. This script inserts local Cast nodes around selected operations so
the rest of the model can stay FP16.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

import onnx
from onnx import TensorProto, helper


DECODER_REL_PATH = Path("tokenizer12hz") / "tokenizer12hz_decode.onnx"


class GraphPatcher:
    def __init__(self, model: onnx.ModelProto) -> None:
        self.model = model
        self.graph = model.graph
        self.nodes = list(self.graph.node)
        self.output_to_node = {out: node for node in self.nodes for out in node.output}
        self.used_names = {
            name
            for node in self.nodes
            for name in [node.name, *node.input, *node.output]
            if name
        }
        self.used_names.update(inp.name for inp in self.graph.input)
        self.used_names.update(out.name for out in self.graph.output)
        self.used_names.update(init.name for init in self.graph.initializer)

    def unique_name(self, base: str) -> str:
        clean = base.replace(":", "_")
        candidate = clean
        idx = 0
        while candidate in self.used_names:
            idx += 1
            candidate = f"{clean}_{idx}"
        self.used_names.add(candidate)
        return candidate

    def cast_node(self, src: str, dst_type: int, tag: str) -> tuple[onnx.NodeProto, str]:
        dst = self.unique_name(f"{src}_{tag}")
        node_name = self.unique_name(f"{src}_{tag}_Cast")
        node = helper.make_node("Cast", [src], [dst], name=node_name, to=dst_type)
        return node, dst

    def patch_node_to_float32(
        self,
        target: onnx.NodeProto,
        input_indices: Iterable[int],
        output_type: int,
        tag: str,
    ) -> tuple[list[onnx.NodeProto], list[onnx.NodeProto]]:
        before: list[onnx.NodeProto] = []
        after: list[onnx.NodeProto] = []

        for input_index in input_indices:
            cast, cast_output = self.cast_node(
                target.input[input_index], TensorProto.FLOAT, f"{tag}_to_fp32"
            )
            before.append(cast)
            target.input[input_index] = cast_output

        if len(target.output) != 1:
            raise ValueError(f"Expected one output for {target.name}, got {len(target.output)}")

        original_output = target.output[0]
        fp32_output = self.unique_name(f"{original_output}_{tag}_fp32")
        target.output[0] = fp32_output
        cast_back = helper.make_node(
            "Cast",
            [fp32_output],
            [original_output],
            name=self.unique_name(f"{original_output}_{tag}_back_Cast"),
            to=output_type,
        )
        after.append(cast_back)
        return before, after

    def replace_nodes(self, replacements: dict[str, tuple[list[onnx.NodeProto], list[onnx.NodeProto]]]) -> None:
        new_nodes: list[onnx.NodeProto] = []
        for node in self.nodes:
            before, after = replacements.get(node.name, ([], []))
            new_nodes.extend(before)
            new_nodes.append(node)
            new_nodes.extend(after)
        del self.graph.node[:]
        self.graph.node.extend(new_nodes)
        self.nodes = list(self.graph.node)
        self.output_to_node = {out: node for node in self.nodes for out in node.output}

    def find_swiglu_final_muls(self) -> list[onnx.NodeProto]:
        result: list[onnx.NodeProto] = []
        for node in self.nodes:
            if node.op_type != "Mul":
                continue
            name = node.name or ""
            if "/mlp/Mul" not in name or "/act_fn/" in name:
                continue
            if not any("/mlp/act_fn/Mul" in inp for inp in node.input):
                continue
            if not any("/mlp/up_proj/MatMul" in inp for inp in node.input):
                continue
            result.append(node)
        return result

    def find_down_proj_matmul(self, swiglu_mul: onnx.NodeProto) -> onnx.NodeProto | None:
        swiglu_output = swiglu_mul.output[0]
        consumers = [
            node
            for node in self.nodes
            if node.op_type == "MatMul" and swiglu_output in list(node.input)
        ]
        if len(consumers) != 1:
            return None
        consumer = consumers[0]
        if "/mlp/down_proj/MatMul" not in (consumer.name or ""):
            return None
        return consumer

    def patch_reduce_sum_lengths(self) -> int:
        replacements: dict[str, tuple[list[onnx.NodeProto], list[onnx.NodeProto]]] = {}
        patched = 0
        for node in self.nodes:
            if node.op_type != "ReduceSum":
                continue
            if len(node.input) < 1:
                continue
            if (node.name or "") != "/ReduceSum" and "length" not in " ".join(node.output).lower():
                continue
            before, after = self.patch_node_to_float32(
                node, input_indices=[0], output_type=TensorProto.INT64, tag="fp32_reduce"
            )
            replacements[node.name] = (before, after)
            patched += 1
        self.replace_nodes(replacements)
        return patched

    def patch_swiglu_mul(self) -> int:
        replacements: dict[str, tuple[list[onnx.NodeProto], list[onnx.NodeProto]]] = {}
        for node in self.find_swiglu_final_muls():
            before, after = self.patch_node_to_float32(
                node, input_indices=[0, 1], output_type=TensorProto.FLOAT16, tag="fp32_swiglu"
            )
            replacements[node.name] = (before, after)
        self.replace_nodes(replacements)
        return len(replacements)

    def patch_swiglu_down_proj(self) -> int:
        replacements: dict[str, tuple[list[onnx.NodeProto], list[onnx.NodeProto]]] = {}
        patched = 0

        for swiglu in self.find_swiglu_final_muls():
            down_proj = self.find_down_proj_matmul(swiglu)
            if down_proj is None:
                continue

            before: list[onnx.NodeProto] = []
            swiglu_after: list[onnx.NodeProto] = []
            for input_index in [0, 1]:
                cast, cast_output = self.cast_node(
                    swiglu.input[input_index], TensorProto.FLOAT, "fp32_swiglu_to_fp32"
                )
                before.append(cast)
                swiglu.input[input_index] = cast_output

            original_swiglu_output = swiglu.output[0]
            fp32_swiglu_output = self.unique_name(f"{original_swiglu_output}_fp32_swiglu")
            swiglu.output[0] = fp32_swiglu_output

            for input_index, input_name in enumerate(down_proj.input):
                if input_name == original_swiglu_output:
                    down_proj.input[input_index] = fp32_swiglu_output
                else:
                    cast, cast_output = self.cast_node(
                        input_name, TensorProto.FLOAT, "fp32_down_proj_to_fp32"
                    )
                    replacements.setdefault(down_proj.name, ([], []))[0].append(cast)
                    down_proj.input[input_index] = cast_output

            original_down_output = down_proj.output[0]
            fp32_down_output = self.unique_name(f"{original_down_output}_fp32_down_proj")
            down_proj.output[0] = fp32_down_output
            cast_back = helper.make_node(
                "Cast",
                [fp32_down_output],
                [original_down_output],
                name=self.unique_name(f"{original_down_output}_fp32_down_proj_back_Cast"),
                to=TensorProto.FLOAT16,
            )
            replacements.setdefault(down_proj.name, ([], []))[1].append(cast_back)
            replacements[swiglu.name] = (before, swiglu_after)
            patched += 1

        self.replace_nodes(replacements)
        return patched


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.onnx_root:
        input_root = Path(args.onnx_root)
        output_root = Path(args.output_root) if args.output_root else input_root.with_name(input_root.name + "_fp32_islands")
        return input_root / DECODER_REL_PATH, output_root / DECODER_REL_PATH

    if not args.input:
        raise SystemExit("Either --onnx-root or --input is required")
    input_model = Path(args.input)
    output_model = Path(args.output) if args.output else input_model.with_name(input_model.stem + "_fp32_islands.onnx")
    return input_model, output_model


def prepare_output_tree(args: argparse.Namespace, input_model: Path, output_model: Path) -> None:
    if args.onnx_root:
        input_root = Path(args.onnx_root)
        output_root = output_model.parents[1]
        if output_root.exists() and not args.overwrite:
            raise SystemExit(f"Output root exists: {output_root}. Use --overwrite to replace it.")
        if output_root.exists():
            shutil.rmtree(output_root)
        shutil.copytree(input_root, output_root)
    else:
        output_model.parent.mkdir(parents=True, exist_ok=True)
        if output_model.exists() and not args.overwrite:
            raise SystemExit(f"Output file exists: {output_model}. Use --overwrite to replace it.")


def patch_model(args: argparse.Namespace) -> None:
    input_model, output_model = resolve_paths(args)
    if not input_model.exists():
        raise SystemExit(f"Input model not found: {input_model}")

    prepare_output_tree(args, input_model, output_model)
    model = onnx.load(str(input_model), load_external_data=True)
    patcher = GraphPatcher(model)

    reduce_count = 0
    swiglu_count = 0
    if args.patch_reduce_sum:
        reduce_count = patcher.patch_reduce_sum_lengths()
    if args.patch_swiglu:
        if args.swiglu_mode == "mul":
            swiglu_count = patcher.patch_swiglu_mul()
        elif args.swiglu_mode == "down-proj":
            swiglu_count = patcher.patch_swiglu_down_proj()
        else:
            raise ValueError(args.swiglu_mode)

    onnx.checker.check_model(model)
    onnx.save_model(
        model,
        str(output_model),
        save_as_external_data=False,
    )

    print(f"wrote: {output_model}")
    print(f"patched ReduceSum nodes: {reduce_count}")
    print(f"patched SwiGLU blocks: {swiglu_count} ({args.swiglu_mode})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--onnx-root", help="ONNX root containing tokenizer12hz/tokenizer12hz_decode.onnx")
    source.add_argument("--input", help="Input tokenizer12hz_decode.onnx")
    parser.add_argument("--output-root", help="Output ONNX root. Defaults to <onnx-root>_fp32_islands")
    parser.add_argument("--output", help="Output ONNX model. Defaults to *_fp32_islands.onnx")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output path")
    parser.add_argument("--patch-reduce-sum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patch-swiglu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--swiglu-mode",
        choices=["mul", "down-proj"],
        default="down-proj",
        help="Patch only the SwiGLU Mul, or keep the SwiGLU product in FP32 through down_proj MatMul",
    )
    return parser


def main() -> None:
    patch_model(build_parser().parse_args())


if __name__ == "__main__":
    main()
