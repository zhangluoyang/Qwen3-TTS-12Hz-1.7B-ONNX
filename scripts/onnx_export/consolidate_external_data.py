#!/usr/bin/env python3
"""把每个 ONNX 模型的外部权重文件合并成一个 .data 文件。

大模型 ONNX 导出后经常会拆出很多 external data 文件。这个脚本把同一个
.onnx 旁边的外部权重合并为 `<model>.onnx.data`，目录更干净，也更方便发布。
"""

import argparse
from pathlib import Path

import onnx
from onnx import external_data_helper


def _external_locations(model):
    """收集 ONNX initializer 中引用的 external_data location。"""
    locations = set()
    for init in model.graph.initializer:
        if init.data_location == onnx.TensorProto.EXTERNAL:
            for entry in init.external_data:
                if entry.key == "location":
                    locations.add(entry.value)
    return locations


def consolidate_onnx(onnx_path: Path, remove_old: bool) -> tuple[int, int]:
    """合并单个 .onnx 的外部数据，返回 old/new 外部文件数量。"""
    model_dir = onnx_path.parent
    model = onnx.load(onnx_path, load_external_data=True)
    old_model = onnx.load(onnx_path, load_external_data=False)
    old_locations = _external_locations(old_model)

    if not old_locations:
        # 没有 external data 的小模型直接跳过。
        return 0, 0

    data_name = onnx_path.name + ".data"
    data_path = model_dir / data_name
    backup_data_path = model_dir / (data_name + ".bak_consolidate")
    tmp_onnx_path = model_dir / (onnx_path.name + ".tmp_consolidate")

    external_data_helper.convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=0,
        convert_attribute=False,
    )

    # 如果目标 .onnx.data 已经存在，onnx.save 可能会在旧文件后继续追加数据。
    # 先把旧文件挪走，再写新的 .data，避免已经合并过的模型越合并越大。
    if backup_data_path.exists():
        backup_data_path.unlink()
    if tmp_onnx_path.exists():
        tmp_onnx_path.unlink()
    if data_path.exists():
        data_path.replace(backup_data_path)

    try:
        onnx.save(model, tmp_onnx_path)
        tmp_onnx_path.replace(onnx_path)
        if backup_data_path.exists():
            backup_data_path.unlink()
    except Exception:
        if tmp_onnx_path.exists():
            tmp_onnx_path.unlink()
        if data_path.exists():
            data_path.unlink()
        if backup_data_path.exists():
            backup_data_path.replace(data_path)
        raise

    removed = 0
    if remove_old:
        for location in old_locations:
            if location == data_name:
                continue
            candidate = model_dir / location
            if candidate.exists() and candidate.is_file():
                candidate.unlink()
                removed += 1

    return len(old_locations), removed


def main():
    # root 可以是 onnx_isolated 或 onnx_isolated_fp16，脚本会递归查找 .onnx。
    parser = argparse.ArgumentParser(description="Consolidate ONNX external data files")
    parser.add_argument("--root", type=str, default="./onnx_isolated")
    parser.add_argument("--remove-old", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    for onnx_path in sorted(root.glob("*/*.onnx")):
        external_count, removed = consolidate_onnx(onnx_path, args.remove_old)
        if external_count:
            print(
                f"{onnx_path}: consolidated {external_count} external file(s), "
                f"removed {removed}"
            )
        else:
            print(f"{onnx_path}: no external data")


if __name__ == "__main__":
    main()
