#!/usr/bin/env python3
"""打印 PyTorch 模型和 ONNX 输入输出 dtype 信息的小工具。

导出 FP16/FP32 子模型时，最容易出错的是 runtime 输入 dtype 和 ONNX 声明不一致。
这些打印函数帮助在导出阶段直接看到模型参数 dtype、ONNX IO dtype 和预期输入。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import onnx


def print_torch_dtype_summary(model, title="PyTorch model") -> None:
    """统计 PyTorch parameters/buffers 的 dtype 分布。"""
    param_dtypes = Counter(str(param.dtype).replace("torch.", "") for param in model.parameters())
    buffer_dtypes = Counter(str(buf.dtype).replace("torch.", "") for buf in model.buffers())
    print(f"\n[DType] {title}")
    print(f"  parameter dtypes: {dict(param_dtypes)}")
    print(f"  buffer dtypes:    {dict(buffer_dtypes)}")


def print_onnx_io_dtypes(onnx_path, title=None) -> None:
    """打印 ONNX graph input/output 的元素类型和形状。"""
    model = onnx.load(str(onnx_path), load_external_data=False)
    title = title or Path(onnx_path).name
    print(f"\n[DType] ONNX IO: {title}")
    for value in list(model.graph.input) + list(model.graph.output):
        tensor_type = value.type.tensor_type
        elem_type = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
        dims = []
        for dim in tensor_type.shape.dim:
            dims.append(dim.dim_param or dim.dim_value)
        print(f"  {value.name}: {elem_type} {dims}")


def print_expected_inputs(title, specs) -> None:
    """打印 runtime 期望喂给某个 ONNX 子图的输入 dtype/shape。"""
    print(f"\n[DType] Expected runtime inputs: {title}")
    for name, dtype, shape in specs:
        print(f"  {name}: {dtype} {shape}")
