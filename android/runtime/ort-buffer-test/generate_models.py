"""Tiny real ORT I/O fixtures; no model weights or external services."""
from pathlib import Path
import onnx
from onnx import TensorProto, helper

for name, dtype in (("float16", TensorProto.FLOAT16), ("float32", TensorProto.FLOAT)):
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "pinned_buffer_identity",
        [helper.make_tensor_value_info("input", dtype, [1, 2, 3])],
        [helper.make_tensor_value_info("output", dtype, [1, 2, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, Path(__file__).parent / f"{name}.onnx")
