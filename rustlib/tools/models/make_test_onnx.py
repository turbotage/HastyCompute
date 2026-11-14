"""Tiny ONNX test model for hasty-infer: 1x1x1 Conv3d, 1 -> 2 channels, logit1 - logit0 = 20 (x - 0.5).
Dynamic batch / spatial dims. usage: python make_test_onnx.py out.onnx"""
import sys

import numpy as np
import onnx
from onnx import TensorProto as TP, helper, numpy_helper

w = np.array([0.0, 20.0], np.float32).reshape(2, 1, 1, 1, 1)
b = np.array([0.0, -10.0], np.float32)
node = helper.make_node("Conv", ["x", "w", "b"], ["y"], kernel_shape=[1, 1, 1])
g = helper.make_graph(
    [node], "pointwise",
    [helper.make_tensor_value_info("x", TP.FLOAT, ["n", 1, "z", "y", "x"])],
    [helper.make_tensor_value_info("y", TP.FLOAT, ["n", 2, "z", "y", "x"])],
    [numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b")],
)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
onnx.checker.check_model(m)
onnx.save(m, sys.argv[1])
