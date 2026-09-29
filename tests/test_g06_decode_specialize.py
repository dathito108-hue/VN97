import sys
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T, numpy_helper as n
import onnxruntime as ort
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from g06_decode_specialize import specialize


def test_dead_transition_pruned_without_changing_state():
    nodes = [h.make_node('Gather', ['valid_length', 'index'], ['length'], axis=0),
             h.make_node('Less', ['one', 'length'], ['active']),
             h.make_node('Mul', ['state', 'two'], ['unused_transition']),
             h.make_node('Where', ['active', 'unused_transition', 'state'], ['output'])]
    model = h.make_model(h.make_graph(nodes, 'scan',
        [h.make_tensor_value_info('valid_length', T.INT64, [1]),
         h.make_tensor_value_info('state', T.FLOAT, [3])],
        [h.make_tensor_value_info('output', T.FLOAT, [3])],
        [n.from_array(np.array(0, np.int64), 'index'),
         n.from_array(np.array(1, np.int64), 'one'),
         n.from_array(np.array(2, np.float32), 'two')]),
        opset_imports=[h.make_opsetid('', 18)], ir_version=10)
    fixed, report = specialize(model)
    onnx.checker.check_model(fixed)
    assert report['where_bypassed'] == 1
    assert [v.name for v in fixed.graph.input] == ['state']
    assert all(v.op_type != 'Mul' for v in fixed.graph.node)
    session = ort.InferenceSession(fixed.SerializeToString(), providers=['CPUExecutionProvider'])
    state = np.array([1, -2, 3], np.float32)
    np.testing.assert_array_equal(session.run(None, {'state': state})[0], state)
    assert len(model.graph.node) == 4  # no mutation of the source
