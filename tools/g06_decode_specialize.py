"""Prune inactive SSM transitions for valid_length=1; reuse external INT8 bytes.

Experimental decoder, NOT a production package. The input token tensor still
has width eight; only position zero is consumed. Prefill must use the original.
"""
import argparse
import copy
from pathlib import Path
import numpy as np


def specialize(model):
    import onnx
    from onnx import helper, numpy_helper
    from onnx.reference import ReferenceEvaluator
    model = copy.deepcopy(model)
    graph = model.graph
    if [v.name for v in graph.input].count('valid_length') != 1:
        raise ValueError('expected one valid_length input')
    if any(t.name == 'valid_length' for t in graph.initializer):
        raise ValueError('valid_length already has a default')
    graph.input.remove(next(v for v in graph.input if v.name == 'valid_length'))
    graph.initializer.append(numpy_helper.from_array(np.array([1], np.int64), 'valid_length'))
    # Only small inline metadata is evaluated, never model weights or activations.
    known = {t.name: numpy_helper.to_array(t) for t in graph.initializer
             if t.data_location != onnx.TensorProto.EXTERNAL and np.prod(t.dims) <= 4096}
    safe = {'Gather', 'Less', 'LessOrEqual', 'Greater', 'GreaterOrEqual', 'Equal',
            'Not', 'And', 'Or', 'Cast', 'Reshape', 'Unsqueeze', 'Squeeze',
            'Concat', 'Add', 'Sub', 'Mul', 'Div', 'Range', 'Identity'}
    nodes = []
    bypassed = 0
    for node in graph.node:
        if node.op_type == 'Where' and node.input[0] in known:
            condition = known[node.input[0]]
            # Identity is safe only for a scalar condition: a non-scalar
            # condition can broadcast/expand the selected branch's shape.
            if condition.size == 1 and condition.ndim == 0:
                source = node.input[1 if bool(condition.item()) else 2]
                nodes.append(helper.make_node('Identity', [source], list(node.output), name=node.name))
                if source in known:
                    known[node.output[0]] = known[source]
                bypassed += 1
                continue
        if not node.domain and (node.op_type == 'Constant' or
                (node.op_type in safe and all(i in known for i in node.input if i))):
            try:
                values = ReferenceEvaluator(node).run(None, {i: known[i] for i in node.input if i})
                if all(v.size <= 4096 for v in values):
                    for name, value in zip(node.output, values):
                        known[name] = value
                        graph.initializer.append(numpy_helper.from_array(value, name))
                    continue
            except (ValueError, TypeError, NotImplementedError):
                pass
        nodes.append(node)
    needed = {v.name for v in graph.output}
    live = []
    for node in reversed(nodes):
        if needed.intersection(node.output):
            live.append(node)
            needed.update(i for i in node.input if i)
    del graph.node[:]
    graph.node.extend(reversed(live))
    kept = [t for t in graph.initializer if t.name in needed]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    del graph.value_info[:]
    return model, {'where_bypassed': bypassed, 'nodes_before': len(nodes),
                   'nodes_after': len(live)}


def run(source, output):
    import json
    import onnx
    from g06_quantize import sha
    if output.exists() or source.resolve().parent != output.resolve().parent:
        raise ValueError('output must be new and beside the original external weights')
    raw = onnx.load(str(source), load_external_data=False)
    model, report = specialize(raw)
    report['nodes_before'] = len(raw.graph.node)
    if report['where_bypassed'] == 0 or report['nodes_after'] >= report['nodes_before']:
        raise ValueError('no inactive scan pruning occurred')
    # External references are preserved byte-for-byte, no re-quantization.
    original = {t.name: t.SerializeToString() for t in raw.graph.initializer}
    for t in model.graph.initializer:
        if t.name in original and original[t.name] != t.SerializeToString():
            raise ValueError('weight changed')
    onnx.save_model(model, str(output))
    onnx.checker.check_model(str(output))
    report.update(schema='VN97_DECODE_SPECIALIZATION_1', fixed_valid_length=1,
                  input_width=8, source_graph_sha256=sha(source), graph_sha256=sha(output),
                  external_weights_unchanged=True, production_activation_authorized=False,
                  device_measured=False)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.source, a.output)
