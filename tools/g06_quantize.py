"""Weight-only G06 experiment. Output is never an activatable G06 package."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def quantize_graph(model, cfg, mode):
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    if mode not in ('int8', 'mixed-int4'):
        raise ValueError('unsupported mode')
    d = cfg['d_model']
    inner = d * cfg['expand']
    proj = 2 * inner + 2 * cfg['d_state'] * cfg['n_groups'] + inner // cfg['head_dim']
    shapes = {(d, proj): (8 if mode == 'int8' else 4), (inner, d): 8}
    init = {t.name: t for t in model.graph.initializer}
    selected = {}
    for i, node in enumerate(model.graph.node):
        if not node.name:
            node.name = f'vn97_node_{i}'
        if node.op_type == 'MatMul' and len(node.input) == 2 and node.input[1] in init:
            shape = tuple(init[node.input[1]].dims)
            if shape in shapes:
                selected[node.name] = shapes[shape]
    if len(selected) != 2 * cfg['n_layers']:
        raise ValueError(f'expected exactly two projection matrices per layer, found {len(selected)}')
    if len({n.name for n in model.graph.node}) != len(model.graph.node):
        raise ValueError('duplicate node names')
    signature = [v.SerializeToString() for v in (*model.graph.input, *model.graph.output)]
    protected = {n.name: n.SerializeToString() for n in model.graph.node if n.name not in selected}
    protected_weights = {name: init[name].SerializeToString()
                         for n in model.graph.node if n.name not in selected
                         for name in n.input if name in init}
    # Never quantize an initializer shared with a protected operation.
    for n in model.graph.node:
        if n.name in selected and n.input[1] in protected_weights:
            raise ValueError('projection weight is shared with a protected operation')
    for bits in sorted(set(selected.values()), reverse=True):
        include = [name for name, value in selected.items() if value == bits]
        exclude = [n.name for n in model.graph.node if n.name not in include]
        quant = MatMulNBitsQuantizer(model, bits=bits, block_size=32,
                                    is_symmetric=True, accuracy_level=1,
                                    nodes_to_include=include, nodes_to_exclude=exclude,
                                    op_types_to_quantize=('MatMul',))
        quant.process()
        model = quant.model.model
    if signature != [v.SerializeToString() for v in (*model.graph.input, *model.graph.output)]:
        raise ValueError('graph state/I/O contract changed')
    actual_nodes = {n.name: n.SerializeToString() for n in model.graph.node}
    for name, value in protected.items():
        if actual_nodes.get(name) != value:
            raise ValueError(f'protected operator changed: {name}')
    actual_init = {t.name: t for t in model.graph.initializer}
    for name, value in protected_weights.items():
        if name not in actual_init or actual_init[name].SerializeToString() != value:
            raise ValueError(f'protected weight changed: {name}')
    packed = [n for n in model.graph.node if n.op_type == 'MatMulNBits']
    if len(packed) != len(selected):
        raise ValueError('quantizer did not convert every selected projection')
    from onnx import helper
    counts = {}
    for node in packed:
        bits = next(helper.get_attribute_value(a) for a in node.attribute if a.name == 'bits')
        counts[str(bits)] = counts.get(str(bits), 0) + 1
    return model, counts


def run(source, output, mode):
    import onnx
    import onnxruntime as ort
    import g06_deployment_preflight as preflight
    contract = importlib.import_module('_g06_contracts.mamba2_bundle_contract')
    manifest = contract.verify_g05_bundle(source)
    runtime = preflight.load(source, 'runtime.vn97m2g06.json')
    if runtime != preflight.compile_runtime(manifest):
        raise ValueError('source runtime descriptor mismatch')
    if output.exists() or output.resolve().is_relative_to(source.resolve()):
        raise ValueError('output must be a new directory outside source')
    graph_path = source / runtime['graph_filename']
    model = onnx.load(str(graph_path))
    model, counts = quantize_graph(model, manifest['config'], mode)
    output.mkdir(parents=True)
    target = output / 'candidate.onnx'
    onnx.save_model(model, str(target), save_as_external_data=True,
                    all_tensors_to_one_file=True, location='candidate.onnx.data', size_threshold=1024)
    del model
    onnx.checker.check_model(str(target))
    files = [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)}
             for p in sorted(output.iterdir()) if p.is_file()]
    report = {'schema': 'VN97G06QUANTCANDIDATE1', 'mode': mode,
              'source_runtime_id': runtime['runtime_id'],
              'source_weight_sha256': runtime['source_weight_sha256'],
              'source_g05_manifest_id': manifest['manifest_id'],
              'onnxruntime_version': ort.__version__, 'block_size': 32,
              'quantized_nodes_by_bits': counts, 'state_dtype': runtime['state_dtype'],
              'protected_operators_and_weights_unchanged': True,
              'files': files, 'total_bytes': sum(p['bytes'] for p in files),
              'quality_qualified': False, 'device_measured': False,
              'production_activation_authorized': False}
    report['candidate_id'] = hashlib.sha256(json.dumps(report, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    (output / 'quantization.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('int8', 'mixed-int4'), required=True)
    args = parser.parse_args()
    run(args.source, args.output, args.mode)
