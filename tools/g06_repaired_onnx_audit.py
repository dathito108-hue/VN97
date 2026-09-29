"""Staged real-weight G06 numerical audit; never a production promotion receipt."""
import argparse
import importlib
import json
from pathlib import Path
import time

CAPSULE = '8e1aff6c6b6d2947cb25111e48febae6850100fd782a2f112adcab2e047f674e'
SOURCE = '254d89bfef6dd9f44c8e3f54e0730af3f10c482969117cb37fb33486348b00be'


def difference(actual, expected):
    import numpy as np
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError('output shape/dtype differs from reference')
    left, right = actual.reshape(-1), expected.reshape(-1)
    maximum = 0.0
    for start in range(0, left.size, 1024 * 1024):
        a, b = left[start:start + 1024 * 1024].astype(np.float32), right[start:start + 1024 * 1024].astype(np.float32)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('non-finite numerical output')
        maximum = max(maximum, float(np.max(np.abs(a - b), initial=0.0)))
    return maximum


def compare(candidate, reference_dir, report_path, disable_optimizations=False):
    import numpy as np
    import onnxruntime as ort
    import g06_deployment_preflight as contract
    bundle = importlib.import_module('_g06_contracts.mamba2_bundle_contract')
    runtime_root = candidate / 'runtime'
    manifest = bundle.verify_g05_bundle(runtime_root)
    runtime = contract.load(runtime_root, 'runtime.vn97m2g06.json')
    assert runtime == contract.compile_runtime(manifest)
    assert manifest['parallel_algorithm'] == 'parallel_projection_conv_token_rounded_state_scan'
    assert runtime['capsule_id'] == CAPSULE and runtime['source_weight_sha256'] == SOURCE
    ref = json.loads((reference_dir / 'reference.json').read_text())
    assert ref['capsule_id'] == CAPSULE and ref['source_weight_sha256'] == SOURCE
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if disable_optimizations:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(runtime_root / runtime['graph_filename']), sess_options=options, providers=['CPUExecutionProvider'])
    results = []
    conv = ssm = None
    for case in ref['cases']:
        folder = reference_dir / str(case['index'])
        if case['reset']:
            conv = np.load(folder / 'conv_input.npy', allow_pickle=False)
            ssm = np.load(folder / 'ssm_input.npy', allow_pickle=False)
        started = time.perf_counter()
        logits, conv, ssm = session.run(['logits', 'next_conv_state', 'next_ssm_state'], {
            'input_ids': np.load(folder / 'input_ids.npy', allow_pickle=False),
            'valid_length': np.asarray([case['valid_length']], dtype=np.int64),
            'conv_state': conv, 'ssm_state': ssm,
        })
        metrics = {name: difference(actual, np.load(folder / (name + '.npy'), mmap_mode='r', allow_pickle=False))
                   for name, actual in (('logits', logits), ('conv_state', conv), ('ssm_state', ssm))}
        result = {**case, 'case_wall_seconds': time.perf_counter() - started, 'max_abs_error': metrics,
                  'passed': all(value <= 0.002 for value in metrics.values())}
        valid = case['valid_length']
        expected_logits = np.load(folder / 'logits.npy', mmap_mode='r', allow_pickle=False)
        result['argmax_ids'] = {'repeated': expected_logits[:, :valid].argmax(axis=-1).tolist(),
                                'onnx': logits[:, :valid].argmax(axis=-1).tolist()}
        parallel_report_path = reference_dir / 'parallel-reference.json'
        if parallel_report_path.is_file():
            parallel_report = json.loads(parallel_report_path.read_text())
            assert parallel_report['capsule_id'] == CAPSULE and parallel_report['source_weight_sha256'] == SOURCE
            result['onnx_vs_parallel_max_abs_error'] = {
                name: difference(actual, np.load(folder / ('parallel_' + name + '.npy'), mmap_mode='r', allow_pickle=False))
                for name, actual in (('logits', logits), ('conv_state', conv), ('ssm_state', ssm))}
            result['parallel_vs_repeated_max_abs_error'] = parallel_report['cases'][case['index']]['parallel_vs_repeated_max_abs_error']
            parallel_logits = np.load(folder / 'parallel_logits.npy', mmap_mode='r', allow_pickle=False)
            result['argmax_ids']['parallel'] = parallel_logits[:, :valid].argmax(axis=-1).tolist()
        results.append(result)
        print(json.dumps(results[-1]), flush=True)
    report = {'schema': 'VN97G06NUMAUDIT1', 'runtime_id': runtime['runtime_id'], 'capsule_id': CAPSULE,
              'source_weight_sha256': SOURCE, 'onnxruntime_version': ort.__version__,
              'graph_optimizations': 'disabled' if disable_optimizations else 'default',
              'reference_torch_version': ref['torch_version'], 'max_abs_error_limit': 0.002,
              'cases': results, 'passed': all(case['passed'] for case in results),
              'source_to_vn97_parity_tested': False, 'device_measured': False,
              'production_activation_authorized': False,
              'scope': 'CPU ONNX vs existing VN97 repeated-step reference; includes recurrent continuation; not source-model or Android qualification'}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    return report['passed']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audit newly exported G06 state-repair graph')
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--disable-optimizations', action='store_true')
    args = parser.parse_args()
    raise SystemExit(0 if compare(args.candidate, args.reference, args.report, args.disable_optimizations) else 1)
