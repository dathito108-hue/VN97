"""Staged real-weight G06 numerical audit; never a production promotion receipt."""
import argparse
import importlib
import json
from pathlib import Path
import time

CAPSULE = '8e1aff6c6b6d2947cb25111e48febae6850100fd782a2f112adcab2e047f674e'
SOURCE = '254d89bfef6dd9f44c8e3f54e0730af3f10c482969117cb37fb33486348b00be'
RUNTIME = '4f292ea6cf37554ea1ddefa97aa642572989ade51ae6745cf4f075cdc04dda21'


def reference(capsule_root, out):
    import numpy as np
    import torch
    import g06_deployment_preflight  # installs dependency-free package namespace
    capsule_module = importlib.import_module('_g06_contracts.mamba2_g03_capsule')
    model_module = importlib.import_module('_g06_contracts.mamba2_onnx')
    parallel = importlib.import_module('_g06_contracts.mamba2_parallel_onnx')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    capsule = capsule_module.load_g03_capsule(capsule_root, verify_large_weight_sha256=True)
    assert capsule.manifest.capsule_id() == CAPSULE
    assert capsule.manifest.source_weight_sha256 == SOURCE
    model = model_module.VN97Mamba2StepOnnx(model_module.Mamba2OnnxConfig.from_source_spec(capsule.spec), capsule.tensors).eval()
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    conv = ssm = None
    with torch.inference_mode():
        for index, (length, reset) in enumerate(((1, True), (8, True), (1, False))):
            if reset:
                conv, ssm = model.initial_state(batch_size=1)
            tokens = torch.randint(0, 50277, (1, 8), generator=torch.Generator().manual_seed(9710 + index))
            folder = out / str(index)
            folder.mkdir()
            for name, value in (('input_ids', tokens), ('conv_input', conv), ('ssm_input', ssm)):
                np.save(folder / (name + '.npy'), value.detach().cpu().numpy(), allow_pickle=False)
            started = time.perf_counter()
            logits, conv, ssm = parallel.repeated_step_reference(model, tokens, valid_length=length, conv_state=conv, ssm_state=ssm)
            for name, value in (('logits', logits), ('conv_state', conv), ('ssm_state', ssm)):
                np.save(folder / (name + '.npy'), value.detach().cpu().numpy(), allow_pickle=False)
            cases.append({'index': index, 'valid_length': length, 'reset': reset, 'reference_seconds': time.perf_counter() - started})
            print(json.dumps(cases[-1]), flush=True)
    report = {'schema': 'VN97G06NUMREF1', 'capsule_id': CAPSULE, 'source_weight_sha256': SOURCE,
              'reference': 'existing VN97Mamba2StepOnnx repeated-step implementation',
              'torch_version': torch.__version__, 'cases': cases, 'production_activation_authorized': False}
    (out / 'reference.json').write_text(json.dumps(report, indent=2) + '\n')


def parallel_reference(capsule_root, reference_dir):
    import numpy as np
    import torch
    import g06_deployment_preflight
    capsule_module = importlib.import_module('_g06_contracts.mamba2_g03_capsule')
    model_module = importlib.import_module('_g06_contracts.mamba2_onnx')
    parallel_module = importlib.import_module('_g06_contracts.mamba2_parallel_onnx')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    ref = json.loads((reference_dir / 'reference.json').read_text())
    assert ref['capsule_id'] == CAPSULE and ref['source_weight_sha256'] == SOURCE
    capsule = capsule_module.load_g03_capsule(capsule_root, verify_large_weight_sha256=True)
    assert capsule.manifest.capsule_id() == CAPSULE and capsule.manifest.source_weight_sha256 == SOURCE
    step = model_module.VN97Mamba2StepOnnx(model_module.Mamba2OnnxConfig.from_source_spec(capsule.spec), capsule.tensors).eval()
    model = parallel_module.VN97Mamba2ParallelChunkOnnx(step, chunk_size=8).eval()
    conv = ssm = None
    results = []
    with torch.inference_mode():
        for case in ref['cases']:
            folder = reference_dir / str(case['index'])
            if case['reset']:
                conv = torch.from_numpy(np.load(folder / 'conv_input.npy', allow_pickle=False))
                ssm = torch.from_numpy(np.load(folder / 'ssm_input.npy', allow_pickle=False))
            tokens = torch.from_numpy(np.load(folder / 'input_ids.npy', allow_pickle=False))
            logits, conv, ssm = model(tokens, torch.tensor([case['valid_length']], dtype=torch.long), conv, ssm)
            metrics = {}
            for name, value in (('logits', logits), ('conv_state', conv), ('ssm_state', ssm)):
                actual = value.detach().cpu().numpy()
                np.save(folder / ('parallel_' + name + '.npy'), actual, allow_pickle=False)
                metrics[name] = difference(actual, np.load(folder / (name + '.npy'), mmap_mode='r', allow_pickle=False))
            result = {**case, 'parallel_vs_repeated_max_abs_error': metrics}
            results.append(result)
            print(json.dumps(result), flush=True)
    report = {'schema': 'VN97G06PARALLELREF1', 'capsule_id': CAPSULE, 'source_weight_sha256': SOURCE,
              'torch_version': torch.__version__, 'cases': results, 'production_activation_authorized': False}
    (reference_dir / 'parallel-reference.json').write_text(json.dumps(report, indent=2) + '\n')


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


def compare(candidate, reference_dir, report_path):
    import numpy as np
    import onnxruntime as ort
    import g06_deployment_preflight as contract
    bundle = importlib.import_module('_g06_contracts.mamba2_bundle_contract')
    runtime_root = candidate / 'runtime'
    manifest = bundle.verify_g05_bundle(runtime_root)
    runtime = contract.load(runtime_root, 'runtime.vn97m2g06.json')
    assert runtime == contract.compile_runtime(manifest)
    assert runtime['runtime_id'] == RUNTIME
    assert runtime['capsule_id'] == CAPSULE and runtime['source_weight_sha256'] == SOURCE
    ref = json.loads((reference_dir / 'reference.json').read_text())
    assert ref['capsule_id'] == CAPSULE and ref['source_weight_sha256'] == SOURCE
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
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
    report = {'schema': 'VN97G06NUMAUDIT1', 'runtime_id': RUNTIME, 'capsule_id': CAPSULE,
              'source_weight_sha256': SOURCE, 'onnxruntime_version': ort.__version__,
              'reference_torch_version': ref['torch_version'], 'max_abs_error_limit': 0.002,
              'cases': results, 'passed': all(case['passed'] for case in results),
              'source_to_vn97_parity_tested': False, 'device_measured': False,
              'production_activation_authorized': False,
              'scope': 'CPU ONNX vs existing VN97 repeated-step reference; includes recurrent continuation; not source-model or Android qualification'}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    return report['passed']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    ref = sub.add_parser('reference')
    ref.add_argument('--capsule', type=Path, required=True)
    ref.add_argument('--output', type=Path, required=True)
    parallel = sub.add_parser('parallel-reference')
    parallel.add_argument('--capsule', type=Path, required=True)
    parallel.add_argument('--reference', type=Path, required=True)
    audit = sub.add_parser('compare')
    audit.add_argument('--candidate', type=Path, required=True)
    audit.add_argument('--reference', type=Path, required=True)
    audit.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'reference':
        reference(args.capsule, args.output)
        return 0
    if args.mode == 'parallel-reference':
        parallel_reference(args.capsule, args.reference)
        return 0
    return 0 if compare(args.candidate, args.reference, args.report) else 1


if __name__ == '__main__':
    raise SystemExit(main())
