"""Staged real-weight G06 numerical audit; never a production promotion receipt."""
import argparse
import importlib
import json
from pathlib import Path
import time

CAPSULE = '8e1aff6c6b6d2947cb25111e48febae6850100fd782a2f112adcab2e047f674e'
SOURCE = '254d89bfef6dd9f44c8e3f54e0730af3f10c482969117cb37fb33486348b00be'
RUNTIME = '4f292ea6cf37554ea1ddefa97aa642572989ade51ae6745cf4f075cdc04dda21'


def parallel_reference(capsule_root, reference_dir, report_path):
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
            started = time.perf_counter()
            logits, conv, ssm = model(tokens, torch.tensor([case['valid_length']], dtype=torch.long), conv, ssm)
            forward_seconds = time.perf_counter() - started
            metrics = {}
            for name, value in (('logits', logits), ('conv_state', conv), ('ssm_state', ssm)):
                actual = value.detach().cpu().numpy()
                metrics[name] = difference(actual, np.load(folder / (name + '.npy'), mmap_mode='r', allow_pickle=False))
            result = {**case, 'forward_seconds': forward_seconds, 'parallel_vs_repeated_max_abs_error': metrics, 'passed': all(v <= 0.002 for v in metrics.values())}
            results.append(result)
            print(json.dumps(result), flush=True)
    report = {'schema': 'VN97G06STATEREPAIR1', 'capsule_id': CAPSULE, 'source_weight_sha256': SOURCE,
              'torch_version': torch.__version__, 'cases': results, 'max_abs_error_limit': 0.002, 'passed': all(c['passed'] for c in results), 'device_measured': False, 'onnx_measured': False, 'production_activation_authorized': False}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    return report['passed']


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Full-weight state repair audit, no activation')
    parser.add_argument('--capsule', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if parallel_reference(args.capsule, args.reference, args.report) else 1)
