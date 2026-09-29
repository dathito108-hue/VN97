"""Real quantized G06 execution diagnostics; never a quality/promotion receipt."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from g06_quantize import sha
import g06_deployment_preflight as contract

IDS = {
    'int8': '331dcb61ad1547667fc24cb000fabe202d8380937b57859e0aa209814e4ab8fd',
    'mixed-int4': 'a0acd7ecf1355e1279ecda110e01bb593b6a34a78c5d3da5f69e763853e0ac93',
}


def write(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def error_metrics(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError('shape/dtype mismatch')
    maximum = total = 0.0
    count = actual.size
    for begin in range(0, count, 1048576):
        a = actual.reshape(-1)[begin:begin+1048576].astype(np.float32)
        b = expected.reshape(-1)[begin:begin+1048576].astype(np.float32)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('nonfinite output/reference')
        diff = np.abs(a - b)
        maximum = max(maximum, float(diff.max(initial=0)))
        total += float(diff.sum(dtype=np.float64))
    return {'max_abs': maximum, 'mean_abs': total / max(count, 1)}


def evaluate(candidate, reference, mode, output):
    import onnxruntime as ort
    report = {'schema': 'VN97G06QUANTEVAL1', 'mode': mode, 'cases': [],
              'complete': False, 'execution_passed': False, 'quality_qualified': False,
              'device_measured': False, 'production_activation_authorized': False,
              'scope': '3 hosted CPU probes against frozen FP16 PyTorch outputs; not a holdout quality evaluation'}
    write(output, report)
    try:
        meta = contract.load(candidate, 'quantization.json')
        identity = meta.pop('candidate_id')
        digest = hashlib.sha256(json.dumps(meta, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if identity != digest or identity != IDS[mode] or meta['mode'] != mode:
            raise ValueError('candidate identity mismatch')
        if {f['path'] for f in meta['files']} != {'candidate.onnx', 'candidate.onnx.data'} or len(meta['files']) != 2:
            raise ValueError('unexpected candidate inventory')
        for record in meta['files']:
            path = contract.regular(candidate, record['path'])
            if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
                raise ValueError('candidate payload mismatch')
        ref = contract.load(reference, 'reference.json')
        if ref['source_weight_sha256'] != meta['source_weight_sha256']:
            raise ValueError('reference lineage mismatch')
        if [(c['index'], c['valid_length'], c['reset']) for c in ref['cases']] != [(0, 1, True), (1, 8, True), (2, 1, False)]:
            raise ValueError('unexpected reference cases')
        report.update(candidate_id=identity, onnxruntime_version=ort.__version__, cpu_threads=2)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        started = time.perf_counter()
        session = ort.InferenceSession(str(candidate / 'candidate.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
        report['session_seconds'] = time.perf_counter() - started
        write(output, report)
        for case in ref['cases']:
            folder = reference / str(case['index'])
            if case['reset']:
                conv = np.load(folder / 'conv_input.npy', allow_pickle=False)
                ssm = np.load(folder / 'ssm_input.npy', allow_pickle=False)
            tokens = np.load(folder / 'input_ids.npy', allow_pickle=False)
            started = time.perf_counter()
            logits, conv, ssm = session.run(['logits', 'next_conv_state', 'next_ssm_state'], {
                'input_ids': tokens, 'valid_length': np.array([case['valid_length']], dtype=np.int64),
                'conv_state': conv, 'ssm_state': ssm,
            })
            seconds = time.perf_counter() - started
            errors = {name: error_metrics(actual, np.load(folder / (name + '.npy'), mmap_mode='r', allow_pickle=False))
                      for name, actual in (('logits', logits), ('conv_state', conv), ('ssm_state', ssm))}
            valid = case['valid_length']
            if np.count_nonzero(logits[:, valid:]):
                raise ValueError('nonzero padded logits')
            expected = np.load(folder / 'logits.npy', mmap_mode='r', allow_pickle=False)
            # Ignore padded vocabulary IDs, as the Android tokenizer does.
            actual_ids = logits[:, :valid, :50277].argmax(axis=-1)
            expected_ids = expected[:, :valid, :50277].argmax(axis=-1)
            result = {'index': case['index'], 'valid_length': valid, 'reset': case['reset'],
                      'inference_seconds': seconds, 'errors_vs_fp16_pytorch': errors,
                      'top1_matches': int((actual_ids == expected_ids).sum()), 'decisions': valid,
                      'actual_ids': actual_ids.tolist(), 'reference_ids': expected_ids.tolist(),
                      'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
            report['cases'].append(result)
            write(output, report)
            print(json.dumps(result), flush=True)
        report.update(complete=True, execution_passed=True)
    except Exception as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        write(output, report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--mode', choices=IDS, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    evaluate(args.candidate, args.reference, args.mode, args.output)
