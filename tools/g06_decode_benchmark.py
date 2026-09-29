"""Isolated-process, same-input decode comparison against the pinned INT8 graph."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from g06_quantize import sha


def worker(root, variant, result):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    session = ort.InferenceSession(str(root / ('candidate.onnx' if variant == 'baseline' else 'decode.onnx')),
                                   sess_options=options, providers=['CPUExecutionProvider'])
    conv = np.zeros((64, 1, 5376, 4), np.float16)
    ssm = np.zeros((64, 1, 80, 64, 128), np.float16)
    times = []
    for index, token in enumerate([100, 101, 102, 103]):
        ids = np.zeros((1, 8), np.int64)
        ids[0, 0] = token
        inputs = {'input_ids': ids, 'conv_state': conv, 'ssm_state': ssm}
        if variant == 'baseline':
            inputs['valid_length'] = np.array([1], np.int64)
        started = time.perf_counter()
        outputs = session.run(['logits', 'next_conv_state', 'next_ssm_state'], inputs)
        times.append(time.perf_counter() - started)
        for name, value in zip(['logits', 'conv', 'ssm'], outputs):
            if not np.isfinite(value).all():
                raise ValueError('nonfinite output')
            np.save(result / f'{variant}-{index}-{name}.npy', value)
        _, conv, ssm = outputs
    (result / f'{variant}.json').write_text(json.dumps({'seconds': times, 'ort': ort.__version__}))


def run(root, result):
    result.mkdir(parents=True, exist_ok=True)
    # Pin the real graph and external data before any execution.
    assert sha(root / 'candidate.onnx') == '403c934fcc683b55225664f2b6721e5b20fd5398b9cd356485f002bede5ee5b8'
    assert sha(root / 'candidate.onnx.data') == 'f15fc9c45b508fcd28d0d6be724c1ab1c5f8694fcc6395a9167c495448e27493'
    from g06_decode_specialize import run as specialize
    specialize(root / 'candidate.onnx', root / 'decode.onnx')
    for variant in ['baseline', 'decode']:
        subprocess.run([sys.executable, __file__, '--root', str(root), '--result', str(result),
                        '--worker', variant], check=True)
    cases = []
    for index in range(4):
        errors = {}
        for name in ['logits', 'conv', 'ssm']:
            a = np.load(result / f'baseline-{index}-{name}.npy', mmap_mode='r')
            b = np.load(result / f'decode-{index}-{name}.npy', mmap_mode='r')
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError('output contract mismatch')
            maximum = 0.0
            for offset in range(0, a.size, 1048576):
                diff = np.abs(a.reshape(-1)[offset:offset+1048576].astype(np.float32) -
                              b.reshape(-1)[offset:offset+1048576].astype(np.float32))
                maximum = max(maximum, float(diff.max(initial=0)))
            errors[name] = maximum
        cases.append(errors)
    baseline = json.loads((result / 'baseline.json').read_text())
    decode = json.loads((result / 'decode.json').read_text())
    report = {'schema': 'VN97_DECODE_BENCHMARK_1', 'cases_max_abs': cases,
              'exact_match': all(v == 0 for c in cases for v in c.values()),
              'baseline': baseline, 'decode': decode,
              'warm_speedup': sum(baseline['seconds'][1:]) / sum(decode['seconds'][1:]),
              'device_measured': False, 'quality_qualified': False,
              'production_activation_authorized': False}
    (result / 'comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not report['exact_match']:
        raise ValueError('specialized path differs from baseline')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--worker', choices=['baseline', 'decode'])
    a = p.parse_args()
    if a.worker:
        worker(a.root, a.worker, a.result)
    else:
        run(a.root, a.result)
