"""Measure prepacking tradeoff on pinned real INT8 weights in separate processes."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np


def worker(root, out, mode):
    import onnxruntime as ort
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = 2
    opt.inter_op_num_threads = 1
    opt.enable_cpu_mem_arena = False
    opt.enable_mem_pattern = False
    if mode == 'unpacked':
        opt.add_session_config_entry('session.disable_prepacking', '1')
    start = time.perf_counter()
    session = ort.InferenceSession(str(root / 'decode.onnx'), sess_options=opt,
                                   providers=['CPUExecutionProvider'])
    report = {'load_seconds': time.perf_counter() - start, 'mode': mode, 'steps': []}
    conv = np.zeros((64, 1, 5376, 4), np.float16)
    ssm = np.zeros((64, 1, 80, 64, 128), np.float16)
    for i, token in enumerate([100, 101, 102, 103]):
        ids = np.zeros((1, 8), np.int64); ids[0, 0] = token
        start = time.perf_counter()
        logits, conv, ssm = session.run(None, {'input_ids': ids, 'conv_state': conv, 'ssm_state': ssm})
        elapsed = time.perf_counter() - start
        for name, value in [('logits', logits), ('conv', conv), ('ssm', ssm)]:
            if not np.isfinite(value).all():
                raise ValueError('nonfinite output')
            np.save(out / f'{mode}-{i}-{name}.npy', value)
        report['steps'].append({'seconds': elapsed,
                               'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    (out / (mode + '.json')).write_text(json.dumps(report, indent=2))


def run(root, out):
    from g06_quantize import sha
    out.mkdir(parents=True, exist_ok=True)
    if sha(root / 'candidate.onnx.data') != 'f15fc9c45b508fcd28d0d6be724c1ab1c5f8694fcc6395a9167c495448e27493':
        raise ValueError('weight mismatch')
    graph = gzip.decompress(Path('android/app/src/main/assets/vn97-int8-decode.bin').read_bytes())
    if hashlib.sha256(graph).hexdigest() != '873c862d50cfe7c10b47b5c02e11fc8dea42ad381431570e657a0eba99bdd88e':
        raise ValueError('graph mismatch')
    (root / 'decode.onnx').write_bytes(graph)
    for mode in ['packed', 'unpacked']:
        subprocess.run([sys.executable, __file__, '--root', str(root), '--out', str(out), '--worker', mode], check=True)
    errors = []
    for i in range(4):
        case = {}
        for name in ['logits', 'conv', 'ssm']:
            a = np.load(out / f'packed-{i}-{name}.npy', mmap_mode='r')
            b = np.load(out / f'unpacked-{i}-{name}.npy', mmap_mode='r')
            assert a.shape == b.shape and a.dtype == b.dtype
            maximum = 0.0
            for start in range(0, a.size, 1048576):
                x = a.reshape(-1)[start:start+1048576].astype(np.float32)
                y = b.reshape(-1)[start:start+1048576].astype(np.float32)
                maximum = max(maximum, float(np.abs(x-y).max(initial=0)))
            case[name] = maximum
        errors.append(case)
    report = {mode: json.loads((out / (mode + '.json')).read_text()) for mode in ['packed', 'unpacked']}
    report.update(errors=errors, exact_match=all(v == 0 for c in errors for v in c.values()),
                  device_measured=False, production_activation_authorized=False)
    (out / 'comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--worker', choices=['packed', 'unpacked'])
    a = p.parse_args()
    if a.worker: worker(a.root, a.out, a.worker)
    else: run(a.root, a.out)
