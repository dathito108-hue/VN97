"""Reproducible CPU pipeline probe, NOT a natural-language quality benchmark."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from language_candidate import NativeLanguageCandidate, save_candidate, load_candidate


def records():
    # Original synthetic fixtures, shared template. Only full records are disjoint.
    return [f"việc csv mã {i:03d}: bỏ dòng trùng.\n" for i in range(64)]


def fingerprint(items):
    return hashlib.sha256(json.dumps(items, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def batch(items):
    encoded = [list(text.encode('utf-8')) for text in items]
    length = max(map(len, encoded)) - 1
    x = torch.zeros(len(items), length, dtype=torch.long)
    y = torch.full_like(x, -100)
    for i, row in enumerate(encoded):
        x[i, :len(row)-1] = torch.tensor(row[:-1])
        y[i, :len(row)-1] = torch.tensor(row[1:])
    return x, y


@torch.no_grad()
def metrics(model, items):
    model.eval()
    x, y = batch(items)
    logits, _ = model(x)
    nll = torch.nn.functional.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)).item()
    mask = y != -100
    accuracy = ((logits.argmax(-1) == y) & mask).sum().item() / mask.sum().item()
    if not math.isfinite(nll):
        raise ValueError('non-finite evaluation')
    return {'byte_nll': nll, 'byte_accuracy': accuracy, 'target_bytes': mask.sum().item()}


def run(output):
    torch.manual_seed(97)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    all_records = records()
    splits = {'train': all_records[:48], 'validation': all_records[48:56], 'test': all_records[56:]}
    assert not (set(splits['train']) & set(splits['test']))
    assert not (set(splits['validation']) & (set(splits['train']) | set(splits['test'])))
    model = NativeLanguageCandidate()
    initial_weights = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    x, y = batch(splits['train'])
    model.train()
    # Fixed steps and hyperparameters; no validation/test-based checkpoint choice.
    for _ in range(80):
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    final = {name: metrics(model, rows) for name, rows in splits.items()}
    baseline = NativeLanguageCandidate(model.config)
    baseline.load_state_dict(initial_weights)
    initial = {name: metrics(baseline, rows) for name, rows in splits.items()}
    output.mkdir(parents=True, exist_ok=False)
    manifest = save_candidate(model, output / 'checkpoint')
    restored = load_candidate(output / 'checkpoint')
    torch.testing.assert_close(model(x)[0], restored(x)[0], rtol=0, atol=0)
    report = {'schema': 'VN97ASC-LMEVAL1', 'source_commit': os.environ.get('GITHUB_SHA', 'local-unrecorded'),
              'torch_version': str(torch.__version__), 'seed': 97, 'steps': 80, 'learning_rate': .01,
              'config': asdict(model.config), 'architecture': model.config.fingerprint(),
              'parameters': sum(p.numel() for p in model.parameters()),
              'recurrent_values_per_sample': 2 * model.config.d_inner * model.config.n_layers,
              'splits': {name: {'records': len(rows), 'sha256': fingerprint(rows)} for name, rows in splits.items()},
              'initial': initial, 'final': final, 'checkpoint': manifest,
              'production_activation_authorized': False,
              'limitations': ['synthetic shared-template byte prediction, not general language proficiency',
                             'full records disjoint; templates and many byte subsequences overlap',
                             'single seed; no equal-budget competing-architecture ablation',
                             'no ONNX export, device performance or revenue evidence']}
    (output / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    (output / 'dataset.json').write_text(json.dumps(splits, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'initial_test': initial['test'], 'final_test': final['test'],
                      'parameters': report['parameters'], 'production_activation_authorized': False}, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
