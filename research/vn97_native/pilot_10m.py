"""Bounded 10M CPU pilot. Original synthetic tasks, no AGI/proficiency claim."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import zipfile

import torch
from language_candidate import CandidateConfig, NativeLanguageCandidate, save_candidate, load_candidate


def dataset():
    splits = {}
    for split, start, count in [('train', 100, 128), ('validation', 500, 16), ('test', 900, 16)]:
        rows = []
        for i in range(count):
            code = str(start + i)
            a, b = (i % 40 + 1, i % 11 + 1) if split == 'train' else (41 + i, 12 + i)
            letter = chr(97 + i % 26)
            category = ['csv', 'tổng', 'mã', 'danh mục'][i % 4]
            label = ['clean_csv', 'sum', 'copy_id', 'catalog'][i % 4]
            if split == 'train':
                prompts = [f'Trả lại mã {code}.', f'Cộng {a} và {b}.',
                           f'Bỏ trùng: {letter},{code},{letter}.', f'Loại yêu cầu: {category}.']
            elif split == 'validation':
                prompts = [f'Mã cần chép là {code}.', f'Tính {a}+{b}.',
                           f'Giữ mục duy nhất: {letter},{code},{letter}.', f'Chọn nhãn cho: {category}.']
            else:
                prompts = [f'Chép mã: {code}.', f'Cho tổng của {a} với {b}.',
                           f'Xóa mục lặp: {letter},{code},{letter}.', f'Phân loại: {category}.']
            for task, prompt, answer in zip(('copy', 'sum', 'deduplicate', 'classify'), prompts,
                                            (code, str(a+b), f'{letter},{code}', label)):
                rows.append({'task': task, 'prompt': prompt + '\nĐáp: ', 'answer': answer + '\n'})
        splits[split] = rows
    # Grouped templates differ; verify no complete prompt crosses a split.
    sets = [set(r['prompt'] for r in splits[k]) for k in splits]
    assert all(not (sets[i] & sets[j]) for i in range(3) for j in range(i))
    return splits


def batch(rows):
    values = [(list(r['prompt'].encode()), list(r['answer'].encode())) for r in rows]
    length = max(len(p) + len(a) - 1 for p, a in values)
    if length > 128:
        raise ValueError('pilot sequence budget exceeded')
    x = torch.zeros(len(rows), length, dtype=torch.long)
    y = torch.full_like(x, -100)
    for i, (prompt, answer) in enumerate(values):
        combined = prompt + answer
        x[i, :len(combined)-1] = torch.tensor(combined[:-1])
        y[i, len(prompt)-1:len(combined)-1] = torch.tensor(answer)
    return x, y


@torch.no_grad()
def evaluate(model, rows):
    model.eval()
    totals = defaultdict(lambda: [0., 0, 0])
    for offset in range(0, len(rows), 4):
        chunk = rows[offset:offset+4]
        x, y = batch(chunk)
        logits, _ = model(x)
        losses = torch.nn.functional.cross_entropy(logits.transpose(1, 2), y, reduction='none')
        predictions = logits.argmax(-1)
        for i, row in enumerate(chunk):
            mask = y[i] != -100
            for key in (row['task'], 'all'):
                totals[key][0] += losses[i][mask].sum().item()
                totals[key][1] += ((predictions[i] == y[i]) & mask).sum().item()
                totals[key][2] += mask.sum().item()
    result = {k: {'answer_byte_nll': v[0]/v[2], 'answer_byte_accuracy': v[1]/v[2], 'bytes': v[2]} for k,v in totals.items()}
    if not all(math.isfinite(x['answer_byte_nll']) for x in result.values()):
        raise ValueError('non-finite metrics')
    return result


@torch.no_grad()
def generate(model, rows):
    model.eval()
    outputs = []
    for row in rows:
        prompt = torch.tensor([list(row['prompt'].encode())])
        logits, state = model(prompt)
        tokens = []
        for _ in range(24):
            token = logits[0, -1].argmax().item()
            tokens.append(token)
            if token == 10:
                break
            logits, state = model(torch.tensor([token]), state, token=True)
            logits = logits.unsqueeze(1)
        actual = bytes(tokens)
        outputs.append({**row, 'generated': actual.decode('utf-8', errors='replace'),
                        'generated_hex': actual.hex(), 'exact': actual == row['answer'].encode()})
    return outputs


def run(output, steps=160, wall_seconds=480):
    if not 1 <= steps <= 160 or not 1 <= wall_seconds <= 480:
        raise ValueError('bounded pilot requires steps<=160 and wall_seconds<=480')
    available = os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
    if available < 2_500_000_000 or shutil.disk_usage(output.parent).free < 1_000_000_000:
        raise RuntimeError('insufficient CPU pilot memory/disk')
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(97)
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)
    config = CandidateConfig(d_model=512, d_inner=768, n_layers=5)
    model = NativeLanguageCandidate(config)
    count = sum(p.numel() for p in model.parameters())
    if count != 9_979_914 or count != config.parameter_count():
        raise RuntimeError('actual parameter budget mismatch')
    splits = dataset()
    (output / 'dataset.json').write_text(json.dumps(splits, ensure_ascii=False, indent=2))
    initial_validation = evaluate(model, splits['validation'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01)
    rng = random.Random(97)
    history = []
    started = time.monotonic()
    # All training batches selected only from train. Test never selects checkpoint.
    model.train()
    for step in range(steps):
        x, y = batch(rng.sample(splits['train'], 4))
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        history.append({'step': step+1, 'loss': loss.item(), 'gradient_norm': grad.item()})
        if (step+1) % 20 == 0:
            print(json.dumps(history[-1]), flush=True)
        if time.monotonic() - started >= wall_seconds:
            break
    training_seconds = time.monotonic() - started
    del optimizer
    manifest = save_candidate(model, output / 'checkpoint')
    restored = load_candidate(output / 'checkpoint')
    x, _ = batch(splits['validation'][:1])
    with torch.no_grad():
        torch.testing.assert_close(model(x)[0], restored(x)[0], rtol=0, atol=0)
    del restored
    final = {k: evaluate(model, rows) for k, rows in splits.items() if k != 'train'}
    generated = generate(model, splits['test'][:8])
    report = {'schema': 'VN97ASC-10MPILOT1', 'source_commit': os.environ.get('GITHUB_SHA'),
              'seed': 97, 'torch_version': str(torch.__version__), 'config': asdict(config), 'parameters': count,
              'steps_requested': steps, 'steps_completed': len(history), 'training_seconds': training_seconds,
              'stop_reason': 'step_budget' if len(history)==steps else 'wall_budget',
              'batch_size': 4, 'learning_rate': .0003, 'max_sequence_bytes': 128,
              'available_memory_before_bytes': available, 'cpu_threads': torch.get_num_threads(),
              'recurrent_values_per_sample': 2*config.d_inner*config.n_layers,
              'initial_validation': initial_validation, 'final': final, 'history': history,
              'generation': generated, 'generation_exact': sum(x['exact'] for x in generated),
              'splits': {k: {'records':len(v),'sha256':hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True).encode()).hexdigest()} for k,v in splits.items()},
              'checkpoint': manifest, 'production_activation_authorized': False,
              'limitations': ['original synthetic tasks only, small data and one seed',
                             'split templates differ but task rules and some answers overlap',
                             'no equal-budget model comparison, general language/AGI/device/revenue evidence',
                             'weights saved exactly in FP32; optimizer/RNG not saved for exact training resume']}
    (output / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True))
    # Preserve exact FP32 weights while keeping each downloadable artifact <32 MiB.
    bundle = output / 'VN97-10M-pilot.zip'
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in ('report.json','dataset.json','checkpoint/manifest.json','checkpoint/weights.pt'):
            archive.write(output / name, name)
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    with bundle.open('rb') as handle:
        index = 0
        while block := handle.read(20*1024*1024):
            (output / f'bundle.part{index:02d}').write_bytes(block)
            index += 1
    (output / 'bundle-sha256.json').write_text(json.dumps({'sha256':digest,'parts':index,'bytes':bundle.stat().st_size}))
    print(json.dumps({'parameters':count,'steps_completed':len(history),'training_seconds':training_seconds,
                      'validation':final['validation']['all'],'test':final['test']['all'],
                      'free_generation_exact':report['generation_exact'],'free_generation_cases':len(generated)}), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    run(args.output)
