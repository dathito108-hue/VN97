"""Offline package integrity/evidence check; never inference or device qualification."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

# Load the existing dependency-free contracts without the training facade.
CONTRACTS = Path(__file__).resolve().parents[1] / 'src/vn97/r2'
SPEC = importlib.util.spec_from_file_location('_g06_contracts', CONTRACTS / '__init__.py', submodule_search_locations=[str(CONTRACTS)])
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
compile_runtime = importlib.import_module('_g06_contracts.mamba2_android_runtime').compile_g06_runtime_descriptor
compile_bridge = importlib.import_module('_g06_contracts.mamba2_cognition_bridge').compile_g09_bridge_binding
compile_promotion = importlib.import_module('_g06_contracts.mamba2_promotion').compile_g10_promotion

DESCRIPTORS = {
    'runtime/runtime.vn97m2g06.json': ('VN97M2G06RUNTIME1', 'runtime_id'),
    'tokenizer/tokenizer.vn97m2g08.json': ('VN97M2G08TOK1', 'tokenizer_id'),
    'tuning.vn97m2g07.json': ('VN97M2G07TUNE1', 'tuning_id'),
    'binding.vn97m2g09.json': ('VN97M2G09BRIDGE1', 'bridge_id'),
    'promotion.vn97m2g10.json': ('VN97M2G10PROMOTE1', 'promotion_id'),
}
EVIDENCE = ('manifest.vn97m2g05.json', 'model-parity.json', 'token-parity.json', 'mobile-qualification.json')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def regular(root, relative):
    path = root
    for part in Path(relative).parts:
        if part in ('.', '..') or not part or '/' in part or '\\' in part or ':' in part:
            raise ValueError(f'unsafe path: {relative}')
        path = path / part
        if path.is_symlink():
            raise ValueError(f'symlink not permitted: {path}')
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'missing/unsafe regular file: {path}')
    return path


def load(root, name):
    path = regular(root, name)
    if path.stat().st_size > 1024 * 1024:
        raise ValueError(f'descriptor exceeds 1 MiB: {name}')
    def reject(value):
        raise ValueError(f'non-finite JSON number: {value}')
    value = json.loads(path.read_text(encoding='ascii'), object_pairs_hook=unique, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f'JSON object required: {name}')
    return value


def identity(value, schema, field):
    body = dict(value)
    actual = body.pop(field, None)
    expected = hashlib.sha256(schema.encode('ascii') + b'\0' + canonical(body)).hexdigest()
    if value.get('schema') != schema or actual != expected:
        raise ValueError(f'{schema}: invalid {field}')


def payload(root, name, record):
    if not isinstance(name, str) or not name or name in ('.', '..') or any(c in name for c in '/\\:') or any(ord(c) < 32 for c in name):
        raise ValueError('unsafe payload filename')
    if not isinstance(record, dict) or set(record) != {'bytes', 'sha256'}:
        raise ValueError(f'invalid inventory entry: {name}')
    size = record['bytes']
    if type(size) is not int or size <= 0:
        raise ValueError(f'invalid payload byte count: {name}')
    path = regular(root, name)
    if path.stat().st_size != size:
        raise ValueError(f'payload size mismatch: {name}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != record['sha256']:
        raise ValueError(f'payload SHA-256 mismatch: {name}')
    return size


def verify(root: Path, evidence: Path):
    for folder in (root, evidence):
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError(f'missing/unsafe directory: {folder}')
    missing = [str(root / name) for name in DESCRIPTORS if not (root / name).is_file()]
    missing += [str(evidence / name) for name in EVIDENCE if not (evidence / name).is_file()]
    if missing:
        raise ValueError('missing required files: ' + ', '.join(missing))
    objects = []
    for name, (schema, field) in DESCRIPTORS.items():
        value = load(root, name)
        identity(value, schema, field)
        objects.append(value)
    runtime, tokenizer, tuning, bridge, promotion = objects
    manifest = load(evidence, EVIDENCE[0])
    identity(manifest, 'VN97M2G05ONNX1', 'manifest_id')
    if runtime != compile_runtime(manifest):
        raise ValueError('G06 runtime differs from the G05 export contract')
    if bridge != compile_bridge(runtime, tokenizer, tuning):
        raise ValueError('G09 binding differs from deployment components')
    expected_promotion = compile_promotion(bridge, *(load(evidence, name) for name in EVIDENCE[1:]))
    if promotion != expected_promotion:
        raise ValueError('G10 promotion differs from supplied qualification receipts')
    total = 0
    seen = set()
    for item in runtime['graph_files']:
        name = item.get('filename')
        if name in seen:
            raise ValueError('duplicate graph inventory filename')
        seen.add(name)
        total += payload(root / 'runtime', name, {k: v for k, v in item.items() if k != 'filename'})
    assets = tokenizer.get('assets')
    if not isinstance(assets, dict) or set(assets) != {'vocab.json', 'merges.txt'}:
        raise ValueError('G08 tokenizer inventory must contain vocab.json and merges.txt')
    for name, record in assets.items():
        total += payload(root / 'tokenizer', name, record)
    return {
        'schema': 'VN97G06PREFLIGHT1', 'status': 'PACKAGE_INTEGRITY_PASS',
        'runtime_id': runtime['runtime_id'], 'promotion_id': promotion['promotion_id'],
        'payload_bytes_verified': total, 'device_test_performed': False,
        'production_activation_authorized': False,
        'note': 'Verifies supplied files and receipt consistency; not publisher authentication, ONNX execution or a new physical-device measurement.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--evidence-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args.root, args.evidence_dir)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({'schema': 'VN97G06PREFLIGHT1', 'status': 'BLOCKED', 'reason': str(error), 'production_activation_authorized': False}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
