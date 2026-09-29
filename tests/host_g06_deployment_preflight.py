"""Synthetic inventory/receipt fixtures; no inference or device qualification."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('preflight', ROOT / 'tools/g06_deployment_preflight.py')
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)
fixtures = runpy.run_path(str(ROOT / 'tests/host_mamba2_runtime_contract.py'))
# Reuse only the existing pure fixture builders; no pytest/training dependency.
fixture_ast = ast.parse((ROOT / 'tests/test_r2_mamba2_g10_promotion.py').read_text())
receipts = dict(vars(preflight.importlib.import_module('_g06_contracts.mamba2_promotion')))
fixture_functions = [node for node in fixture_ast.body if isinstance(node, ast.FunctionDef) and node.name.startswith('_')]
exec(compile(ast.Module(body=fixture_functions, type_ignores=[]), '<existing G10 fixtures>', 'exec'), receipts)


def seal(value, field):
    body = dict(value)
    body.pop(field, None)
    return {**body, field: hashlib.sha256(body['schema'].encode() + b'\0' + preflight.canonical(body)).hexdigest()}


def write(root, name, value):
    (root / name).write_text(json.dumps(value), encoding='ascii')


def fixture(root, evidence):
    runtime_dir = root / 'runtime'
    runtime_dir.mkdir()
    manifest = fixtures['bundle'](runtime_dir)
    write(evidence, 'manifest.vn97m2g05.json', manifest)
    runtime = preflight.compile_runtime(manifest)
    write(root, 'runtime/runtime.vn97m2g06.json', runtime)
    tokenizer_dir = root / 'tokenizer'
    tokenizer_dir.mkdir()
    assets = {}
    for name in ('vocab.json', 'merges.txt'):
        data = b'synthetic inventory bytes'
        (tokenizer_dir / name).write_bytes(data)
        assets[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    tokenizer = seal({'schema': 'VN97M2G08TOK1', 'capsule_id': runtime['capsule_id'], 'same_token_ids_required': True,
                      'production_activation_authorized': False, 'token_id_space': 50277, 'runtime_logits_size': 50288,
                      'eos_token_id': 0, 'assets': assets}, 'tokenizer_id')
    tuning = seal({'schema': 'VN97M2G07TUNE1', 'runtime_id': runtime['runtime_id'], 'profile_receipt_id': '7' * 64,
                   'profile_is_device_measured': True, 'production_activation_authorized': False}, 'tuning_id')
    bridge = preflight.compile_bridge(runtime, tokenizer, tuning)
    model, token, mobile = (receipts[name](bridge) for name in ('_model', '_token', '_mobile'))
    promotion = preflight.compile_promotion(bridge, model, token, mobile)
    for name, value in zip(list(preflight.DESCRIPTORS)[1:], (tokenizer, tuning, bridge, promotion)):
        write(root, name, value)
    for name, value in zip(preflight.EVIDENCE[1:], (model, token, mobile)):
        write(evidence, name, value)
    return runtime


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'deployment'
        self.evidence = Path(self.temp.name) / 'evidence'
        self.root.mkdir()
        self.evidence.mkdir()
        self.runtime = fixture(self.root, self.evidence)

    def test_integrity_pass_is_not_activation_or_device_evidence(self):
        report = preflight.verify(self.root, self.evidence)
        self.assertEqual(report['status'], 'PACKAGE_INTEGRITY_PASS')
        self.assertFalse(report['production_activation_authorized'])
        self.assertFalse(report['device_test_performed'])
        self.assertGreater(report['payload_bytes_verified'], 0)

    def test_corrupt_graph_and_tokenizer(self):
        for relative in ('runtime/' + self.runtime['graph_filename'], 'tokenizer/vocab.json'):
            path = self.root / relative
            original = path.read_bytes()
            path.write_bytes(b'x' * len(original))
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                preflight.verify(self.root, self.evidence)
            path.write_bytes(original)

    def test_missing_receipt_and_descriptor_reported_together(self):
        (self.evidence / 'mobile-qualification.json').unlink()
        (self.root / 'tuning.vn97m2g07.json').unlink()
        with self.assertRaises(ValueError) as error:
            preflight.verify(self.root, self.evidence)
        self.assertIn('mobile-qualification.json', str(error.exception))
        self.assertIn('tuning.vn97m2g07.json', str(error.exception))

    def test_resealed_mobile_failure_cannot_promote(self):
        path = self.evidence / 'mobile-qualification.json'
        value = json.loads(path.read_text())
        value['latency_passed'] = False
        write(self.evidence, path.name, seal(value, 'receipt_id'))
        with self.assertRaisesRegex(ValueError, 'latency'):
            preflight.verify(self.root, self.evidence)

    def test_resealed_tuning_wrong_lineage(self):
        value = json.loads((self.root / 'tuning.vn97m2g07.json').read_text())
        value['runtime_id'] = '0' * 64
        write(self.root, 'tuning.vn97m2g07.json', seal(value, 'tuning_id'))
        with self.assertRaisesRegex(ValueError, 'another runtime'):
            preflight.verify(self.root, self.evidence)

    def test_duplicate_json_key(self):
        path = self.root / 'promotion.vn97m2g10.json'
        path.write_text('{"schema":"a","schema":"b"}')
        with self.assertRaisesRegex(ValueError, 'duplicate JSON key'):
            preflight.verify(self.root, self.evidence)

    def test_symlink_payload(self):
        path = self.root / 'tokenizer/vocab.json'
        path.unlink()
        path.symlink_to(self.root / 'tokenizer/merges.txt')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            preflight.verify(self.root, self.evidence)

    def test_payload_path_escape(self):
        with self.assertRaisesRegex(ValueError, 'unsafe payload'):
            preflight.payload(self.root, '../outside', {'bytes': 1, 'sha256': '0' * 64})

    def test_cli_exit_status_and_json(self):
        command = [sys.executable, str(ROOT / 'tools/g06_deployment_preflight.py'), '--root', str(self.root), '--evidence-dir', str(self.evidence)]
        passed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(passed.returncode, 0, passed.stderr)
        self.assertFalse(json.loads(passed.stdout)['production_activation_authorized'])
        (self.evidence / 'model-parity.json').unlink()
        blocked = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(blocked.returncode, 1, blocked.stderr)
        self.assertEqual(json.loads(blocked.stdout)['status'], 'BLOCKED')


if __name__ == '__main__':
    unittest.main(verbosity=2)
