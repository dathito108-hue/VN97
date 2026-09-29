"""Compare real Python G08 implementation with verified official-reference fixtures."""
import argparse
import importlib
import json
from pathlib import Path
from g06_deployment_preflight import load, identity

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--tokenizer-root', type=Path, required=True)
p.add_argument('--reference', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
g08 = importlib.import_module('_g06_contracts.gpt_neox_tokenizer')
descriptor = load(a.tokenizer_root, 'tokenizer.vn97m2g08.json')
identity(descriptor, 'VN97M2G08TOK1', 'tokenizer_id')
reference = json.loads(a.reference.read_text())
assert reference['tokenizer_id'] == descriptor['tokenizer_id']
tokenizer = g08.GptNeoXBpeTokenizer.from_files(a.tokenizer_root / 'vocab.json', a.tokenizer_root / 'merges.txt')
mismatches = []
for i, case in enumerate(reference['cases']):
    try:
        ids = tokenizer.encode(case['text'])
        decoded = tokenizer.decode_bytes(case['ids']).hex()
        if ids != case['ids'] or decoded != case['decode_hex']:
            mismatches.append({'case': i, 'token_ids_exact': ids == case['ids'], 'decode_bytes_exact': decoded == case['decode_hex']})
    except Exception as error:
        mismatches.append({'case': i, 'error': str(error)})
report = {'schema': 'VN97G08PYREFERENCEAUDIT1', 'tokenizer_id': descriptor['tokenizer_id'],
          'corpus_sha256': reference['corpus_sha256'], 'cases': len(reference['cases']),
          'passed': not mismatches, 'mismatches': mismatches, 'device_measured': False,
          'production_activation_authorized': False}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
raise SystemExit(0 if report['passed'] else 1)
