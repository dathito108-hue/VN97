"""Audit Python G08 against the pinned official tokenizer.json; not Android qualification."""
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    from tokenizers import Tokenizer
    path = Path(__file__).resolve().parents[1] / 'src/vn97/r2'
    spec = importlib.util.spec_from_file_location('_g08_audit', path / '__init__.py', submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    g08 = importlib.import_module('_g08_audit.gpt_neox_tokenizer')
    candidate = json.loads((args.candidate / 'CANDIDATE.json').read_text())
    assert candidate['runtime_id'] == '4f292ea6cf37554ea1ddefa97aa642572989ade51ae6745cf4f075cdc04dda21'
    assert candidate['production_activation_authorized'] is False
    root = args.candidate / 'tokenizer'
    expected_id = '27ce0a2f005befaa98ec4ee05d5d83a71aac1bcc5e4dd00032e33a17615ca575'
    descriptor = json.loads((root / 'tokenizer.vn97m2g08.json').read_text())
    rebuilt = g08.compile_g08_tokenizer_descriptor(root, capsule_id='8e1aff6c6b6d2947cb25111e48febae6850100fd782a2f112adcab2e047f674e', expected_token_id_space=50277)
    assert rebuilt == descriptor and descriptor['tokenizer_id'] == expected_id
    reference = Tokenizer.from_file(str(root / 'tokenizer.json'))
    native = g08.GptNeoXBpeTokenizer.from_files(root / 'vocab.json', root / 'merges.txt')
    texts = [
        '', 'Xin chào Việt Nam!', 'Tôi muốn VN97 trả lời chính xác.',
        'Đặng Thị Ánh, Nguyễn Văn Bình', 'a\u0301 e\u0302 o\u031b', '你好世界', 'こんにちは',
        'مرحبا بالعالم', 'Привет мир', '👩🏽‍💻 🚀 ❤️', '1234567890 -3.14 1e-8',
        'def f(x):\n    return x + 1\n', '{"tool":"file.read","path":"a.txt"}',
        "I'm we've he'll can't John's", '\t\n\r\n  ', 'a  b   c\n\n', '\u00a0\u2003\u200b',
        '<|endoftext|>', 'a<|endoftext|>b', 'https://example.test/a?q=VN97&x=1',
    ]
    cases = [prefix + text + suffix for text in texts for prefix, suffix in [('', ''), (' ', ''), ('', ' '), ('\n', '\n'), ('97:', ':42'), ('Tiếng Việt: ', '\n'), ('[', ']'), ('\t', '\t')]]
    corpus_hash = hashlib.sha256(json.dumps(cases, ensure_ascii=True, separators=(',', ':')).encode()).hexdigest()
    mismatches = []
    ids_exact = True
    bytes_exact = True
    for index, text in enumerate(cases):
        expected = reference.encode(text, add_special_tokens=False).ids
        actual = native.encode(text)
        ids_match = expected == actual
        expected_bytes = reference.decode(expected, skip_special_tokens=False).encode('utf-8')
        actual_bytes = native.decode_bytes(expected, skip_eos=False)
        bytes_match = expected_bytes == actual_bytes
        ids_exact &= ids_match
        bytes_exact &= bytes_match
        if not ids_match or not bytes_match:
            mismatches.append({'case': index, 'text': text, 'token_ids_exact': ids_match, 'decode_bytes_exact': bytes_match, 'expected_ids': expected, 'actual_ids': actual})
    report = {
        'schema': 'VN97G08PYREFERENCEAUDIT1', 'tokenizer_id': expected_id,
        'runtime_id': '4f292ea6cf37554ea1ddefa97aa642572989ade51ae6745cf4f075cdc04dda21',
        'reference': 'pinned official tokenizer.json via Hugging Face tokenizers',
        'reference_corpus_sha256': corpus_hash, 'cases': len(cases),
        'token_ids_exact': ids_exact, 'decode_bytes_exact': bytes_exact,
        'passed': ids_exact and bytes_exact, 'mismatches': mismatches,
        'android_tokenizer_tested': False, 'device_measured': False,
        'production_activation_authorized': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'mismatches'}, sort_keys=True))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
