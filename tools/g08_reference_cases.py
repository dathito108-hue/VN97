"""Create exact reference cases from verified candidate tokenizer assets."""
import argparse
import hashlib
import json
from pathlib import Path
from g06_deployment_preflight import load, identity, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer-root', required=True, type=Path)
    parser.add_argument('--corpus', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    from tokenizers import Tokenizer
    descriptor = load(args.tokenizer_root, 'tokenizer.vn97m2g08.json')
    identity(descriptor, 'VN97M2G08TOK1', 'tokenizer_id')
    assert descriptor['tokenizer_id'] == '27ce0a2f005befaa98ec4ee05d5d83a71aac1bcc5e4dd00032e33a17615ca575'
    for name, record in descriptor['assets'].items():
        payload(args.tokenizer_root, name, record)
    corpus = json.loads(args.corpus.read_text())
    assert len(corpus) >= 100 and all(isinstance(text, str) for text in corpus)
    reference = Tokenizer.from_file(str(args.tokenizer_root / 'tokenizer.json'))
    cases = []
    for text in corpus:
        ids = reference.encode(text, add_special_tokens=False).ids
        cases.append({'text': text, 'ids': ids, 'decode_hex': reference.decode(ids, skip_special_tokens=False).encode('utf-8').hex()})
    report = {'schema': 'VN97G08REFERENCECASES1', 'tokenizer_id': descriptor['tokenizer_id'],
              'corpus_sha256': hashlib.sha256(json.dumps(corpus, ensure_ascii=True, separators=(',', ':')).encode()).hexdigest(),
              'cases': cases, 'production_activation_authorized': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=True, indent=2) + '\n')
    print('Official reference cases:', len(cases))


if __name__ == '__main__':
    main()
