from __future__ import annotations

import importlib.util
from pathlib import Path
import random
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "vn97"

pkg = types.ModuleType("vn97")
pkg.__path__ = [str(SRC)]
sys.modules["vn97"] = pkg

spec = importlib.util.spec_from_file_location(
    "vn97.tokenizer",
    SRC / "tokenizer.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load tokenizer")
module = importlib.util.module_from_spec(spec)
sys.modules["vn97.tokenizer"] = module
spec.loader.exec_module(module)


def main() -> None:
    corpora = [
        ["banana bandana", "ananas"],
        ["xin chào", "chào bạn", "xin chào bạn"],
        [b"aaaaabaaaa", b"abababab", b"xyzxyzxyz"],
    ]
    rng = random.Random(97)
    alphabet = "abcde 123"
    corpora.append(
        [
            "".join(
                rng.choice(alphabet)
                for _ in range(rng.randint(5, 30))
            )
            for _ in range(30)
        ]
    )

    sample_source = [
        "record-z",
        "record-a",
        "record-c",
        "record-b",
    ]
    sample_a = module.select_deterministic_tokenizer_corpus(
        sample_source,
        max_samples=2,
    )
    sample_b = module.select_deterministic_tokenizer_corpus(
        list(reversed(sample_source)),
        max_samples=2,
    )
    assert set(sample_a) == set(sample_b)
    assert len(sample_a) == 2

    for corpus in corpora:
        for merges in (0, 1, 2, 8, 32):
            expected = module.learn_byte_bpe(
                corpus,
                max_learned_tokens=merges,
                min_pair_count=2,
            )
            actual = module.learn_byte_bpe_fast(
                corpus,
                max_learned_tokens=merges,
                min_pair_count=2,
            )
            assert actual == expected, (
                corpus,
                merges,
                expected.learned_tokens,
                actual.learned_tokens,
            )

    print("VN97 fast byte-BPE exact-equivalence PASS")


if __name__ == "__main__":
    main()
