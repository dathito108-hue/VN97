import os
import random

import pytest

from vn97 import (
    BYTE_BASE,
    LEARNED_BASE,
    VN97Tokenizer,
    VN97TokenizerPackage,
    learn_byte_bpe,
)


def test_byte_fallback_roundtrips_all_bytes():
    tokenizer = VN97Tokenizer()
    data = bytes(range(256))
    ids = tokenizer.encode_bytes(data)
    assert ids == [
        BYTE_BASE + value
        for value in range(256)
    ]
    assert (
        tokenizer.decode_bytes(ids)
        == data
    )


def test_utf8_roundtrip_and_controls():
    tokenizer = VN97Tokenizer()
    text = (
        "VN97 tiếng Việt — 日本語 — 🤖"
    )
    ids = tokenizer.encode(
        text,
        add_bos=True,
        add_text_tag=True,
        add_eos=True,
    )
    assert ids[0] == tokenizer.bos_id
    assert ids[1] == tokenizer.text_id
    assert ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids) == text

    with pytest.raises(ValueError):
        tokenizer.decode_bytes(
            ids,
            skip_control=False,
        )


def test_longest_prefix_roundtrip():
    package = VN97TokenizerPackage(
        (
            b"ab",
            b"abc",
            " Việt".encode("utf-8"),
        )
    )
    tokenizer = VN97Tokenizer(
        package
    )
    ids = tokenizer.encode_bytes(
        b"abcabx"
    )
    assert ids[0] == (
        LEARNED_BASE + 1
    )
    assert ids[1] == LEARNED_BASE
    assert ids[2] == (
        BYTE_BASE + ord("x")
    )
    assert (
        tokenizer.decode_bytes(ids)
        == b"abcabx"
    )


def test_package_roundtrip_and_corruption():
    package = VN97TokenizerPackage(
        (
            b"the",
            b"ing",
            " Việt".encode("utf-8"),
        )
    )
    blob = package.to_bytes()
    restored = (
        VN97TokenizerPackage.from_bytes(
            blob
        )
    )
    assert restored == package
    assert restored.vocab_size == (
        LEARNED_BASE + 3
    )

    with pytest.raises(ValueError):
        VN97TokenizerPackage.from_bytes(
            blob + b"x"
        )
    with pytest.raises(ValueError):
        VN97TokenizerPackage.from_bytes(
            b"BAD" + blob[3:]
        )


def test_random_binary_roundtrip():
    random.seed(97)
    package = VN97TokenizerPackage(
        (
            b"\x00\x01",
            b"\xff\x00",
            b"abc",
            b"abcd",
        )
    )
    tokenizer = VN97Tokenizer(
        package
    )

    for _ in range(100):
        data = os.urandom(
            random.randint(0, 128)
        )
        assert (
            tokenizer.decode_bytes(
                tokenizer.encode_bytes(
                    data
                )
            )
            == data
        )


def test_bpe_is_deterministic_and_reduces_repetition():
    corpus = [
        "banana banana",
        "bandana banana",
        "banana",
    ]
    first = learn_byte_bpe(
        corpus,
        max_learned_tokens=12,
        min_pair_count=2,
    )
    second = learn_byte_bpe(
        corpus,
        max_learned_tokens=12,
        min_pair_count=2,
    )
    assert first == second

    tokenizer = VN97Tokenizer(
        first
    )
    raw = b"banana banana"
    assert len(
        tokenizer.encode_bytes(raw)
    ) < len(raw)
    assert tokenizer.decode(
        tokenizer.encode(
            "banana banana"
        )
    ) == "banana banana"
