import json
from pathlib import Path

import pytest

from vn97.r2.gpt_neox_tokenizer import (
    GptNeoXBpeTokenizer,
    bytes_to_unicode,
    compile_g08_tokenizer_descriptor,
    gpt_neox_pretokenize,
)


def _write_tokenizer(root: Path) -> GptNeoXBpeTokenizer:
    root.mkdir(parents=True)
    byte_encoder, _ = bytes_to_unicode()

    vocab: dict[str, int] = {}
    for value in range(256):
        vocab[byte_encoder[value]] = value

    eos = "<|endoftext|>"
    vocab[eos] = len(vocab)

    merges = [
        ("H", "e"),
        ("He", "l"),
        ("Hel", "l"),
        ("Hell", "o"),
        (byte_encoder[32], "w"),
        (byte_encoder[32] + "w", "o"),
        (byte_encoder[32] + "wo", "r"),
        (byte_encoder[32] + "wor", "l"),
        (byte_encoder[32] + "worl", "d"),
    ]
    for left, right in merges:
        merged = left + right
        if merged not in vocab:
            vocab[merged] = len(vocab)

    (root / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    (root / "merges.txt").write_text(
        "#version: 0.2\n"
        + "".join(f"{left} {right}\n" for left, right in merges),
        encoding="utf-8",
    )
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "unk_token": eos,
                "bos_token": eos,
                "eos_token": eos,
                "add_prefix_space": False,
                "tokenizer_class": "GPTNeoXTokenizer",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "bos_token": eos,
                "eos_token": eos,
                "unk_token": eos,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "model": {"type": "BPE"},
                "pre_tokenizer": {"type": "ByteLevel"},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return GptNeoXBpeTokenizer.from_files(
        root / "vocab.json",
        root / "merges.txt",
        eos_token=eos,
    )


def test_gpt_neox_pretokenizer_space_and_contraction_semantics() -> None:
    assert gpt_neox_pretokenize("Hello world") == (
        "Hello",
        " world",
    )
    assert gpt_neox_pretokenize("Hello  world") == (
        "Hello",
        " ",
        " world",
    )
    assert gpt_neox_pretokenize("we're ready") == (
        "we",
        "'re",
        " ready",
    )
    assert gpt_neox_pretokenize("A\nB") == (
        "A",
        "\n",
        "B",
    )


def test_gpt_neox_bpe_roundtrip_and_space_sensitive_tokens(
    tmp_path: Path,
) -> None:
    tokenizer = _write_tokenizer(tmp_path / "tokenizer")
    hello = tokenizer.encode("Hello")
    hello_space = tokenizer.encode(" Hello")
    assert hello != hello_space
    assert tokenizer.decode(hello) == "Hello"
    assert tokenizer.decode(hello_space) == " Hello"
    assert tokenizer.decode(tokenizer.encode("Hello world")) == "Hello world"
    assert tokenizer.decode(tokenizer.encode("xin chào")) == "xin chào"
    special = tokenizer.eos_token
    assert tokenizer.encode(special) == [tokenizer.eos_token_id]
    assert tokenizer.decode([tokenizer.eos_token_id]) == special


def test_g08_descriptor_binds_assets_and_padded_sampling_mask(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tokenizer"
    tokenizer = _write_tokenizer(root)
    descriptor = compile_g08_tokenizer_descriptor(
        root,
        capsule_id="a" * 64,
        expected_token_id_space=tokenizer.token_id_space,
    )
    assert descriptor["schema"] == "VN97M2G08TOK1"
    assert descriptor["tokenizer_class"] == "GPTNeoXTokenizer"
    assert descriptor["add_prefix_space"] is False
    assert descriptor["bos_token_id"] == tokenizer.eos_token_id
    assert descriptor["eos_token_id"] == tokenizer.eos_token_id
    assert descriptor["unk_token_id"] == tokenizer.eos_token_id
    assert descriptor["runtime_logits_size"] == 50288
    assert (
        descriptor["invalid_padded_token_start"]
        == tokenizer.token_id_space
    )
    assert descriptor["sampling_must_mask_padded_ids"] is True
    assert descriptor["same_token_ids_required"] is True
    assert descriptor["production_activation_authorized"] is False
    assert len(descriptor["tokenizer_id"]) == 64


def test_g08_descriptor_rejects_wrong_token_id_space(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tokenizer"
    tokenizer = _write_tokenizer(root)
    with pytest.raises(ValueError, match="ID space"):
        compile_g08_tokenizer_descriptor(
            root,
            capsule_id="b" * 64,
            expected_token_id_space=tokenizer.token_id_space + 1,
        )


def test_gpt_neox_decode_rejects_padded_runtime_id(
    tmp_path: Path,
) -> None:
    tokenizer = _write_tokenizer(tmp_path / "tokenizer")
    with pytest.raises(ValueError, match="out of vocabulary"):
        tokenizer.decode([tokenizer.token_id_space])
