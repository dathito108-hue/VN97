from __future__ import annotations

import json
from pathlib import Path

import pytest

from vn97.production_corpus import prepare_corpus
from vn97.r2.production_corpus_scale import (
    R2D6_DEFINITION_SCHEMA,
    build_r2d6_corpus_index,
    load_r2d6_definition,
    verify_r2d6_corpus_index,
)
from vn97.tokenizer import VN97TokenizerPackage
from vn97.training import VN97ChatMessage


def _conversation(user: str, assistant: str):
    return (
        VN97ChatMessage(role="user", content=user),
        VN97ChatMessage(role="assistant", content=assistant),
    )


def _build_corpus(
    root: Path,
    label: str,
    *,
    training_record=None,
) -> Path:
    train = (
        training_record
        if training_record is not None
        else _conversation(
            f"{label} train question",
            f"{label} train answer",
        )
    )
    rows = (
        (
            f"{label}-train-source",
            f"https://example.test/{label}/train",
            "MIT",
            "training",
            "chat",
            f"{label}-train-raw".encode(),
            (train,),
        ),
        (
            f"{label}-validation-source",
            f"https://example.test/{label}/validation",
            "Apache-2.0",
            "validation",
            "chat",
            f"{label}-validation-raw".encode(),
            (
                _conversation(
                    f"{label} validation question",
                    f"{label} validation answer",
                ),
            ),
        ),
        (
            f"{label}-release-source",
            f"https://example.test/{label}/release",
            "CC-BY-4.0",
            "release",
            "chat",
            f"{label}-release-raw".encode(),
            (
                _conversation(
                    f"{label} release question",
                    f"{label} release answer",
                ),
            ),
        ),
    )
    manifest, prepared = prepare_corpus(
        rows,
        profile_id="vn97-production-intelligence-v1",
    )
    root.mkdir()
    for split in ("training", "validation", "release"):
        (root / f"{split}.jsonl").write_bytes(
            prepared[split].bytes
        )
    (root / "corpus.vn97corpus1.json").write_bytes(
        manifest.to_bytes()
    )
    return root


def _tokenizer(path: Path) -> Path:
    path.write_bytes(VN97TokenizerPackage().to_bytes())
    return path


def _definition(
    path: Path,
    corpora: list[tuple[str, list[str]]],
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": R2D6_DEFINITION_SCHEMA,
                "corpora": [
                    {
                        "path": corpus_path,
                        "task_families": families,
                    }
                    for corpus_path, families in corpora
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_r2d6_builds_deterministic_sharded_index(
    tmp_path: Path,
) -> None:
    _build_corpus(tmp_path / "corpus-a", "a")
    _build_corpus(tmp_path / "corpus-b", "b")
    tokenizer = _tokenizer(tmp_path / "tokenizer.vn97tk1")

    definition_a = _definition(
        tmp_path / "definition-a.json",
        [
            ("corpus-a", ["language"]),
            ("corpus-b", ["reasoning", "language"]),
        ],
    )
    definition_b = _definition(
        tmp_path / "definition-b.json",
        [
            ("corpus-b", ["reasoning", "language"]),
            ("corpus-a", ["language"]),
        ],
    )

    package_a = tmp_path / "package-a"
    package_b = tmp_path / "package-b"
    index_a = build_r2d6_corpus_index(
        corpus_inputs=load_r2d6_definition(definition_a),
        tokenizer_path=tokenizer,
        output_dir=package_a,
        sequence_length=128,
    )
    index_b = build_r2d6_corpus_index(
        corpus_inputs=load_r2d6_definition(definition_b),
        tokenizer_path=tokenizer,
        output_dir=package_b,
        sequence_length=128,
    )

    assert index_a["index_id"] == index_b["index_id"]
    assert len(index_a["source_manifests"]) == 2
    assert len(index_a["shards"]) == 6
    assert index_a["release_held_out"] is True
    assert index_a["split_totals"]["training"]["records"] == 2
    assert index_a["split_totals"]["validation"]["records"] == 2
    assert index_a["split_totals"]["release"]["records"] == 2
    assert index_a["scale"]["training_target_tokens"] > 0
    assert index_a["scale"]["tokens_per_parameter"] > 0.0
    assert index_a["scale"]["scale_floor_passed"] is False
    assert index_a["scale"]["scale_target_passed"] is False

    verified = verify_r2d6_corpus_index(package_a)
    assert verified["index_id"] == index_a["index_id"]


def test_r2d6_scale_policy_can_be_evaluated_without_quality_claim(
    tmp_path: Path,
) -> None:
    _build_corpus(tmp_path / "corpus", "scale")
    tokenizer = _tokenizer(tmp_path / "tokenizer.vn97tk1")
    definition = _definition(
        tmp_path / "definition.json",
        [("corpus", ["language"])],
    )

    payload = build_r2d6_corpus_index(
        corpus_inputs=load_r2d6_definition(definition),
        tokenizer_path=tokenizer,
        output_dir=tmp_path / "package",
        minimum_tokens_per_parameter=1e-12,
        target_tokens_per_parameter=1e-11,
    )
    scale = payload["scale"]
    assert scale["scale_floor_passed"] is True
    assert scale["scale_target_passed"] is True


def test_r2d6_rejects_cross_corpus_overlap(tmp_path: Path) -> None:
    shared = _conversation("shared question", "shared answer")
    _build_corpus(
        tmp_path / "corpus-a",
        "a",
        training_record=shared,
    )
    _build_corpus(
        tmp_path / "corpus-b",
        "b",
        training_record=shared,
    )
    tokenizer = _tokenizer(tmp_path / "tokenizer.vn97tk1")
    definition = _definition(
        tmp_path / "definition.json",
        [
            ("corpus-a", ["language"]),
            ("corpus-b", ["language"]),
        ],
    )

    with pytest.raises(ValueError, match="cross-corpus record overlap"):
        build_r2d6_corpus_index(
            corpus_inputs=load_r2d6_definition(definition),
            tokenizer_path=tokenizer,
            output_dir=tmp_path / "package",
        )


def test_r2d6_verify_rejects_shard_tamper(tmp_path: Path) -> None:
    _build_corpus(tmp_path / "corpus", "tamper")
    tokenizer = _tokenizer(tmp_path / "tokenizer.vn97tk1")
    definition = _definition(
        tmp_path / "definition.json",
        [("corpus", ["language"])],
    )
    package = tmp_path / "package"
    payload = build_r2d6_corpus_index(
        corpus_inputs=load_r2d6_definition(definition),
        tokenizer_path=tokenizer,
        output_dir=package,
    )
    training = next(
        item
        for item in payload["shards"]
        if item["split"] == "training"
    )
    shard = package / training["filename"]
    with shard.open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(ValueError, match="shard"):
        verify_r2d6_corpus_index(package)


def test_r2d6_definition_rejects_escape_and_duplicate_families(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / "outside-r2d6"
    outside.mkdir(exist_ok=True)
    escape = _definition(
        tmp_path / "escape.json",
        [("../outside-r2d6", ["language"])],
    )
    with pytest.raises(ValueError, match="stay relative"):
        load_r2d6_definition(escape)

    _build_corpus(tmp_path / "corpus", "dup")
    duplicate = _definition(
        tmp_path / "duplicate.json",
        [("corpus", ["language", "language"])],
    )
    with pytest.raises(ValueError, match="unique"):
        load_r2d6_definition(duplicate)


def test_r2d6_verify_rejects_manifest_tamper(tmp_path: Path) -> None:
    _build_corpus(tmp_path / "corpus", "manifest")
    tokenizer = _tokenizer(tmp_path / "tokenizer.vn97tk1")
    definition = _definition(
        tmp_path / "definition.json",
        [("corpus", ["language"])],
    )
    package = tmp_path / "package"
    payload = build_r2d6_corpus_index(
        corpus_inputs=load_r2d6_definition(definition),
        tokenizer_path=tokenizer,
        output_dir=package,
    )
    manifest = payload["source_manifests"][0]
    path = package / manifest["filename"]
    with path.open("ab") as handle:
        handle.write(b" ")

    with pytest.raises(ValueError, match="source manifest hash mismatch"):
        verify_r2d6_corpus_index(package)
