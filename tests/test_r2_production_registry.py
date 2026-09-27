from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vn97.r2.production_acquisition import build_r2d9_campaign
from vn97.r2.production_corpus_scale import (
    build_r2d6_corpus_index,
    load_r2d6_definition,
)
from vn97.r2.production_registry import (
    R2D11_DEFINITION_SCHEMA,
    admit_r2d11_pack,
    attach_r2d11_campaign,
    init_r2d11_registry,
    verify_r2d11_registry,
)
from vn97.r2.production_source_adapter import (
    R2D10_LOCK_SCHEMA,
    build_r2d10_pack,
)
from vn97.tokenizer import VN97TokenizerPackage


def _rows(prefix: str, count: int = 8) -> list[dict[str, object]]:
    return [
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"{prefix} question {index}",
                },
                {
                    "role": "assistant",
                    "content": f"{prefix} answer {index}",
                },
            ]
        }
        for index in range(count)
    ]


def _write_jsonl(
    path: Path,
    rows: list[dict[str, object]],
) -> tuple[str, int, int]:
    data = b"".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest(), len(rows), len(data)


def _build_pack(
    root: Path,
    *,
    source_id: str,
    family: str,
    prefix: str,
    rows: list[dict[str, object]] | None = None,
) -> Path:
    root.mkdir()
    raw_rows = _rows(prefix) if rows is None else rows
    sha, records, raw_bytes = _write_jsonl(
        root / "raw.jsonl",
        raw_rows,
    )
    lock = {
        "schema": R2D10_LOCK_SCHEMA,
        "profile_id": "vn97-production-intelligence-v1",
        "shard_target_training_records": 10000,
        "validation_fraction": 0.125,
        "release_fraction": 0.125,
        "sources": [
            {
                "source_id": source_id,
                "origin": f"https://example.test/{source_id}",
                "revision": "release-2026-09-27",
                "license": "MIT",
                "license_approved": True,
                "family": family,
                "adapter": "messages",
                "path": "raw.jsonl",
                "expected_sha256": sha,
                "expected_records": records,
                "max_bytes": raw_bytes,
            }
        ],
    }
    lock_path = root / "lock.json"
    lock_path.write_text(
        json.dumps(lock, sort_keys=True),
        encoding="utf-8",
    )
    pack = root / "pack"
    build_r2d10_pack(
        lock_path=lock_path,
        output_dir=pack,
    )
    return pack


def _build_campaign_and_d6(
    root: Path,
    *,
    pack: Path,
    tokenizer: Path,
) -> tuple[Path, Path]:
    campaign = root / "campaign"
    build_r2d9_campaign(
        definition_path=pack / "r2d9-definition.json",
        output_dir=campaign,
    )
    d6 = root / "d6"
    build_r2d6_corpus_index(
        corpus_inputs=load_r2d6_definition(
            campaign / "r2d6-definition.json"
        ),
        tokenizer_path=tokenizer,
        output_dir=d6,
        sequence_length=64,
    )
    return campaign, d6


def _init_registry(tmp_path: Path) -> tuple[Path, Path]:
    tokenizer = tmp_path / "tokenizer.vn97tk1"
    tokenizer.write_bytes(VN97TokenizerPackage().to_bytes())
    definition = tmp_path / "registry-definition.json"
    definition.write_text(
        json.dumps(
            {
                "schema": R2D11_DEFINITION_SCHEMA,
                "sequence_length": 64,
                "family_weights": {
                    "language": 0.5,
                    "reasoning": 0.5,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "registry"
    init_r2d11_registry(
        definition_path=definition,
        tokenizer_path=tokenizer,
        registry_dir=registry,
    )
    return registry, tokenizer


def test_r2d11_incremental_batches_preserve_prior_evidence(
    tmp_path: Path,
) -> None:
    registry, tokenizer = _init_registry(tmp_path)

    language_pack = _build_pack(
        tmp_path / "language-batch",
        source_id="language-batch-1",
        family="language",
        prefix="language-one",
    )
    admitted_language = admit_r2d11_pack(
        registry_dir=registry,
        pack_dir=language_pack,
    )
    language_pack_id = admitted_language["event"]["pack_id"]
    language_digest = (
        registry / "digests" / f"{language_pack_id}.jsonl"
    )
    language_digest_sha = hashlib.sha256(
        language_digest.read_bytes()
    ).hexdigest()

    language_campaign, language_d6 = _build_campaign_and_d6(
        tmp_path / "language-products",
        pack=language_pack,
        tokenizer=tokenizer,
    )
    attached_language = attach_r2d11_campaign(
        registry_dir=registry,
        pack_dir=language_pack,
        campaign_dir=language_campaign,
        d6_package_dir=language_d6,
    )
    assert attached_language["state"][
        "training_target_tokens_by_family"
    ]["language"] > 0
    assert attached_language["state"][
        "training_target_tokens_by_family"
    ]["reasoning"] == 0

    reasoning_pack = _build_pack(
        tmp_path / "reasoning-batch",
        source_id="reasoning-batch-1",
        family="reasoning",
        prefix="reasoning-one",
    )
    admitted_reasoning = admit_r2d11_pack(
        registry_dir=registry,
        pack_dir=reasoning_pack,
    )
    reasoning_pack_id = admitted_reasoning["event"]["pack_id"]
    assert reasoning_pack_id != language_pack_id

    # Incremental admission must never rewrite the first batch digest shard.
    assert hashlib.sha256(
        language_digest.read_bytes()
    ).hexdigest() == language_digest_sha

    reasoning_campaign, reasoning_d6 = _build_campaign_and_d6(
        tmp_path / "reasoning-products",
        pack=reasoning_pack,
        tokenizer=tokenizer,
    )
    final_snapshot = attach_r2d11_campaign(
        registry_dir=registry,
        pack_dir=reasoning_pack,
        campaign_dir=reasoning_campaign,
        d6_package_dir=reasoning_d6,
    )

    assert final_snapshot["generation"] == 4
    state = final_snapshot["state"]
    assert state["admitted_batches"] == sorted(
        [language_pack_id, reasoning_pack_id]
    )
    assert state["attached_batches"] == sorted(
        [language_pack_id, reasoning_pack_id]
    )
    assert state["training_target_tokens_by_family"]["language"] > 0
    assert state["training_target_tokens_by_family"]["reasoning"] > 0
    assert state["global_tokens_per_parameter"] > 0.0
    assert state["production_floor_ready"] is False
    assert (
        state["family_progress"]["language"]["floor_missing_tokens"]
        > 0
    )
    assert (
        state["family_progress"]["reasoning"]["floor_missing_tokens"]
        > 0
    )

    verified = verify_r2d11_registry(registry)
    assert verified["state"] == state


def test_r2d11_rejects_cross_batch_duplicate_record(
    tmp_path: Path,
) -> None:
    registry, _ = _init_registry(tmp_path)
    shared_rows = _rows("shared")

    first = _build_pack(
        tmp_path / "first",
        source_id="first-source",
        family="language",
        prefix="unused",
        rows=shared_rows,
    )
    admit_r2d11_pack(
        registry_dir=registry,
        pack_dir=first,
    )

    second_rows = [shared_rows[0]] + _rows("second")[1:]
    second = _build_pack(
        tmp_path / "second",
        source_id="second-source",
        family="language",
        prefix="unused",
        rows=second_rows,
    )
    with pytest.raises(ValueError, match="overlaps"):
        admit_r2d11_pack(
            registry_dir=registry,
            pack_dir=second,
        )


def test_r2d11_rejects_cross_family_duplicate_inside_pack(
    tmp_path: Path,
) -> None:
    registry, _ = _init_registry(tmp_path)
    root = tmp_path / "mixed"
    root.mkdir()
    shared = _rows("shared", count=1)[0]
    language_rows = [shared] + _rows("language-extra", 7)
    reasoning_rows = [shared] + _rows("reasoning-extra", 7)
    l_sha, l_records, l_bytes = _write_jsonl(
        root / "language.jsonl",
        language_rows,
    )
    r_sha, r_records, r_bytes = _write_jsonl(
        root / "reasoning.jsonl",
        reasoning_rows,
    )
    lock = {
        "schema": R2D10_LOCK_SCHEMA,
        "profile_id": "vn97-production-intelligence-v1",
        "shard_target_training_records": 10000,
        "validation_fraction": 0.125,
        "release_fraction": 0.125,
        "sources": [
            {
                "source_id": "language-source",
                "origin": "https://example.test/language",
                "revision": "v1",
                "license": "MIT",
                "license_approved": True,
                "family": "language",
                "adapter": "messages",
                "path": "language.jsonl",
                "expected_sha256": l_sha,
                "expected_records": l_records,
                "max_bytes": l_bytes,
            },
            {
                "source_id": "reasoning-source",
                "origin": "https://example.test/reasoning",
                "revision": "v1",
                "license": "MIT",
                "license_approved": True,
                "family": "reasoning",
                "adapter": "messages",
                "path": "reasoning.jsonl",
                "expected_sha256": r_sha,
                "expected_records": r_records,
                "max_bytes": r_bytes,
            },
        ],
    }
    lock_path = root / "lock.json"
    lock_path.write_text(
        json.dumps(lock, sort_keys=True),
        encoding="utf-8",
    )
    pack = root / "pack"
    build_r2d10_pack(
        lock_path=lock_path,
        output_dir=pack,
    )

    with pytest.raises(ValueError, match="crosses task families"):
        admit_r2d11_pack(
            registry_dir=registry,
            pack_dir=pack,
        )


def test_r2d11_attach_rejects_campaign_from_other_pack(
    tmp_path: Path,
) -> None:
    registry, tokenizer = _init_registry(tmp_path)

    first = _build_pack(
        tmp_path / "first",
        source_id="first-source",
        family="language",
        prefix="first",
    )
    second = _build_pack(
        tmp_path / "second",
        source_id="second-source",
        family="language",
        prefix="second",
    )
    admit_r2d11_pack(
        registry_dir=registry,
        pack_dir=first,
    )
    second_campaign, second_d6 = _build_campaign_and_d6(
        tmp_path / "second-products",
        pack=second,
        tokenizer=tokenizer,
    )

    with pytest.raises(ValueError, match="D10/D9"):
        attach_r2d11_campaign(
            registry_dir=registry,
            pack_dir=first,
            campaign_dir=second_campaign,
            d6_package_dir=second_d6,
        )


def test_r2d11_registry_detects_ledger_tamper(
    tmp_path: Path,
) -> None:
    registry, _ = _init_registry(tmp_path)
    pack = _build_pack(
        tmp_path / "batch",
        source_id="source",
        family="language",
        prefix="batch",
    )
    admit_r2d11_pack(
        registry_dir=registry,
        pack_dir=pack,
    )

    ledger = registry / "ledger" / "000001.json"
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    payload["event"]["unique_records"] += 1
    ledger.write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="snapshot identity"):
        verify_r2d11_registry(registry)


def test_r2d11_registry_detects_digest_tamper(
    tmp_path: Path,
) -> None:
    registry, _ = _init_registry(tmp_path)
    pack = _build_pack(
        tmp_path / "batch",
        source_id="source",
        family="language",
        prefix="batch",
    )
    admitted = admit_r2d11_pack(
        registry_dir=registry,
        pack_dir=pack,
    )
    digest = (
        registry
        / admitted["event"]["digest_path"]
    )
    with digest.open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(ValueError, match="digest shard hash mismatch"):
        verify_r2d11_registry(registry)
