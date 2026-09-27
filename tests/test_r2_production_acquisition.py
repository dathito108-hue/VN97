from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vn97.r2.production_acquisition import (
    R2D9_DEFINITION_SCHEMA,
    R2D9_PROFILE_ID,
    build_r2d9_campaign,
    load_r2d9_definition,
    verify_r2d9_campaign,
)


def _record(tag: str) -> dict[str, object]:
    return {
        "messages": [
            {"role": "user", "content": f"{tag} question"},
            {"role": "assistant", "content": f"{tag} answer"},
        ]
    }


def _write_source(
    path: Path,
    tags: list[str],
) -> tuple[str, int]:
    data = b"".join(
        json.dumps(
            _record(tag),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        for tag in tags
    )
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest(), len(tags)


def _source_entry(
    *,
    source_id: str,
    family: str,
    filename: str,
    sha256: str,
    records: int,
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "origin": f"https://example.test/{source_id}",
        "revision": "2026-09-27",
        "license": "MIT",
        "license_approved": True,
        "family": family,
        "path": filename,
        "expected_sha256": sha256,
        "expected_records": records,
        "max_bytes": 1024 * 1024,
    }


def _definition(
    path: Path,
    sources: list[dict[str, object]],
    *,
    target: int = 3,
    validation_fraction: float = 0.2,
    release_fraction: float = 0.2,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": R2D9_DEFINITION_SCHEMA,
                "profile_id": R2D9_PROFILE_ID,
                "shard_target_training_records": target,
                "validation_fraction": validation_fraction,
                "release_fraction": release_fraction,
                "sources": sources,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_r2d9_builds_multi_family_sealed_campaign(
    tmp_path: Path,
) -> None:
    language_sha, language_records = _write_source(
        tmp_path / "language.jsonl",
        [f"language-{index}" for index in range(10)],
    )
    reasoning_sha, reasoning_records = _write_source(
        tmp_path / "reasoning.jsonl",
        [f"reasoning-{index}" for index in range(10)],
    )
    definition = _definition(
        tmp_path / "definition.json",
        [
            _source_entry(
                source_id="language-source",
                family="language",
                filename="language.jsonl",
                sha256=language_sha,
                records=language_records,
            ),
            _source_entry(
                source_id="reasoning-source",
                family="reasoning",
                filename="reasoning.jsonl",
                sha256=reasoning_sha,
                records=reasoning_records,
            ),
        ],
    )

    campaign_dir = tmp_path / "campaign"
    campaign = build_r2d9_campaign(
        definition_path=definition,
        output_dir=campaign_dir,
    )

    assert campaign["global_unique_records"] == 20
    assert set(campaign["family_totals"]) == {
        "language",
        "reasoning",
    }
    assert campaign["family_totals"]["language"] == {
        "unique_records": 10,
        "seals": 2,
        "training_records": 6,
        "validation_records": 2,
        "release_records": 2,
    }
    assert campaign["family_totals"]["reasoning"] == {
        "unique_records": 10,
        "seals": 2,
        "training_records": 6,
        "validation_records": 2,
        "release_records": 2,
    }
    assert len(campaign["seals"]) == 4

    d6 = json.loads(
        (campaign_dir / "r2d6-definition.json").read_text(
            encoding="utf-8"
        )
    )
    assert d6["schema"] == "VN97R2D6DEF1"
    assert len(d6["corpora"]) == 4
    assert all(
        len(item["task_families"]) == 1
        for item in d6["corpora"]
    )

    primary = json.loads(
        (campaign_dir / "r2d8-primary-family.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(primary) == 4
    assert set(primary.values()) == {"language", "reasoning"}

    verified = verify_r2d9_campaign(campaign_dir)
    assert verified["campaign_id"] == campaign["campaign_id"]


def test_r2d9_deduplicates_same_family_across_sources(
    tmp_path: Path,
) -> None:
    a_sha, a_records = _write_source(
        tmp_path / "a.jsonl",
        [f"a-{index}" for index in range(9)] + ["shared"],
    )
    b_sha, b_records = _write_source(
        tmp_path / "b.jsonl",
        ["shared"] + [f"b-{index}" for index in range(9)],
    )
    definition = _definition(
        tmp_path / "definition.json",
        [
            _source_entry(
                source_id="a-source",
                family="language",
                filename="a.jsonl",
                sha256=a_sha,
                records=a_records,
            ),
            _source_entry(
                source_id="b-source",
                family="language",
                filename="b.jsonl",
                sha256=b_sha,
                records=b_records,
            ),
        ],
        target=6,
    )

    campaign = build_r2d9_campaign(
        definition_path=definition,
        output_dir=tmp_path / "campaign",
    )
    assert campaign["global_unique_records"] == 19

    receipts = {
        item["source_id"]: item
        for item in campaign["source_receipts"]
    }
    assert receipts["a-source"]["accepted_records"] == 10
    assert receipts["a-source"]["duplicate_records"] == 0
    assert receipts["b-source"]["accepted_records"] == 9
    assert receipts["b-source"]["duplicate_records"] == 1


def test_r2d9_rejects_duplicate_crossing_task_families(
    tmp_path: Path,
) -> None:
    a_sha, a_records = _write_source(
        tmp_path / "a.jsonl",
        ["shared"] + [f"a-{index}" for index in range(9)],
    )
    b_sha, b_records = _write_source(
        tmp_path / "b.jsonl",
        ["shared"] + [f"b-{index}" for index in range(9)],
    )
    definition = _definition(
        tmp_path / "definition.json",
        [
            _source_entry(
                source_id="a-source",
                family="language",
                filename="a.jsonl",
                sha256=a_sha,
                records=a_records,
            ),
            _source_entry(
                source_id="b-source",
                family="reasoning",
                filename="b.jsonl",
                sha256=b_sha,
                records=b_records,
            ),
        ],
    )

    with pytest.raises(ValueError, match="crosses task families"):
        build_r2d9_campaign(
            definition_path=definition,
            output_dir=tmp_path / "campaign",
        )


def test_r2d9_rejects_too_fine_shards_for_holdout_volume(
    tmp_path: Path,
) -> None:
    sha, records = _write_source(
        tmp_path / "source.jsonl",
        [f"row-{index}" for index in range(10)],
    )
    definition = _definition(
        tmp_path / "definition.json",
        [
            _source_entry(
                source_id="source",
                family="language",
                filename="source.jsonl",
                sha256=sha,
                records=records,
            ),
        ],
        target=1,
    )

    with pytest.raises(RuntimeError, match="holdout volume"):
        build_r2d9_campaign(
            definition_path=definition,
            output_dir=tmp_path / "campaign",
        )


def test_r2d9_rejects_source_identity_mismatch(
    tmp_path: Path,
) -> None:
    _, records = _write_source(
        tmp_path / "source.jsonl",
        [f"row-{index}" for index in range(10)],
    )
    definition = _definition(
        tmp_path / "definition.json",
        [
            _source_entry(
                source_id="source",
                family="language",
                filename="source.jsonl",
                sha256="0" * 64,
                records=records,
            ),
        ],
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_r2d9_campaign(
            definition_path=definition,
            output_dir=tmp_path / "campaign",
        )


def test_r2d9_verify_rejects_seal_or_definition_tamper(
    tmp_path: Path,
) -> None:
    sha, records = _write_source(
        tmp_path / "source.jsonl",
        [f"row-{index}" for index in range(10)],
    )
    definition = _definition(
        tmp_path / "definition.json",
        [
            _source_entry(
                source_id="source",
                family="language",
                filename="source.jsonl",
                sha256=sha,
                records=records,
            ),
        ],
    )
    campaign_dir = tmp_path / "campaign"
    campaign = build_r2d9_campaign(
        definition_path=definition,
        output_dir=campaign_dir,
    )

    first = campaign["seals"][0]
    seal_training = (
        campaign_dir
        / first["corpus_path"]
        / "training.jsonl"
    )
    with seal_training.open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError):
        verify_r2d9_campaign(campaign_dir)

    # Restore by building a second clean campaign and then tamper its D6 handoff.
    clean = tmp_path / "clean"
    build_r2d9_campaign(
        definition_path=definition,
        output_dir=clean,
    )
    d6_path = clean / "r2d6-definition.json"
    d6 = json.loads(d6_path.read_text(encoding="utf-8"))
    d6["corpora"][0]["task_families"] = ["reasoning"]
    d6_path.write_text(
        json.dumps(d6, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="D6 definition"):
        verify_r2d9_campaign(clean)


def test_r2d9_definition_order_does_not_change_identity(
    tmp_path: Path,
) -> None:
    a_sha, a_records = _write_source(
        tmp_path / "a.jsonl",
        [f"a-{index}" for index in range(10)],
    )
    b_sha, b_records = _write_source(
        tmp_path / "b.jsonl",
        [f"b-{index}" for index in range(10)],
    )
    a = _source_entry(
        source_id="a-source",
        family="language",
        filename="a.jsonl",
        sha256=a_sha,
        records=a_records,
    )
    b = _source_entry(
        source_id="b-source",
        family="reasoning",
        filename="b.jsonl",
        sha256=b_sha,
        records=b_records,
    )
    one = _definition(
        tmp_path / "one.json",
        [a, b],
    )
    two = _definition(
        tmp_path / "two.json",
        [b, a],
    )

    assert (
        load_r2d9_definition(one).fingerprint()
        == load_r2d9_definition(two).fingerprint()
    )
