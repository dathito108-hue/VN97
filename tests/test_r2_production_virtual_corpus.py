from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vn97.r2.data_bridge import load_vn97tk1
from vn97.r2.production_acquisition import build_r2d9_campaign
from vn97.r2.production_corpus_scale import (
    build_r2d6_corpus_index,
    load_r2d6_definition,
)
from vn97.r2.production_curriculum import (
    R2D8CurriculumDefinition,
    compile_r2d8_plan,
)
from vn97.r2.production_registry import (
    R2D11_DEFINITION_SCHEMA,
    admit_r2d11_pack,
    attach_r2d11_campaign,
    init_r2d11_registry,
)
from vn97.r2.production_source_adapter import (
    R2D10_LOCK_SCHEMA,
    build_r2d10_pack,
)
from vn97.r2.production_streaming import (
    R2StreamingCursor,
    iter_epoch_training_windows,
)
from vn97.r2.production_virtual_corpus import (
    R2D12_MOUNT_DEFINITION_SCHEMA,
    build_r2d12_view,
    project_r2d12_index,
    verify_r2d12_view,
)
from vn97.tokenizer import VN97TokenizerPackage


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


def _rows(prefix: str, count: int = 8) -> list[dict[str, object]]:
    return [
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"{prefix} question {index} "
                        "with enough repeated context for windows"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        f"{prefix} answer {index} "
                        "with a deterministic supervised response"
                    ),
                },
            ]
        }
        for index in range(count)
    ]


def _build_pack(
    root: Path,
    *,
    source_id: str,
    family: str,
    prefix: str,
) -> Path:
    root.mkdir()
    sha, records, size = _write_jsonl(
        root / "raw.jsonl",
        _rows(prefix),
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
                "max_bytes": size,
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


def _build_campaign_d6(
    root: Path,
    *,
    pack: Path,
    tokenizer: Path,
) -> tuple[Path, Path]:
    root.mkdir()
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


def _registry(tmp_path: Path) -> tuple[Path, Path]:
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


def _two_batch_frozen_registry(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, Path], int]:
    registry, tokenizer = _registry(tmp_path)

    language_pack = _build_pack(
        tmp_path / "language",
        source_id="language-source",
        family="language",
        prefix="language",
    )
    language_admit = admit_r2d11_pack(
        registry_dir=registry,
        pack_dir=language_pack,
    )
    language_campaign, language_d6 = _build_campaign_d6(
        tmp_path / "language-products",
        pack=language_pack,
        tokenizer=tokenizer,
    )
    language_attach = attach_r2d11_campaign(
        registry_dir=registry,
        pack_dir=language_pack,
        campaign_dir=language_campaign,
        d6_package_dir=language_d6,
    )
    assert language_attach["generation"] == 2

    reasoning_pack = _build_pack(
        tmp_path / "reasoning",
        source_id="reasoning-source",
        family="reasoning",
        prefix="reasoning",
    )
    reasoning_admit = admit_r2d11_pack(
        registry_dir=registry,
        pack_dir=reasoning_pack,
    )
    reasoning_campaign, reasoning_d6 = _build_campaign_d6(
        tmp_path / "reasoning-products",
        pack=reasoning_pack,
        tokenizer=tokenizer,
    )
    reasoning_attach = attach_r2d11_campaign(
        registry_dir=registry,
        pack_dir=reasoning_pack,
        campaign_dir=reasoning_campaign,
        d6_package_dir=reasoning_d6,
    )
    assert reasoning_attach["generation"] == 4

    return (
        registry,
        tokenizer,
        {
            str(language_admit["event"]["pack_id"]): language_d6,
            str(reasoning_admit["event"]["pack_id"]): reasoning_d6,
        },
        4,
    )


def _mount_definition(
    path: Path,
    *,
    workspace: Path,
    packages: dict[str, Path],
) -> Path:
    payload = {
        "schema": R2D12_MOUNT_DEFINITION_SCHEMA,
        "packages": {
            pack_id: package.resolve(strict=True)
            .relative_to(workspace.resolve(strict=True))
            .as_posix()
            for pack_id, package in sorted(packages.items())
        },
    }
    path.write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )
    return path


def test_r2d12_freeze_combines_batches_without_copying_shards(
    tmp_path: Path,
) -> None:
    registry, _, packages, generation = _two_batch_frozen_registry(
        tmp_path
    )
    mount_def = _mount_definition(
        tmp_path / "mounts.json",
        workspace=tmp_path,
        packages=packages,
    )
    view_dir = tmp_path / "view"
    view = build_r2d12_view(
        registry_dir=registry,
        generation=generation,
        workspace_root=tmp_path,
        mount_definition_path=mount_def,
        output_dir=view_dir,
    )

    assert view["registry_generation"] == 4
    assert len(view["attached_batches"]) == 2
    assert not (view_dir / "shards").exists()
    assert not (view_dir / "manifests").exists()

    verified = verify_r2d12_view(
        view_dir,
        workspace_root=tmp_path,
    )
    index = verified["index"]
    assert index["schema"] == "VN97R2D12INDEX1"
    assert len(index["source_manifests"]) == 2
    assert len(index["shards"]) == 6
    assert {
        item["task_families"][0]
        for item in index["source_manifests"]
    } == {"language", "reasoning"}
    assert set(verified["shard_roots"]) == set(packages)
    assert index["scale"]["tokens_per_parameter"] > 0.0


def test_r2d12_view_identity_is_mount_location_independent(
    tmp_path: Path,
) -> None:
    registry, _, packages, generation = _two_batch_frozen_registry(
        tmp_path
    )
    first_mounts = _mount_definition(
        tmp_path / "mounts-a.json",
        workspace=tmp_path,
        packages=packages,
    )
    first = build_r2d12_view(
        registry_dir=registry,
        generation=generation,
        workspace_root=tmp_path,
        mount_definition_path=first_mounts,
        output_dir=tmp_path / "view-a",
    )

    relocated: dict[str, Path] = {}
    for index, (pack_id, package) in enumerate(
        sorted(packages.items())
    ):
        target = tmp_path / f"relocated-{index}"
        package.rename(target)
        relocated[pack_id] = target

    second_mounts = _mount_definition(
        tmp_path / "mounts-b.json",
        workspace=tmp_path,
        packages=relocated,
    )
    second = build_r2d12_view(
        registry_dir=registry,
        generation=generation,
        workspace_root=tmp_path,
        mount_definition_path=second_mounts,
        output_dir=tmp_path / "view-b",
    )

    assert first["view_id"] == second["view_id"]
    assert (
        first["virtual_index"]["index_id"]
        == second["virtual_index"]["index_id"]
    )


def test_r2d12_projection_and_d8_plan_are_deterministic(
    tmp_path: Path,
) -> None:
    registry, _, packages, generation = _two_batch_frozen_registry(
        tmp_path
    )
    mount_def = _mount_definition(
        tmp_path / "mounts.json",
        workspace=tmp_path,
        packages=packages,
    )
    view_dir = tmp_path / "view"
    build_r2d12_view(
        registry_dir=registry,
        generation=generation,
        workspace_root=tmp_path,
        mount_definition_path=mount_def,
        output_dir=view_dir,
    )
    verified = verify_r2d12_view(
        view_dir,
        workspace_root=tmp_path,
    )
    full = verified["index"]

    language = project_r2d12_index(
        full,
        families=("language",),
    )
    reasoning = project_r2d12_index(
        full,
        families=("reasoning",),
    )
    assert language["index_id"] != reasoning["index_id"]
    assert {
        item["task_families"][0]
        for item in language["source_manifests"]
    } == {"language"}
    assert {
        item["task_families"][0]
        for item in reasoning["source_manifests"]
    } == {"reasoning"}

    assignments = tuple(
        sorted(
            (
                str(item["manifest_id"]),
                str(item["task_families"][0]),
            )
            for item in full["source_manifests"]
        )
    )
    definition = R2D8CurriculumDefinition(
        stage="dense_pretrain",
        epochs=2,
        seed=9710,
        family_weights=(
            ("language", 0.5),
            ("reasoning", 0.5),
        ),
        primary_family_by_manifest=assignments,
    )
    plan_a = compile_r2d8_plan(full, definition)
    plan_b = compile_r2d8_plan(full, definition)
    assert plan_a == plan_b


def test_r2d7_streams_virtual_shards_across_batches(
    tmp_path: Path,
) -> None:
    registry, _, packages, generation = _two_batch_frozen_registry(
        tmp_path
    )
    mount_def = _mount_definition(
        tmp_path / "mounts.json",
        workspace=tmp_path,
        packages=packages,
    )
    view_dir = tmp_path / "view"
    build_r2d12_view(
        registry_dir=registry,
        generation=generation,
        workspace_root=tmp_path,
        mount_definition_path=mount_def,
        output_dir=view_dir,
    )
    verified = verify_r2d12_view(
        view_dir,
        workspace_root=tmp_path,
    )
    index = verified["index"]
    tokenizer = load_vn97tk1(verified["tokenizer_path"])

    assignments = tuple(
        sorted(
            (
                str(item["manifest_id"]),
                str(item["task_families"][0]),
            )
            for item in index["source_manifests"]
        )
    )
    plan = compile_r2d8_plan(
        index,
        R2D8CurriculumDefinition(
            stage="dense_pretrain",
            epochs=1,
            seed=97,
            family_weights=(
                ("language", 0.5),
                ("reasoning", 0.5),
            ),
            primary_family_by_manifest=assignments,
        ),
    )

    rows = list(
        iter_epoch_training_windows(
            view_dir,
            index,
            tokenizer,
            cursor=R2StreamingCursor(0, 0, 0, 0),
            seed=97,
            sequence_length=64,
            curriculum_plan=plan,
            shard_roots=verified["shard_roots"],
        )
    )
    assert rows
    assert rows[-1][1] == R2StreamingCursor(1, 0, 0, 0)


def test_r2d12_rejects_wrong_pack_mount(
    tmp_path: Path,
) -> None:
    registry, _, packages, generation = _two_batch_frozen_registry(
        tmp_path
    )
    pack_ids = sorted(packages)
    wrong = {
        pack_ids[0]: packages[pack_ids[1]],
        pack_ids[1]: packages[pack_ids[0]],
    }
    mount_def = _mount_definition(
        tmp_path / "wrong-mounts.json",
        workspace=tmp_path,
        packages=wrong,
    )
    with pytest.raises(ValueError, match="does not match D11"):
        build_r2d12_view(
            registry_dir=registry,
            generation=generation,
            workspace_root=tmp_path,
            mount_definition_path=mount_def,
            output_dir=tmp_path / "view",
        )


def test_r2d12_rejects_freeze_with_pending_admission(
    tmp_path: Path,
) -> None:
    registry, _, packages, generation = _two_batch_frozen_registry(
        tmp_path
    )
    pending = _build_pack(
        tmp_path / "pending",
        source_id="pending-language",
        family="language",
        prefix="pending",
    )
    admitted = admit_r2d11_pack(
        registry_dir=registry,
        pack_dir=pending,
    )
    assert admitted["generation"] == generation + 1

    mount_def = _mount_definition(
        tmp_path / "mounts.json",
        workspace=tmp_path,
        packages=packages,
    )
    with pytest.raises(RuntimeError, match="every admitted pack"):
        build_r2d12_view(
            registry_dir=registry,
            generation=generation + 1,
            workspace_root=tmp_path,
            mount_definition_path=mount_def,
            output_dir=tmp_path / "view",
        )


def test_r2d12_detects_frozen_ledger_tamper(
    tmp_path: Path,
) -> None:
    registry, _, packages, generation = _two_batch_frozen_registry(
        tmp_path
    )
    mount_def = _mount_definition(
        tmp_path / "mounts.json",
        workspace=tmp_path,
        packages=packages,
    )
    view_dir = tmp_path / "view"
    build_r2d12_view(
        registry_dir=registry,
        generation=generation,
        workspace_root=tmp_path,
        mount_definition_path=mount_def,
        output_dir=view_dir,
    )

    ledger = view_dir / "r2d11-ledger-snapshot.json"
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    payload["generation"] += 1
    ledger.write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ledger evidence"):
        verify_r2d12_view(
            view_dir,
            workspace_root=tmp_path,
        )
