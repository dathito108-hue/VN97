from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import vn97.r2.production_frozen_campaign as d13
from vn97.r2.production_acquisition import build_r2d9_campaign
from vn97.r2.production_campaign import build_r2d5_preflight_package
from vn97.r2.production_contract import R2ProductionTrainingRecipe
from vn97.r2.production_corpus_scale import (
    build_r2d6_corpus_index,
    load_r2d6_definition,
)
from vn97.r2.production_curriculum import (
    R2D8CurriculumDefinition,
    compile_r2d8_plan,
)
from vn97.r2.production_frozen_campaign import (
    R2D13_READY_SCHEMA,
    assert_r2d13_preflight_allowed,
    build_r2d13_campaign,
    seal_r2d13_ready,
    verify_r2d13_campaign,
    verify_r2d13_ready,
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
from vn97.r2.production_training import R2ProductionTrainerConfig
from vn97.r2.production_virtual_corpus import (
    R2D12_MOUNT_DEFINITION_SCHEMA,
    build_r2d12_view,
    verify_r2d12_view,
)
from vn97.tokenizer import VN97TokenizerPackage


COMMIT = "a" * 40


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_jsonl(
    path: Path,
    rows: list[dict[str, object]],
) -> tuple[str, int, int]:
    data = b"".join(
        _canonical_json(row) + b"\n"
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
                        "with deterministic context"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        f"{prefix} answer {index} "
                        "with deterministic supervision"
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


def _campaign_and_d6(
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


def _frozen_inputs(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    tokenizer = tmp_path / "tokenizer.vn97tk1"
    tokenizer.write_bytes(VN97TokenizerPackage().to_bytes())

    registry_def = tmp_path / "registry-definition.json"
    registry_def.write_text(
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
        definition_path=registry_def,
        tokenizer_path=tokenizer,
        registry_dir=registry,
    )

    packages: dict[str, Path] = {}
    campaigns: dict[str, Path] = {}
    for family in ("language", "reasoning"):
        pack = _build_pack(
            tmp_path / family,
            source_id=f"{family}-source",
            family=family,
            prefix=family,
        )
        admitted = admit_r2d11_pack(
            registry_dir=registry,
            pack_dir=pack,
        )
        campaign, d6 = _campaign_and_d6(
            tmp_path / f"{family}-products",
            pack=pack,
            tokenizer=tokenizer,
        )
        attach_r2d11_campaign(
            registry_dir=registry,
            pack_dir=pack,
            campaign_dir=campaign,
            d6_package_dir=d6,
        )
        pack_id = str(admitted["event"]["pack_id"])
        packages[pack_id] = d6
        campaigns[family] = campaign

    mount_def = tmp_path / "mounts.json"
    mount_def.write_text(
        json.dumps(
            {
                "schema": R2D12_MOUNT_DEFINITION_SCHEMA,
                "packages": {
                    pack_id: package.resolve(strict=True)
                    .relative_to(tmp_path.resolve(strict=True))
                    .as_posix()
                    for pack_id, package in sorted(packages.items())
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    view = tmp_path / "view"
    build_r2d12_view(
        registry_dir=registry,
        generation=4,
        workspace_root=tmp_path,
        mount_definition_path=mount_def,
        output_dir=view,
    )

    verified = verify_r2d12_view(
        view,
        workspace_root=tmp_path,
    )
    index = verified["index"]
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
            seed=9710,
            family_weights=(
                ("language", 0.5),
                ("reasoning", 0.5),
            ),
            primary_family_by_manifest=assignments,
        ),
    )
    plan_path = tmp_path / "curriculum.json"
    plan_path.write_text(
        json.dumps(
            plan,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    recipe = R2ProductionTrainingRecipe(
        sequence_length=64,
        micro_batch_size=1,
        gradient_accumulation_steps=2,
        precision="fp32",
        activation_checkpointing=True,
        memory_efficient_scan=True,
        optimizer_state_offload=True,
        full_parameter_training=True,
        quantization_used=False,
    )
    preflight = tmp_path / "preflight"
    build_r2d5_preflight_package(
        corpus_dir=campaigns["language"] / "seals" / "language-00000",
        tokenizer_path=tokenizer,
        repository_commit=COMMIT,
        output_dir=preflight,
        task_families=("language", "reasoning"),
        recipe=recipe,
        safety_fraction=0.90,
    )
    return view, plan_path, preflight


def test_r2d13_builds_portable_underfloor_campaign(
    tmp_path: Path,
) -> None:
    view, plan, preflight = _frozen_inputs(tmp_path)
    output = tmp_path / "r2d13"
    campaign = build_r2d13_campaign(
        view_dir=view,
        workspace_root=tmp_path,
        curriculum_plan_path=plan,
        preflight_package_dir=preflight,
        repository_commit=COMMIT,
        output_dir=output,
        max_run_seconds=1234.0,
    )
    verified = verify_r2d13_campaign(
        output,
        workspace_root=tmp_path,
    )

    assert verified["campaign"] == campaign
    assert campaign["scale_floor_passed"] is False
    assert campaign["gpu_preflight_allowed"] is False
    assert campaign["training_launch_allowed"] is False
    assert campaign["max_run_seconds"] == 1234.0
    assert not (output / "view" / "shards").exists()
    assert (output / "preflight" / "training.jsonl").is_file()

    preflight_script = (
        output / "run_t4_preflight.sh"
    ).read_text(encoding="utf-8")
    assert COMMIT in preflight_script
    assert "gate-preflight" in preflight_script
    assert "preflight/run_t4_preflight.sh" in preflight_script

    train_script = (
        output / "run_t4_train.sh"
    ).read_text(encoding="utf-8")
    assert COMMIT in train_script
    assert "verify-ready" in train_script
    assert "--max-run-seconds 1234.000000" in train_script
    assert "--virtual-view" in train_script
    assert "release.jsonl" not in train_script


def test_r2d13_refuses_gpu_preflight_below_scale_floor(
    tmp_path: Path,
) -> None:
    view, plan, preflight = _frozen_inputs(tmp_path)
    output = tmp_path / "r2d13"
    build_r2d13_campaign(
        view_dir=view,
        workspace_root=tmp_path,
        curriculum_plan_path=plan,
        preflight_package_dir=preflight,
        repository_commit=COMMIT,
        output_dir=output,
    )
    with pytest.raises(RuntimeError, match="below the stage scale floor"):
        assert_r2d13_preflight_allowed(
            output,
            workspace_root=tmp_path,
        )


def test_r2d13_rejects_script_tamper(
    tmp_path: Path,
) -> None:
    view, plan, preflight = _frozen_inputs(tmp_path)
    output = tmp_path / "r2d13"
    build_r2d13_campaign(
        view_dir=view,
        workspace_root=tmp_path,
        curriculum_plan_path=plan,
        preflight_package_dir=preflight,
        repository_commit=COMMIT,
        output_dir=output,
    )
    with (output / "run_t4_train.sh").open("a", encoding="utf-8") as handle:
        handle.write("\necho tampered\n")

    with pytest.raises(ValueError, match="script hash mismatch"):
        verify_r2d13_campaign(
            output,
            workspace_root=tmp_path,
        )


def test_r2d13_rejects_repository_commit_mismatch(
    tmp_path: Path,
) -> None:
    view, plan, preflight = _frozen_inputs(tmp_path)
    with pytest.raises(ValueError, match="repository commit mismatch"):
        build_r2d13_campaign(
            view_dir=view,
            workspace_root=tmp_path,
            curriculum_plan_path=plan,
            preflight_package_dir=preflight,
            repository_commit="b" * 40,
            output_dir=tmp_path / "r2d13",
        )


def test_r2d13_rejects_preflight_family_mismatch(
    tmp_path: Path,
) -> None:
    view, plan, _ = _frozen_inputs(tmp_path)
    verified = verify_r2d12_view(
        view,
        workspace_root=tmp_path,
    )
    tokenizer = verified["tokenizer_path"]

    # Build a valid D5 package whose declared preflight family set does not
    # match the two-family D8 dense-pretrain curriculum.
    language_manifest = next(
        item
        for item in verified["index"]["source_manifests"]
        if item["task_families"] == ["language"]
    )
    pack_id = str(language_manifest["batch_pack_id"])
    # The mounted D6 manifest points back to the D9 seal, but D5 requires the
    # original VN97CORPUS1 directory. Reuse the known test source campaign.
    language_campaign = tmp_path / "language-products" / "campaign"
    mismatch = tmp_path / "preflight-mismatch"
    build_r2d5_preflight_package(
        corpus_dir=language_campaign / "seals" / "language-00000",
        tokenizer_path=tokenizer,
        repository_commit=COMMIT,
        output_dir=mismatch,
        task_families=("language",),
        recipe=R2ProductionTrainingRecipe(
            sequence_length=64,
            micro_batch_size=1,
            gradient_accumulation_steps=2,
            precision="fp32",
            activation_checkpointing=True,
            memory_efficient_scan=True,
            optimizer_state_offload=True,
            full_parameter_training=True,
            quantization_used=False,
        ),
        safety_fraction=0.90,
    )
    assert pack_id
    with pytest.raises(ValueError, match="task families differ"):
        build_r2d13_campaign(
            view_dir=view,
            workspace_root=tmp_path,
            curriculum_plan_path=plan,
            preflight_package_dir=mismatch,
            repository_commit=COMMIT,
            output_dir=tmp_path / "r2d13",
        )


def _measured_receipt(
    path: Path,
    *,
    campaign_id: str,
    architecture: str,
    recipe: R2ProductionTrainingRecipe,
) -> dict[str, object]:
    body = {
        "architecture_fingerprint": architecture,
        "campaign_id": campaign_id,
        "device_name": "Tesla T4",
        "free_device_bytes_before": 14_000_000_000,
        "micro_batch_size": recipe.micro_batch_size,
        "passed": True,
        "peak_allocated_bytes": 8_000_000_000,
        "peak_reserved_bytes": 9_000_000_000,
        "preflight_bundle_sha256": "c" * 64,
        "reason": "within_measured_safety_budget",
        "recipe_fingerprint": recipe.fingerprint(),
        "safety_fraction": 0.90,
        "schema": "VN97R2D5PREFLIGHT1",
        "sequence_length": recipe.sequence_length,
        "total_device_bytes": 16_000_000_000,
        "training_allowed": False,
    }
    receipt = dict(body)
    receipt["receipt_id"] = hashlib.sha256(
        b"VN97R2D5PREFLIGHT1\0" + _canonical_json(body)
    ).hexdigest()
    path.write_bytes(_canonical_json(receipt) + b"\n")
    return receipt


def test_r2d13_ready_receipt_binds_exact_measured_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = R2ProductionTrainingRecipe(
        sequence_length=128,
        micro_batch_size=1,
        gradient_accumulation_steps=16,
        precision="fp16",
        activation_checkpointing=True,
        memory_efficient_scan=True,
        optimizer_state_offload=True,
        full_parameter_training=True,
        quantization_used=False,
    )
    campaign = {
        "campaign_id": "1" * 64,
        "preflight_campaign_id": "2" * 64,
        "architecture_fingerprint": "3" * 64,
        "repository_commit": COMMIT,
        "virtual_view_id": "4" * 64,
        "projected_index_id": "5" * 64,
        "curriculum_plan_id": "6" * 64,
        "recipe_fingerprint": recipe.fingerprint(),
        "scale_floor_passed": True,
        "gpu_preflight_allowed": True,
    }
    trainer = R2ProductionTrainerConfig(
        epochs=1,
        seed=9710,
    )
    monkeypatch.setattr(
        d13,
        "assert_r2d13_preflight_allowed",
        lambda package_dir, workspace_root: {
            "campaign": campaign,
            "recipe": recipe,
            "trainer": trainer,
        },
    )

    preflight = tmp_path / "preflight-receipt.json"
    receipt = _measured_receipt(
        preflight,
        campaign_id=campaign["preflight_campaign_id"],
        architecture=campaign["architecture_fingerprint"],
        recipe=recipe,
    )
    ready_path = tmp_path / "ready.json"
    ready = seal_r2d13_ready(
        package_dir=tmp_path,
        workspace_root=tmp_path,
        preflight_receipt_path=preflight,
        output_path=ready_path,
    )
    assert ready["schema"] == R2D13_READY_SCHEMA
    assert ready["training_allowed"] is True
    assert ready["preflight_receipt_id"] == receipt["receipt_id"]
    assert ready["device_name"] == "Tesla T4"

    verified = verify_r2d13_ready(
        package_dir=tmp_path,
        workspace_root=tmp_path,
        ready_receipt_path=ready_path,
        preflight_receipt_path=preflight,
    )
    assert verified["ready"] == ready

    foreign = tmp_path / "foreign.json"
    _measured_receipt(
        foreign,
        campaign_id="7" * 64,
        architecture=campaign["architecture_fingerprint"],
        recipe=recipe,
    )
    with pytest.raises(ValueError, match="another D5 campaign"):
        verify_r2d13_ready(
            package_dir=tmp_path,
            workspace_root=tmp_path,
            ready_receipt_path=ready_path,
            preflight_receipt_path=foreign,
        )
