from __future__ import annotations

import json
from pathlib import Path

import pytest

from vn97.r2.production_contract import (
    R2ProductionCorpusManifest,
    R2ProductionShard,
    R2ProductionTrainingRecipe,
)
from vn97.r2.production_launcher import (
    build_production_manifest,
    load_preflight_bundle,
    save_preflight_bundle,
)
from vn97.r2.production_training import R2MeasuredMemoryEvidence


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def _recipe(sequence_length: int = 128) -> R2ProductionTrainingRecipe:
    return R2ProductionTrainingRecipe(
        sequence_length=sequence_length,
        micro_batch_size=1,
        gradient_accumulation_steps=16,
        precision="fp16",
        activation_checkpointing=True,
        memory_efficient_scan=True,
        optimizer_state_offload=True,
        full_parameter_training=True,
        quantization_used=False,
    )


def _manifest() -> R2ProductionCorpusManifest:
    return R2ProductionCorpusManifest(
        stage="dense_pretrain",
        tokenizer_sha256="a" * 64,
        shards=(
            R2ProductionShard(
                split="train",
                sha256="b" * 64,
                bytes=10,
                records=2,
                task_families=("language",),
            ),
            R2ProductionShard(
                split="validation",
                sha256="c" * 64,
                bytes=5,
                records=1,
                task_families=("language",),
            ),
        ),
    )


def _evidence(recipe: R2ProductionTrainingRecipe) -> R2MeasuredMemoryEvidence:
    return R2MeasuredMemoryEvidence(
        architecture_fingerprint="d" * 64,
        recipe_fingerprint=recipe.fingerprint(),
        sequence_length=recipe.sequence_length,
        micro_batch_size=recipe.micro_batch_size,
        device_name="test-gpu",
        free_device_bytes_before=16 * 1024**3,
        total_device_bytes=16 * 1024**3,
        peak_allocated_bytes=8 * 1024**3,
        peak_reserved_bytes=9 * 1024**3,
        safety_fraction=0.90,
        passed=True,
        reason="within_measured_safety_budget",
    )


def test_build_manifest_binds_exact_input_files(tmp_path: Path) -> None:
    tokenizer = _write(tmp_path / "tokenizer.vn97tk1", b"tokenizer")
    train = _write(tmp_path / "train.jsonl", b'{"messages":[]}\n')
    validation = _write(
        tmp_path / "validation.jsonl",
        b'{"messages":[1]}\n',
    )

    manifest = build_production_manifest(
        stage="dense_pretrain",
        tokenizer_path=tokenizer,
        train_path=train,
        validation_path=validation,
        train_records=7,
        validation_records=3,
        task_families=("reasoning", "language", "language"),
        parent_checkpoint_sha256=None,
    )

    assert manifest.shards[0].records == 7
    assert manifest.shards[1].records == 3
    assert manifest.shards[0].task_families == ("language", "reasoning")
    assert manifest.shards[0].sha256 != manifest.shards[1].sha256
    assert len(manifest.identity()) == 64


def test_preflight_bundle_roundtrip_is_no_promotion(tmp_path: Path) -> None:
    manifest = _manifest()
    recipe = _recipe()
    evidence = _evidence(recipe)
    path = tmp_path / "preflight.json"

    save_preflight_bundle(
        path,
        manifest=manifest,
        recipe=recipe,
        evidence=evidence,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["promotion_allowed"] is False

    loaded = load_preflight_bundle(
        path,
        manifest=manifest,
        recipe=recipe,
    )
    assert loaded == evidence


def test_preflight_bundle_rejects_recipe_reuse(tmp_path: Path) -> None:
    manifest = _manifest()
    recipe = _recipe()
    path = tmp_path / "preflight.json"
    save_preflight_bundle(
        path,
        manifest=manifest,
        recipe=recipe,
        evidence=_evidence(recipe),
    )

    with pytest.raises(RuntimeError, match="recipe fingerprint mismatch"):
        load_preflight_bundle(
            path,
            manifest=manifest,
            recipe=_recipe(sequence_length=256),
        )


def test_preflight_bundle_rejects_manifest_reuse(tmp_path: Path) -> None:
    manifest = _manifest()
    recipe = _recipe()
    path = tmp_path / "preflight.json"
    save_preflight_bundle(
        path,
        manifest=manifest,
        recipe=recipe,
        evidence=_evidence(recipe),
    )

    changed = R2ProductionCorpusManifest(
        stage="dense_pretrain",
        tokenizer_sha256="a" * 64,
        shards=(
            R2ProductionShard(
                split="train",
                sha256="e" * 64,
                bytes=10,
                records=2,
                task_families=("language",),
            ),
            manifest.shards[1],
        ),
    )
    with pytest.raises(RuntimeError, match="manifest identity mismatch"):
        load_preflight_bundle(
            path,
            manifest=changed,
            recipe=recipe,
        )


def test_preflight_bundle_rejects_promotion_flag_tamper(tmp_path: Path) -> None:
    manifest = _manifest()
    recipe = _recipe()
    path = tmp_path / "preflight.json"
    save_preflight_bundle(
        path,
        manifest=manifest,
        recipe=recipe,
        evidence=_evidence(recipe),
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["promotion_allowed"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="no-promotion"):
        load_preflight_bundle(
            path,
            manifest=manifest,
            recipe=recipe,
        )
