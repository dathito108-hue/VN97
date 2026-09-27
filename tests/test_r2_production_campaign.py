from __future__ import annotations

import json
from pathlib import Path

import pytest

from vn97.production_corpus import prepare_corpus
from vn97.r2.production_campaign import (
    R2D5_CORPUS_PROFILE,
    R2D5_PREFLIGHT_RECEIPT_SCHEMA,
    build_r2d5_preflight_package,
    seal_r2d5_preflight,
    verify_r2d5_corpus,
    verify_r2d5_package,
)
from vn97.r2.production_contract import R2ProductionTrainingRecipe
from vn97.tokenizer import VN97TokenizerPackage
from vn97.training import VN97ChatMessage


REPOSITORY_COMMIT = "a" * 40


def _conversation(user: str, assistant: str):
    return (
        VN97ChatMessage(role="user", content=user),
        VN97ChatMessage(role="assistant", content=assistant),
    )


def _build_corpus(root: Path) -> Path:
    rows = (
        (
            "train-source",
            "https://example.test/train",
            "MIT",
            "training",
            "chat",
            b"train-source-bytes",
            (
                _conversation("train one", "answer one"),
                _conversation("train two", "answer two"),
            ),
        ),
        (
            "validation-source",
            "https://example.test/validation",
            "Apache-2.0",
            "validation",
            "chat",
            b"validation-source-bytes",
            (
                _conversation("validation one", "held out answer"),
            ),
        ),
        (
            "release-source",
            "https://example.test/release",
            "CC-BY-4.0",
            "release",
            "chat",
            b"release-source-bytes",
            (
                _conversation("release one", "never train on this"),
            ),
        ),
    )
    manifest, prepared = prepare_corpus(
        rows,
        profile_id=R2D5_CORPUS_PROFILE,
    )
    root.mkdir()
    (root / "training.jsonl").write_bytes(prepared["training"].bytes)
    (root / "validation.jsonl").write_bytes(prepared["validation"].bytes)
    (root / "release.jsonl").write_bytes(prepared["release"].bytes)
    (root / "corpus.vn97corpus1.json").write_bytes(manifest.to_bytes())
    return root


def _tokenizer(path: Path) -> Path:
    path.write_bytes(VN97TokenizerPackage().to_bytes())
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


def test_r2d5_verifies_sealed_three_way_holdout(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path / "corpus")
    evidence = verify_r2d5_corpus(corpus)

    assert evidence.profile_id == R2D5_CORPUS_PROFILE
    assert evidence.source_count == 3
    assert evidence.split_identity["training"]["records"] == 2
    assert evidence.split_identity["validation"]["records"] == 1
    assert evidence.split_identity["release"]["records"] == 1
    assert evidence.source_licenses == (
        "Apache-2.0",
        "CC-BY-4.0",
        "MIT",
    )


def test_r2d5_build_and_verify_preflight_only_package(
    tmp_path: Path,
) -> None:
    corpus = _build_corpus(tmp_path / "corpus")
    tokenizer = _tokenizer(tmp_path / "tokenizer.vn97tk1")
    package = tmp_path / "package"

    campaign = build_r2d5_preflight_package(
        corpus_dir=corpus,
        tokenizer_path=tokenizer,
        repository_commit=REPOSITORY_COMMIT,
        output_dir=package,
        task_families=("reasoning", "language", "language"),
        recipe=_recipe(),
        safety_fraction=0.90,
    )

    assert campaign["training_allowed"] is False
    assert campaign["release_used_for_training"] is False
    assert campaign["purpose"] == "measured_cuda_preflight"
    assert campaign["task_families"] == ["language", "reasoning"]
    assert 900_000_000 <= campaign["model_parameter_count"] <= 1_300_000_000

    verified = verify_r2d5_package(package)
    assert verified["campaign_id"] == campaign["campaign_id"]

    script = (package / "run_t4_preflight.sh").read_text(
        encoding="utf-8"
    )
    assert REPOSITORY_COMMIT in script
    assert "--mode preflight" in script
    assert "--stage dense_pretrain" in script
    assert "release.jsonl" not in script


def test_r2d5_package_rejects_script_tamper(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path / "corpus")
    tokenizer = _tokenizer(tmp_path / "tokenizer.vn97tk1")
    package = tmp_path / "package"
    build_r2d5_preflight_package(
        corpus_dir=corpus,
        tokenizer_path=tokenizer,
        repository_commit=REPOSITORY_COMMIT,
        output_dir=package,
        task_families=("language",),
        recipe=_recipe(),
        safety_fraction=0.90,
    )

    script = package / "run_t4_preflight.sh"
    script.write_text(
        script.read_text(encoding="utf-8") + "\necho tampered\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="script hash mismatch"):
        verify_r2d5_package(package)


def test_r2d5_package_rejects_release_tamper(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path / "corpus")
    tokenizer = _tokenizer(tmp_path / "tokenizer.vn97tk1")
    package = tmp_path / "package"
    build_r2d5_preflight_package(
        corpus_dir=corpus,
        tokenizer_path=tokenizer,
        repository_commit=REPOSITORY_COMMIT,
        output_dir=package,
        task_families=("language",),
        recipe=_recipe(),
        safety_fraction=0.90,
    )

    with (package / "release.jsonl").open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError, match="release package hash mismatch"):
        verify_r2d5_package(package)


def test_r2d5_seals_matching_passed_preflight(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path / "corpus")
    tokenizer = _tokenizer(tmp_path / "tokenizer.vn97tk1")
    package = tmp_path / "package"
    campaign = build_r2d5_preflight_package(
        corpus_dir=corpus,
        tokenizer_path=tokenizer,
        repository_commit=REPOSITORY_COMMIT,
        output_dir=package,
        task_families=("language",),
        recipe=_recipe(),
        safety_fraction=0.90,
    )

    bundle = {
        "schema": "VN97R2D4PREFLIGHT1",
        "manifest_identity": campaign["d4_manifest_identity"],
        "recipe_fingerprint": campaign["recipe_fingerprint"],
        "promotion_allowed": False,
        "memory_evidence": {
            "schema": "VN97R2MEMPROBE1",
            "architecture_fingerprint": campaign[
                "architecture_fingerprint"
            ],
            "recipe_fingerprint": campaign["recipe_fingerprint"],
            "sequence_length": 128,
            "micro_batch_size": 1,
            "device_name": "Tesla T4",
            "free_device_bytes_before": 15_000_000_000,
            "total_device_bytes": 16_000_000_000,
            "peak_allocated_bytes": 9_000_000_000,
            "peak_reserved_bytes": 10_000_000_000,
            "safety_fraction": 0.90,
            "passed": True,
            "reason": "within_measured_safety_budget",
        },
    }
    bundle_path = tmp_path / "r2d4-preflight.json"
    bundle_path.write_text(
        json.dumps(bundle, sort_keys=True),
        encoding="utf-8",
    )
    receipt_path = tmp_path / "r2d5-preflight-receipt.json"

    receipt = seal_r2d5_preflight(
        package_dir=package,
        preflight_bundle_path=bundle_path,
        output_path=receipt_path,
    )

    assert receipt["schema"] == R2D5_PREFLIGHT_RECEIPT_SCHEMA
    assert receipt["campaign_id"] == campaign["campaign_id"]
    assert receipt["device_name"] == "Tesla T4"
    assert receipt["training_allowed"] is False
    assert receipt_path.is_file()


def test_r2d5_rejects_failed_or_foreign_preflight(tmp_path: Path) -> None:
    corpus = _build_corpus(tmp_path / "corpus")
    tokenizer = _tokenizer(tmp_path / "tokenizer.vn97tk1")
    package = tmp_path / "package"
    campaign = build_r2d5_preflight_package(
        corpus_dir=corpus,
        tokenizer_path=tokenizer,
        repository_commit=REPOSITORY_COMMIT,
        output_dir=package,
        task_families=("language",),
        recipe=_recipe(),
        safety_fraction=0.90,
    )

    bundle = {
        "schema": "VN97R2D4PREFLIGHT1",
        "manifest_identity": campaign["d4_manifest_identity"],
        "recipe_fingerprint": "b" * 64,
        "promotion_allowed": False,
        "memory_evidence": {
            "passed": False,
        },
    }
    bundle_path = tmp_path / "foreign.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    with pytest.raises(ValueError, match="preflight recipe mismatch"):
        seal_r2d5_preflight(
            package_dir=package,
            preflight_bundle_path=bundle_path,
            output_path=tmp_path / "receipt.json",
        )
