from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from vn97.production_corpus import prepare_corpus
from vn97.r2.config import r2_smoke_config
from vn97.r2.data_bridge import load_vn97tk1
from vn97.r2.model import VN97R2Model
from vn97.r2.production_contract import R2ProductionTrainingRecipe
from vn97.r2.production_curriculum import (
    R2D8CurriculumDefinition,
    compile_r2d8_plan,
)
from vn97.r2.production_corpus_scale import (
    R2D6CorpusInput,
    build_r2d6_corpus_index,
    verify_r2d6_corpus_index,
)
from vn97.r2.production_streaming import (
    R2StreamingCursor,
    iter_epoch_training_windows,
    load_r2d5_memory_receipt,
    production_manifest_from_r2d6,
    train_streaming_production_stage,
)
from vn97.r2.production_training import R2ProductionTrainerConfig
from vn97.tokenizer import VN97TokenizerPackage
from vn97.training import VN97ChatMessage


def _conversation(tag: str, repeat: int = 1):
    return (
        VN97ChatMessage(
            role="user",
            content=(f"{tag} user question " * repeat).strip(),
        ),
        VN97ChatMessage(
            role="assistant",
            content=(f"{tag} assistant answer " * repeat).strip(),
        ),
    )


def _build_corpus(root: Path) -> Path:
    rows = (
        (
            "train-source",
            "https://example.test/train",
            "MIT",
            "training",
            "chat",
            b"train-raw",
            (
                _conversation("alpha", 8),
                _conversation("beta", 7),
                _conversation("gamma", 6),
            ),
        ),
        (
            "validation-source",
            "https://example.test/validation",
            "Apache-2.0",
            "validation",
            "chat",
            b"validation-raw",
            (
                _conversation("validation", 3),
            ),
        ),
        (
            "release-source",
            "https://example.test/release",
            "CC-BY-4.0",
            "release",
            "chat",
            b"release-raw",
            (
                _conversation("release", 3),
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


def _build_d6_package(tmp_path: Path) -> Path:
    corpus = _build_corpus(tmp_path / "corpus")
    tokenizer = tmp_path / "tokenizer.vn97tk1"
    tokenizer.write_bytes(VN97TokenizerPackage().to_bytes())
    package = tmp_path / "d6"
    build_r2d6_corpus_index(
        corpus_inputs=(
            R2D6CorpusInput(
                path=corpus,
                task_families=("language", "reasoning"),
            ),
        ),
        tokenizer_path=tokenizer,
        output_dir=package,
        sequence_length=64,
        minimum_tokens_per_parameter=1e-12,
        target_tokens_per_parameter=1e-12,
    )
    return package


def _recipe() -> R2ProductionTrainingRecipe:
    return R2ProductionTrainingRecipe(
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


def test_stream_cursor_replays_exact_remaining_windows(
    tmp_path: Path,
) -> None:
    package = _build_d6_package(tmp_path)
    index = verify_r2d6_corpus_index(package)
    tokenizer = load_vn97tk1(package / "tokenizer.vn97tk1")

    all_rows = list(
        iter_epoch_training_windows(
            package,
            index,
            tokenizer,
            cursor=R2StreamingCursor(0, 0, 0, 0),
            seed=97,
            sequence_length=64,
        )
    )
    assert len(all_rows) >= 4

    resume_cursor = all_rows[1][1]
    resumed = list(
        iter_epoch_training_windows(
            package,
            index,
            tokenizer,
            cursor=resume_cursor,
            seed=97,
            sequence_length=64,
        )
    )
    expected = all_rows[2:]
    assert [row.input_ids for row, _ in resumed] == [
        row.input_ids for row, _ in expected
    ]
    assert [row.labels for row, _ in resumed] == [
        row.labels for row, _ in expected
    ]
    assert resumed[-1][1] == R2StreamingCursor(1, 0, 0, 0)


def test_stream_manifest_never_contains_release_shards(
    tmp_path: Path,
) -> None:
    package = _build_d6_package(tmp_path)
    index = verify_r2d6_corpus_index(package)
    manifest = production_manifest_from_r2d6(index)

    assert manifest.stage == "dense_pretrain"
    assert {item.split for item in manifest.shards} == {
        "train",
        "validation",
    }
    assert len(manifest.shards) == 2


def test_d7_preflight_receipt_is_identity_bound(tmp_path: Path) -> None:
    recipe = _recipe()
    architecture = "a" * 64
    body = {
        "architecture_fingerprint": architecture,
        "campaign_id": "b" * 64,
        "device_name": "Tesla T4",
        "free_device_bytes_before": 15_000_000_000,
        "micro_batch_size": 1,
        "passed": True,
        "peak_allocated_bytes": 8_000_000_000,
        "peak_reserved_bytes": 9_000_000_000,
        "preflight_bundle_sha256": "c" * 64,
        "reason": "within_measured_safety_budget",
        "recipe_fingerprint": recipe.fingerprint(),
        "safety_fraction": 0.90,
        "schema": "VN97R2D5PREFLIGHT1",
        "sequence_length": 64,
        "total_device_bytes": 16_000_000_000,
        "training_allowed": False,
    }
    receipt_id = hashlib.sha256(
        b"VN97R2D5PREFLIGHT1\0"
        + json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    payload = dict(body)
    payload["receipt_id"] = receipt_id
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    evidence = load_r2d5_memory_receipt(
        path,
        architecture_fingerprint=architecture,
        recipe=recipe,
    )
    assert evidence.device_name == "Tesla T4"
    assert evidence.passed is True
    assert evidence.sequence_length == 64

    payload["peak_reserved_bytes"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        load_r2d5_memory_receipt(
            path,
            architecture_fingerprint=architecture,
            recipe=recipe,
        )


def _tiny_model(vocab_size: int, seed: int) -> VN97R2Model:
    torch.manual_seed(seed)
    model = VN97R2Model(r2_smoke_config(vocab_size))
    model._r2_production_origin = {
        "kind": "initialization",
        "seed": seed,
    }
    return model


def test_stream_pause_resume_matches_uninterrupted_training(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package = _build_d6_package(tmp_path)
    tokenizer = load_vn97tk1(package / "tokenizer.vn97tk1")
    recipe = _recipe()
    trainer = R2ProductionTrainerConfig(
        epochs=2,
        learning_rate=1e-4,
        weight_decay=0.0,
        max_grad_norm=1.0,
        seed=97,
        checkpoint_every_optimizer_steps=1,
        require_measured_cuda_preflight=False,
    )
    index = verify_r2d6_corpus_index(package)
    manifest_id = next(
        item["source_manifest_id"]
        for item in index["shards"]
        if item["split"] == "training"
    )
    curriculum = compile_r2d8_plan(
        index,
        R2D8CurriculumDefinition(
            stage="dense_pretrain",
            epochs=trainer.epochs,
            seed=trainer.seed,
            family_weights=(("language", 1.0),),
            primary_family_by_manifest=(
                (manifest_id, "language"),
            ),
        ),
    )

    import vn97.r2.production_streaming as streaming

    monkeypatch.setattr(
        streaming,
        "_validate_stream_package",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        streaming,
        "assert_production_training_contract",
        lambda *args, **kwargs: None,
    )

    paused_model = _tiny_model(tokenizer.vocab_size, trainer.seed)
    paused = train_streaming_production_stage(
        paused_model,
        package,
        recipe,
        trainer,
        work_dir=tmp_path / "resume-work",
        best_checkpoint_path=tmp_path / "resume-best.pt",
        device="cpu",
        max_run_seconds=1e-12,
        curriculum_plan=curriculum,
    )
    assert paused.completed is False
    assert paused.resume_checkpoint_sha256
    assert paused.next_cursor.epoch in {0, 1}

    resumed_model = _tiny_model(tokenizer.vocab_size, trainer.seed)
    resumed = train_streaming_production_stage(
        resumed_model,
        package,
        recipe,
        trainer,
        work_dir=tmp_path / "resume-work",
        best_checkpoint_path=tmp_path / "resume-best.pt",
        device="cpu",
        curriculum_plan=curriculum,
    )
    assert resumed.completed is True
    assert resumed.next_cursor == R2StreamingCursor(2, 0, 0, 0)

    full_model = _tiny_model(tokenizer.vocab_size, trainer.seed)
    full = train_streaming_production_stage(
        full_model,
        package,
        recipe,
        trainer,
        work_dir=tmp_path / "full-work",
        best_checkpoint_path=tmp_path / "full-best.pt",
        device="cpu",
        curriculum_plan=curriculum,
    )
    assert full.completed is True

    assert resumed.optimizer_steps == full.optimizer_steps
    assert resumed.micro_steps == full.micro_steps
    assert resumed.consumed_windows == full.consumed_windows
    assert resumed.target_tokens == full.target_tokens

    resumed_state = resumed_model.state_dict()
    full_state = full_model.state_dict()
    assert resumed_state.keys() == full_state.keys()
    for name in resumed_state:
        torch.testing.assert_close(
            resumed_state[name],
            full_state[name],
            rtol=0.0,
            atol=0.0,
        )
