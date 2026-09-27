from __future__ import annotations

import pytest

from vn97.r2.config import (
    r2_cpu_pilot_config,
    r2_mobile_1b_config,
)
from vn97.r2.production_contract import (
    R2ProductionCorpusManifest,
    R2ProductionShard,
    R2ProductionTrainingRecipe,
    assert_production_training_contract,
    assert_r2_production_scale,
    assert_stage_transition,
    estimate_production_training_resources,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _manifest(stage: str, parent: str | None) -> R2ProductionCorpusManifest:
    return R2ProductionCorpusManifest(
        stage=stage,
        tokenizer_sha256=SHA_A,
        parent_checkpoint_sha256=parent,
        shards=(
            R2ProductionShard(
                split="train",
                sha256=SHA_B,
                bytes=100,
                records=10,
                task_families=("language",),
            ),
            R2ProductionShard(
                split="validation",
                sha256=SHA_C,
                bytes=50,
                records=5,
                task_families=("language",),
            ),
        ),
    )


def test_mobile_1b_config_is_inside_locked_production_scale() -> None:
    config = r2_mobile_1b_config(4096)
    parameters = assert_r2_production_scale(config)
    assert 900_000_000 <= parameters <= 1_300_000_000


def test_pilot_config_cannot_enter_r2d_production_scale() -> None:
    with pytest.raises(ValueError, match="outside the locked"):
        assert_r2_production_scale(r2_cpu_pilot_config(4096))


def test_dense_pretrain_manifest_has_no_parent() -> None:
    manifest = _manifest("dense_pretrain", None)
    assert len(manifest.identity()) == 64

    with pytest.raises(ValueError, match="without a parent"):
        _manifest("dense_pretrain", SHA_D)


def test_post_pretrain_manifest_requires_parent_checkpoint() -> None:
    manifest = _manifest("instruction_reasoning", SHA_D)
    assert manifest.parent_checkpoint_sha256 == SHA_D

    with pytest.raises(ValueError, match="require parent checkpoint"):
        _manifest("instruction_reasoning", None)


def test_manifest_rejects_duplicate_shard_digest() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        R2ProductionCorpusManifest(
            stage="dense_pretrain",
            tokenizer_sha256=SHA_A,
            shards=(
                R2ProductionShard("train", SHA_B, 100, 10),
                R2ProductionShard("validation", SHA_B, 50, 5),
            ),
        )


def test_recipe_rejects_quantization_and_adapter_only_training() -> None:
    config = r2_mobile_1b_config(4096)

    with pytest.raises(ValueError, match="forbids QAT"):
        assert_production_training_contract(
            config,
            R2ProductionTrainingRecipe(
                sequence_length=128,
                micro_batch_size=1,
                gradient_accumulation_steps=16,
                optimizer_state_offload=True,
                quantization_used=True,
            ),
        )

    with pytest.raises(ValueError, match="full-parameter"):
        assert_production_training_contract(
            config,
            R2ProductionTrainingRecipe(
                sequence_length=128,
                micro_batch_size=1,
                gradient_accumulation_steps=16,
                optimizer_state_offload=True,
                full_parameter_training=False,
            ),
        )


def test_eager_scan_and_no_checkpointing_are_fail_closed() -> None:
    config = r2_mobile_1b_config(4096)

    with pytest.raises(ValueError, match="memory-efficient"):
        assert_production_training_contract(
            config,
            R2ProductionTrainingRecipe(
                sequence_length=128,
                micro_batch_size=1,
                gradient_accumulation_steps=16,
                optimizer_state_offload=True,
                memory_efficient_scan=False,
            ),
        )

    with pytest.raises(ValueError, match="activation checkpointing"):
        assert_production_training_contract(
            config,
            R2ProductionTrainingRecipe(
                sequence_length=128,
                micro_batch_size=1,
                gradient_accumulation_steps=16,
                optimizer_state_offload=True,
                activation_checkpointing=False,
            ),
        )


def test_optimizer_offload_moves_adam_state_to_host_budget() -> None:
    config = r2_mobile_1b_config(4096)
    recipe = R2ProductionTrainingRecipe(
        sequence_length=128,
        micro_batch_size=1,
        gradient_accumulation_steps=16,
        precision="fp16",
        optimizer_state_offload=True,
    )
    resources = estimate_production_training_resources(config, recipe)

    assert resources.device_optimizer_bytes == 0
    assert resources.host_optimizer_bytes == resources.parameter_count * 8
    assert (
        resources.minimum_device_bytes_before_activations
        < 16 * 1024**3
    )


def test_t4_budget_is_lower_bound_not_permission_to_train_eager_scan() -> None:
    config = r2_mobile_1b_config(4096)
    recipe = R2ProductionTrainingRecipe(
        sequence_length=128,
        micro_batch_size=1,
        gradient_accumulation_steps=16,
        precision="fp16",
        optimizer_state_offload=True,
    )
    resources = assert_production_training_contract(
        config,
        recipe,
        available_device_bytes=16 * 1024**3,
        available_host_bytes=32 * 1024**3,
    )
    assert resources.minimum_device_bytes_before_activations < 16 * 1024**3


def test_no_offload_fails_16gb_device_pre_activation_budget() -> None:
    config = r2_mobile_1b_config(4096)
    recipe = R2ProductionTrainingRecipe(
        sequence_length=128,
        micro_batch_size=1,
        gradient_accumulation_steps=16,
        precision="fp16",
        optimizer_state_offload=False,
    )
    with pytest.raises(RuntimeError, match="device memory"):
        assert_production_training_contract(
            config,
            recipe,
            available_device_bytes=16 * 1024**3,
        )


def test_stage_transition_is_strictly_ordered() -> None:
    assert_stage_transition(stage="dense_pretrain", parent_stage=None)
    assert_stage_transition(
        stage="instruction_reasoning",
        parent_stage="dense_pretrain",
    )
    assert_stage_transition(
        stage="tool_action",
        parent_stage="instruction_reasoning",
    )
    assert_stage_transition(stage="capability", parent_stage="tool_action")

    with pytest.raises(ValueError, match="invalid R2-D stage transition"):
        assert_stage_transition(
            stage="tool_action",
            parent_stage="dense_pretrain",
        )
