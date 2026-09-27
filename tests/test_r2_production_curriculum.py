from __future__ import annotations

from copy import deepcopy

import pytest

from vn97.r2.production_curriculum import (
    R2D8CurriculumDefinition,
    compile_r2d8_plan,
    curriculum_epoch_shards,
    verify_r2d8_plan,
)


def _sha(char: str) -> str:
    return char * 64


def _index() -> dict[str, object]:
    shards = [
        {
            "split": "training",
            "source_manifest_id": _sha("a"),
            "shard_id": _sha("1"),
            "task_families": ["language"],
            "target_tokens": 100,
        },
        {
            "split": "training",
            "source_manifest_id": _sha("b"),
            "shard_id": _sha("2"),
            "task_families": ["language"],
            "target_tokens": 100,
        },
        {
            "split": "training",
            "source_manifest_id": _sha("c"),
            "shard_id": _sha("3"),
            "task_families": ["reasoning"],
            "target_tokens": 100,
        },
        {
            "split": "training",
            "source_manifest_id": _sha("d"),
            "shard_id": _sha("4"),
            "task_families": ["reasoning"],
            "target_tokens": 100,
        },
        {
            "split": "validation",
            "source_manifest_id": _sha("e"),
            "shard_id": _sha("5"),
            "task_families": ["language"],
            "target_tokens": 20,
        },
        {
            "split": "release",
            "source_manifest_id": _sha("f"),
            "shard_id": _sha("6"),
            "task_families": ["language"],
            "target_tokens": 20,
        },
    ]
    return {
        "index_id": _sha("9"),
        "release_held_out": True,
        "shards": shards,
        "split_totals": {
            "training": {"target_tokens": 400},
            "validation": {"target_tokens": 20},
            "release": {"target_tokens": 20},
        },
    }


def _definition(
    *,
    weights=(("language", 0.5), ("reasoning", 0.5)),
    assignments=None,
    epochs: int = 2,
    seed: int = 97,
) -> R2D8CurriculumDefinition:
    if assignments is None:
        assignments = (
            (_sha("a"), "language"),
            (_sha("b"), "language"),
            (_sha("c"), "reasoning"),
            (_sha("d"), "reasoning"),
        )
    return R2D8CurriculumDefinition(
        stage="dense_pretrain",
        epochs=epochs,
        seed=seed,
        family_weights=tuple(weights),
        primary_family_by_manifest=tuple(assignments),
    )


def test_curriculum_plan_is_deterministic_and_balanced() -> None:
    index_a = _index()
    index_b = deepcopy(index_a)
    index_b["shards"] = list(reversed(index_b["shards"]))

    plan_a = compile_r2d8_plan(index_a, _definition())
    plan_b = compile_r2d8_plan(index_b, _definition())

    assert plan_a == plan_b
    assert plan_a["plan_id"] == plan_b["plan_id"]
    assert plan_a["stage"] == "dense_pretrain"
    assert plan_a["release_held_out"] is True

    for epoch in plan_a["epoch_schedules"]:
        assert epoch["target_tokens"] == 400
        assert epoch["realized_weights"] == {
            "language": 0.5,
            "reasoning": 0.5,
        }
        assert epoch["absolute_weight_errors"] == {
            "language": 0.0,
            "reasoning": 0.0,
        }
        assert len(epoch["shard_ids"]) == 4


def test_curriculum_epoch_shards_follows_plan_order() -> None:
    index = _index()
    plan = compile_r2d8_plan(index, _definition(epochs=1))
    order = curriculum_epoch_shards(
        plan,
        index,
        epoch=0,
    )
    assert [item["shard_id"] for item in order] == (
        plan["epoch_schedules"][0]["shard_ids"]
    )


def test_curriculum_rejects_primary_family_not_declared_by_source() -> None:
    index = _index()
    assignments = (
        (_sha("a"), "reasoning"),
        (_sha("b"), "language"),
        (_sha("c"), "reasoning"),
        (_sha("d"), "reasoning"),
    )
    with pytest.raises(ValueError, match="must be declared"):
        compile_r2d8_plan(
            index,
            _definition(assignments=assignments),
        )


def test_dense_pretrain_rejects_tool_family() -> None:
    with pytest.raises(ValueError, match="not allowed"):
        R2D8CurriculumDefinition(
            stage="dense_pretrain",
            epochs=1,
            seed=97,
            family_weights=(
                ("language", 0.5),
                ("tool", 0.5),
            ),
            primary_family_by_manifest=(
                (_sha("a"), "language"),
                (_sha("b"), "tool"),
            ),
        )


def test_curriculum_rejects_coarse_shards_that_miss_weight_bound() -> None:
    index = {
        "index_id": _sha("9"),
        "release_held_out": True,
        "shards": [
            {
                "split": "training",
                "source_manifest_id": _sha("a"),
                "shard_id": _sha("1"),
                "task_families": ["language"],
                "target_tokens": 100,
            },
            {
                "split": "training",
                "source_manifest_id": _sha("c"),
                "shard_id": _sha("3"),
                "task_families": ["reasoning"],
                "target_tokens": 100,
            },
        ],
        "split_totals": {
            "training": {"target_tokens": 200},
        },
    }
    definition = R2D8CurriculumDefinition(
        stage="dense_pretrain",
        epochs=1,
        seed=97,
        family_weights=(
            ("language", 0.9),
            ("reasoning", 0.1),
        ),
        primary_family_by_manifest=(
            (_sha("a"), "language"),
            (_sha("c"), "reasoning"),
        ),
    )
    with pytest.raises(RuntimeError, match="shard granularity"):
        compile_r2d8_plan(index, definition)


def test_curriculum_plan_tamper_is_rejected() -> None:
    index = _index()
    plan = compile_r2d8_plan(index, _definition())
    tampered = deepcopy(plan)
    tampered["epoch_schedules"][0]["shard_ids"][0] = _sha("8")

    with pytest.raises(ValueError, match="identity mismatch"):
        verify_r2d8_plan(tampered, index)


def test_stage_specific_family_policy_expands_in_order() -> None:
    instruction = R2D8CurriculumDefinition(
        stage="instruction_reasoning",
        epochs=1,
        seed=97,
        family_weights=(
            ("instruction", 0.5),
            ("reasoning", 0.5),
        ),
        primary_family_by_manifest=(
            (_sha("a"), "instruction"),
            (_sha("b"), "reasoning"),
        ),
    )
    assert instruction.stage == "instruction_reasoning"

    tool = R2D8CurriculumDefinition(
        stage="tool_action",
        epochs=1,
        seed=97,
        family_weights=(
            ("action", 0.5),
            ("tool", 0.5),
        ),
        primary_family_by_manifest=(
            (_sha("a"), "action"),
            (_sha("b"), "tool"),
        ),
    )
    assert tool.stage == "tool_action"

    capability = R2D8CurriculumDefinition(
        stage="capability",
        epochs=1,
        seed=97,
        family_weights=(("capability", 1.0),),
        primary_family_by_manifest=(
            (_sha("a"), "capability"),
        ),
    )
    assert capability.stage == "capability"
