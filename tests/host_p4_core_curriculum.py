from __future__ import annotations

import json

from vn97.p4_core_curriculum import (
    P4C_CATEGORIES,
    TRAIN_PER_CATEGORY,
    VALIDATION_PER_CATEGORY,
    build_p4c_curriculum,
    curriculum_jsonl_bytes,
    default_p4c_training,
    default_p4c_validation,
)


def test_default_p4c_curriculum_counts_and_categories() -> None:
    training = default_p4c_training()
    validation = default_p4c_validation()

    assert len(training) == (
        len(P4C_CATEGORIES)
        * TRAIN_PER_CATEGORY
    )
    assert len(validation) == (
        len(P4C_CATEGORIES)
        * VALIDATION_PER_CATEGORY
    )

    for records, expected_per_category in (
        (training, TRAIN_PER_CATEGORY),
        (validation, VALIDATION_PER_CATEGORY),
    ):
        counts = {
            category: 0
            for category in P4C_CATEGORIES
        }
        for record in records:
            counts[record.category] += 1
        assert counts == {
            category: expected_per_category
            for category in P4C_CATEGORIES
        }


def test_default_p4c_prompts_are_unique_and_disjoint() -> None:
    training = default_p4c_training()
    validation = default_p4c_validation()

    train_prompts = {
        record.prompt
        for record in training
    }
    validation_prompts = {
        record.prompt
        for record in validation
    }

    assert len(train_prompts) == len(training)
    assert len(validation_prompts) == len(validation)
    assert not train_prompts.intersection(
        validation_prompts
    )


def test_p4c_curriculum_is_deterministic() -> None:
    first = build_p4c_curriculum(
        split="training",
        per_category=12,
        seed=1234,
    )
    second = build_p4c_curriculum(
        split="training",
        per_category=12,
        seed=1234,
    )
    assert first == second
    assert (
        curriculum_jsonl_bytes(first)
        == curriculum_jsonl_bytes(
            second
        )
    )


def test_p4c_jsonl_matches_canonical_chat_shape() -> None:
    records = build_p4c_curriculum(
        split="validation",
        per_category=2,
        seed=77,
    )
    data = curriculum_jsonl_bytes(
        records
    )
    assert data.endswith(b"\n")

    rows = [
        json.loads(line)
        for line in data.decode(
            "utf-8"
        ).splitlines()
    ]
    assert len(rows) == (
        len(P4C_CATEGORIES) * 2
    )
    for row in rows:
        assert set(row) == {
            "messages"
        }
        assert len(
            row["messages"]
        ) == 2
        assert row["messages"][0][
            "role"
        ] == "user"
        assert row["messages"][1][
            "role"
        ] == "assistant"


def test_p4c_authority_curriculum_contains_allow_and_deny() -> None:
    records = build_p4c_curriculum(
        split="training",
        per_category=36,
        seed=99,
    )
    authority = [
        record
        for record in records
        if record.category
        == "authority_behavior"
    ]
    answers = {
        record.messages[-1].content
        for record in authority
    }
    assert any(
        '"decision":"allow"'
        in answer
        for answer in answers
    )
    assert any(
        '"decision":"deny"'
        in answer
        for answer in answers
    )
