from __future__ import annotations

import json

import torch

from vn97.cognition_adapter import (
    VN97InferenceLimits,
    _no_repeat_banned_tokens,
    _select_greedy_token,
)
from vn97.p4_generalization_curriculum import (
    P4D_CATEGORIES,
    TRAIN_PER_CATEGORY,
    VALIDATION_PER_CATEGORY,
    default_training,
    default_validation,
)
from vn97.tokenizer import (
    VN97Tokenizer,
    VN97TokenizerPackage,
)
from vn97.training import (
    VN97ChatMessage,
    encode_chat_completion_messages,
)


def test_completion_mask_does_not_supervise_role_marker() -> None:
    tokenizer = VN97Tokenizer(
        VN97TokenizerPackage()
    )
    messages = (
        VN97ChatMessage(
            role="user",
            content="Question",
        ),
        VN97ChatMessage(
            role="assistant",
            content="Answer",
        ),
    )
    example = encode_chat_completion_messages(
        tokenizer,
        messages,
    )

    assistant_marker = tokenizer.encode(
        "\n<|assistant|>\n"
    )
    answer = tokenizer.encode(
        "Answer"
    )

    tokens = list(
        example.token_ids
    )
    mask = list(
        example.target_mask
    )
    start = None
    for index in range(
        len(tokens)
        - len(assistant_marker)
        + 1
    ):
        if (
            tokens[
                index:
                index
                + len(assistant_marker)
            ]
            == assistant_marker
        ):
            start = index
            break
    assert start is not None
    assert not any(
        mask[
            start:
            start
            + len(assistant_marker)
        ]
    )
    answer_start = (
        start
        + len(assistant_marker)
    )
    assert tokens[
        answer_start:
        answer_start
        + len(answer)
    ] == answer
    assert all(
        mask[
            answer_start:
            answer_start
            + len(answer)
        ]
    )


def test_p4d_curriculum_is_balanced_disjoint_and_has_no_split_markers() -> None:
    training = default_training()
    validation = default_validation()

    assert len(training) == (
        len(P4D_CATEGORIES)
        * TRAIN_PER_CATEGORY
    )
    assert len(validation) == (
        len(P4D_CATEGORIES)
        * VALIDATION_PER_CATEGORY
    )

    train_prompts = {
        item.prompt
        for item in training
    }
    validation_prompts = {
        item.prompt
        for item in validation
    }
    assert len(train_prompts) == len(
        training
    )
    assert len(
        validation_prompts
    ) == len(validation)
    assert not train_prompts.intersection(
        validation_prompts
    )

    for item in (
        list(training)
        + list(validation)
    ):
        lowered = (
            item.prompt
            + " "
            + item.answer
        ).casefold()
        assert "training_token" not in lowered
        assert "training-mem" not in lowered
        assert "case training" not in lowered
        assert "case validation" not in lowered

    for records, count in (
        (
            training,
            TRAIN_PER_CATEGORY,
        ),
        (
            validation,
            VALIDATION_PER_CATEGORY,
        ),
    ):
        actual = {
            category: 0
            for category
            in P4D_CATEGORIES
        }
        for item in records:
            actual[item.category] += 1
        assert actual == {
            category: count
            for category
            in P4D_CATEGORIES
        }


def test_p4d_structured_targets_are_canonical_json() -> None:
    for item in default_validation():
        if item.category not in {
            "structured_cognition",
            "tool_intent",
            "authority_behavior",
        }:
            continue
        parsed = json.loads(
            item.answer
        )
        canonical = json.dumps(
            parsed,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        assert item.answer == canonical


def test_no_repeat_ngram_bans_only_seen_continuation() -> None:
    generated = [
        10,
        20,
        30,
        40,
        20,
        30,
        40,
    ]
    banned = _no_repeat_banned_tokens(
        generated,
        4,
    )
    assert banned == {20}


def test_repetition_controls_change_greedy_choice_when_needed() -> None:
    logits = torch.tensor(
        [[[0.0, 10.0, 9.0]]],
        dtype=torch.float32,
    )
    token = _select_greedy_token(
        logits,
        [1],
        repetition_penalty=2.0,
        no_repeat_ngram_size=0,
    )
    assert token == 2


def test_inference_limit_validation() -> None:
    limits = VN97InferenceLimits(
        repetition_penalty=1.12,
        no_repeat_ngram_size=4,
    )
    assert limits.repetition_penalty == 1.12
    assert limits.no_repeat_ngram_size == 4
