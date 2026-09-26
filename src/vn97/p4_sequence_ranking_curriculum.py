from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
import re

from .p4_compositional_repair_curriculum import (
    compositional_training as p4e_g_training,
)
from .p4_generalization_curriculum import (
    VN97P4DRecord,
    default_validation,
)
from .p4_numeric_arithmetic_curriculum import (
    numeric_copy_validation,
)
from .training import VN97ChatMessage


P4E_N_PROFILE_ID = "vn97-p4e-n-sequence-ranking-repair-v2-memory-safe"
P4E_N_SEED = 149197


@dataclass(frozen=True)
class P4ENRankingRecord:
    prompt: str
    correct: str
    negative: str
    operation: str


_EXPR_RE = re.compile(
    r"(?P<paren>\d+\s*\*\s*\(\s*\d+\s*\+\s*\d+\s*\))|"
    r"(?P<simple>\d+\s*[+\-*]\s*\d+)"
)

_TEMPLATES = (
    "Calculate {expr}. Return only the number.",
    "What is {expr}? No explanation, only the numeric result.",
    "Solve {expr}; answer with digits only.",
    "Evaluate {expr}. Return only the decimal integer.",
    "Perform {expr}. Output only the result and no prose.",
    "Tính {expr}. Chỉ trả lời kết quả bằng số.",
)


def _normalize_expression(text: str) -> str:
    return re.sub(
        r"\s+",
        "",
        text,
    )


def _validation_expressions() -> set[str]:
    output: set[str] = set()
    for record in default_validation():
        if record.category != "reasoning_planning":
            continue
        match = _EXPR_RE.search(record.prompt)
        if match is None:
            raise RuntimeError(
                "could not parse frozen reasoning validation expression"
            )
        output.add(
            _normalize_expression(
                match.group(0)
            )
        )
    return output


def _hard_negative(
    *,
    operation: str,
    operands: tuple[int, ...],
    correct: int,
    rng: random.Random,
) -> int:
    candidates: set[int] = {
        correct - 10,
        correct - 2,
        correct - 1,
        correct + 1,
        correct + 2,
        correct + 10,
    }

    if operation == "add":
        a, b = operands
        candidates.update(
            {
                a,
                b,
                abs(a - b),
                a + b - 5,
                a + b + 5,
                a * b,
            }
        )
    elif operation == "subtract":
        a, b = operands
        candidates.update(
            {
                a,
                b,
                a + b,
                abs(a - b) + 10,
                b - a,
            }
        )
    elif operation == "multiply":
        a, b = operands
        candidates.update(
            {
                a,
                b,
                a + b,
                a * max(1, b - 1),
                max(0, (a - 1) * b),
            }
        )
    else:
        a, b, c = operands
        candidates.update(
            {
                a * b + c,
                a + b * c,
                (a + b) * c,
                a * b,
                b + c,
            }
        )

    candidates = {
        value
        for value in candidates
        if value >= 0
        and value != correct
    }
    if not candidates:
        raise RuntimeError(
            "could not construct arithmetic hard negative"
        )
    ordered = sorted(candidates)
    return ordered[
        rng.randrange(
            len(ordered)
        )
    ]


def _sample_fact(
    rng: random.Random,
) -> tuple[
    str,
    tuple[int, ...],
    str,
    int,
]:
    mode = rng.randrange(4)

    if mode == 0:
        a = rng.randrange(11, 97)
        b = rng.randrange(11, 97)
        return (
            "add",
            (a, b),
            f"{a} + {b}",
            a + b,
        )
    if mode == 1:
        a = rng.randrange(30, 120)
        b = rng.randrange(1, a)
        return (
            "subtract",
            (a, b),
            f"{a} - {b}",
            a - b,
        )
    if mode == 2:
        a = rng.randrange(3, 21)
        b = rng.randrange(2, 13)
        return (
            "multiply",
            (a, b),
            f"{a} * {b}",
            a * b,
        )

    a = rng.randrange(2, 12)
    b = rng.randrange(2, 10)
    c = rng.randrange(2, 10)
    return (
        "mul_parenthesized",
        (a, b, c),
        f"{a} * ({b} + {c})",
        a * (b + c),
    )


def ranking_training(
    *,
    count: int = 1800,
    forbidden_expressions: set[str] | None = None,
    seed: int = P4E_N_SEED,
) -> tuple[P4ENRankingRecord, ...]:
    if count <= 0:
        raise ValueError(
            "ranking record count must be positive"
        )

    forbidden = set(
        _validation_expressions()
    )
    if forbidden_expressions:
        forbidden.update(
            _normalize_expression(
                item
            )
            for item in forbidden_expressions
        )

    rng = random.Random(seed)
    output: list[
        P4ENRankingRecord
    ] = []
    seen: set[str] = set()

    attempts = 0
    while len(output) < count:
        attempts += 1
        if attempts > count * 1000:
            raise RuntimeError(
                "could not build enough unique P4E-N ranking records"
            )

        (
            operation,
            operands,
            expression,
            correct,
        ) = _sample_fact(rng)

        normalized = _normalize_expression(
            expression
        )
        if (
            normalized in forbidden
            or normalized in seen
        ):
            continue

        prompt = rng.choice(
            _TEMPLATES
        ).format(
            expr=expression
        )
        negative = _hard_negative(
            operation=operation,
            operands=operands,
            correct=correct,
            rng=rng,
        )

        seen.add(normalized)
        output.append(
            P4ENRankingRecord(
                prompt=prompt,
                correct=str(correct),
                negative=str(negative),
                operation=operation,
            )
        )

    output.sort(
        key=lambda item: hashlib.sha256(
            (
                item.operation
                + "\0"
                + item.prompt
                + "\0"
                + item.correct
                + "\0"
                + item.negative
            ).encode("utf-8")
        ).digest()
    )
    return tuple(output)


def preservation_records(
    *,
    numeric_copy_count: int = 1400,
    anchor_count: int = 1600,
    seed: int = P4E_N_SEED + 1000,
) -> tuple[VN97P4DRecord, ...]:
    if numeric_copy_count <= 0 or anchor_count <= 0:
        raise ValueError(
            "preservation record counts must be positive"
        )

    held_out_copy = {
        item.prompt
        for item in numeric_copy_validation()
    }
    rng = random.Random(seed)

    copies: list[
        VN97P4DRecord
    ] = []
    seen: set[str] = set()

    templates = (
        "Copy exactly this decimal integer: {value}. Return only that integer.",
        "The target number is {value}. Output exactly the target number and nothing else.",
        "Numeric identity task: value={value}. Respond with digits only.",
        "Return the literal integer between brackets and no other text: [{value}]",
        "Read NUMBER={value}. Your complete answer must be NUMBER.",
        "Sao chép chính xác số {value}. Chỉ trả lời bằng các chữ số của số đó.",
    )

    attempts = 0
    while len(copies) < numeric_copy_count:
        attempts += 1
        if attempts > numeric_copy_count * 500:
            raise RuntimeError(
                "could not build enough P4E-N numeric-copy records"
            )
        value = rng.randrange(
            0,
            1500,
        )
        prompt = rng.choice(
            templates
        ).format(
            value=value
        )
        if (
            prompt in held_out_copy
            or prompt in seen
        ):
            continue
        seen.add(prompt)
        copies.append(
            VN97P4DRecord(
                category="reasoning_planning",
                messages=(
                    VN97ChatMessage(
                        role="user",
                        content=prompt,
                    ),
                    VN97ChatMessage(
                        role="assistant",
                        content=str(value),
                    ),
                ),
            )
        )

    pool = [
        item
        for item in p4e_g_training()
        if item.category
        != "reasoning_planning"
    ]
    by_category: dict[
        str,
        list[VN97P4DRecord],
    ] = {}
    for item in pool:
        by_category.setdefault(
            item.category,
            [],
        ).append(item)

    protected: list[
        VN97P4DRecord
    ] = []
    for category in (
        "tool_intent",
        "authority_behavior",
    ):
        rows = by_category.get(
            category,
            [],
        )
        rows.sort(
            key=lambda item: hashlib.sha256(
                (
                    item.category
                    + "\0"
                    + item.prompt
                ).encode("utf-8")
            ).digest()
        )
        protected.extend(rows)

    protected = protected[
        :anchor_count
    ]
    selected_prompts = {
        item.prompt
        for item in protected
    }
    remainder = [
        item
        for item in pool
        if item.prompt
        not in selected_prompts
    ]
    rng.shuffle(remainder)

    need = (
        anchor_count
        - len(protected)
    )
    if len(remainder) < need:
        raise RuntimeError(
            "not enough P4E-N capability anchors"
        )

    anchors = [
        *protected,
        *remainder[:need],
    ]

    output = [
        *copies,
        *anchors,
    ]
    output.sort(
        key=lambda item: hashlib.sha256(
            (
                item.category
                + "\0"
                + item.prompt
            ).encode("utf-8")
        ).digest()
    )
    return tuple(output)
