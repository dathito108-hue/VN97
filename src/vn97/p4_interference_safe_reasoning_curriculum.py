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


P4E_K_PROFILE_ID = "vn97-p4e-k-interference-safe-reasoning-v1"
P4E_K_SEED = 131197


@dataclass(frozen=True)
class P4EKCandidateProfile:
    candidate_id: str
    reasoning_records: int
    numeric_copy_records: int
    anchor_records: int
    p3_replay_records: int
    learning_rate: float
    seed_offset: int


P4E_K_CANDIDATES = (
    P4EKCandidateProfile(
        candidate_id="balanced",
        reasoning_records=1200,
        numeric_copy_records=1600,
        anchor_records=1600,
        p3_replay_records=1200,
        learning_rate=4e-6,
        seed_offset=11,
    ),
    P4EKCandidateProfile(
        candidate_id="reasoning_cautious",
        reasoning_records=1800,
        numeric_copy_records=1800,
        anchor_records=1800,
        p3_replay_records=1200,
        learning_rate=3e-6,
        seed_offset=29,
    ),
    P4EKCandidateProfile(
        candidate_id="retention_heavy",
        reasoning_records=1000,
        numeric_copy_records=2000,
        anchor_records=2000,
        p3_replay_records=1600,
        learning_rate=2e-6,
        seed_offset=47,
    ),
)


_EXPR_RE = re.compile(
    r"(?P<expr>(?:\d+\s*\*\s*\(\s*\d+\s*\+\s*\d+\s*\))|"
    r"(?:\d+\s*[+\-*]\s*\d+))"
)

_COPY_TEMPLATES = (
    "Copy exactly this decimal integer: {value}. Return only that integer.",
    "The target number is {value}. Output exactly the target number and nothing else.",
    "Numeric identity task: value={value}. Respond with digits only.",
    "Return the literal integer between brackets and no other text: [{value}]",
    "Read NUMBER={value}. Your complete answer must be NUMBER.",
    "Sao chép chính xác số {value}. Chỉ trả lời bằng các chữ số của số đó.",
)

_REASONING_TEMPLATES = (
    "Calculate {expr}. Return only the number.",
    "What is {expr}? No explanation, only the numeric result.",
    "Solve {expr}; answer with digits only.",
    "Evaluate {expr}. Return only the decimal integer.",
    "Perform {expr}. Output only the result and no prose.",
    "Tính {expr}. Chỉ trả lời kết quả bằng số.",
)


def normalize_expression(
    expression: str,
) -> str:
    return re.sub(
        r"\s+",
        "",
        expression,
    )


def extract_expression(
    prompt: str,
) -> str | None:
    match = _EXPR_RE.search(
        prompt
    )
    if match is None:
        return None
    return normalize_expression(
        match.group("expr")
    )


def held_out_expressions(
    extra_prompts: tuple[str, ...] = (),
) -> set[str]:
    result: set[str] = set()

    for record in default_validation():
        if (
            record.category
            != "reasoning_planning"
        ):
            continue
        expression = extract_expression(
            record.prompt
        )
        if expression is None:
            raise RuntimeError(
                "could not parse frozen reasoning validation expression"
            )
        result.add(
            expression
        )

    for prompt in extra_prompts:
        expression = extract_expression(
            prompt
        )
        if expression is not None:
            result.add(
                expression
            )

    return result


def _record(
    *,
    category: str,
    prompt: str,
    answer: str,
) -> VN97P4DRecord:
    return VN97P4DRecord(
        category=category,
        messages=(
            VN97ChatMessage(
                role="user",
                content=prompt,
            ),
            VN97ChatMessage(
                role="assistant",
                content=answer,
            ),
        ),
    )


def _arithmetic_fact(
    rng: random.Random,
) -> tuple[str, str]:
    mode = rng.randrange(4)

    if mode == 0:
        a = rng.randrange(11, 97)
        b = rng.randrange(11, 97)
        return (
            f"{a} + {b}",
            str(a + b),
        )

    if mode == 1:
        a = rng.randrange(30, 120)
        b = rng.randrange(1, a)
        return (
            f"{a} - {b}",
            str(a - b),
        )

    if mode == 2:
        a = rng.randrange(3, 21)
        b = rng.randrange(2, 13)
        return (
            f"{a} * {b}",
            str(a * b),
        )

    a = rng.randrange(2, 12)
    b = rng.randrange(2, 10)
    c = rng.randrange(2, 10)
    return (
        f"{a} * ({b} + {c})",
        str(a * (b + c)),
    )


def _reasoning_records(
    *,
    count: int,
    seed: int,
    forbidden_expressions: set[str],
) -> tuple[VN97P4DRecord, ...]:
    rng = random.Random(
        seed
    )
    output: list[
        VN97P4DRecord
    ] = []
    seen_prompts: set[
        str
    ] = set()
    seen_expressions: set[
        str
    ] = set()

    attempts = 0
    while len(output) < count:
        attempts += 1
        if attempts > count * 500:
            raise RuntimeError(
                "could not build enough unique P4E-K reasoning records"
            )

        expression, answer = (
            _arithmetic_fact(
                rng
            )
        )
        normalized = (
            normalize_expression(
                expression
            )
        )
        if (
            normalized
            in forbidden_expressions
            or normalized
            in seen_expressions
        ):
            continue

        template = rng.choice(
            _REASONING_TEMPLATES
        )
        prompt = template.format(
            expr=expression
        )
        if prompt in seen_prompts:
            continue

        seen_expressions.add(
            normalized
        )
        seen_prompts.add(
            prompt
        )
        output.append(
            _record(
                category=
                    "reasoning_planning",
                prompt=prompt,
                answer=answer,
            )
        )

    return tuple(
        output
    )


def _copy_records(
    *,
    count: int,
    seed: int,
) -> tuple[VN97P4DRecord, ...]:
    rng = random.Random(
        seed
    )
    held_out_prompts = {
        record.prompt
        for record in numeric_copy_validation()
    }
    output: list[
        VN97P4DRecord
    ] = []
    seen: set[
        str
    ] = set()

    attempts = 0
    while len(output) < count:
        attempts += 1
        if attempts > count * 500:
            raise RuntimeError(
                "could not build enough unique P4E-K copy records"
            )

        value = rng.randrange(
            0,
            1500,
        )
        prompt = rng.choice(
            _COPY_TEMPLATES
        ).format(
            value=value
        )

        if (
            prompt in held_out_prompts
            or prompt in seen
        ):
            continue

        seen.add(
            prompt
        )
        output.append(
            _record(
                category=
                    "reasoning_planning",
                prompt=prompt,
                answer=str(value),
            )
        )

    return tuple(
        output
    )


def _anchor_records(
    *,
    count: int,
    seed: int,
) -> tuple[VN97P4DRecord, ...]:
    pool = [
        item
        for item
        in p4e_g_training()
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
        ).append(
            item
        )

    # Tool intent and authority are explicitly protected first because P4E-J
    # regressed those capabilities despite improving arithmetic only slightly.
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
                ).encode(
                    "utf-8"
                )
            ).digest()
        )
        protected.extend(
            rows
        )

    if len(protected) > count:
        protected = protected[
            :count
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

    rng = random.Random(
        seed
    )
    rng.shuffle(
        remainder
    )

    need = (
        count
        - len(protected)
    )
    if len(remainder) < need:
        raise RuntimeError(
            "not enough P4E-K anchor records"
        )

    selected = [
        *protected,
        *remainder[:need],
    ]
    selected.sort(
        key=lambda item: hashlib.sha256(
            (
                item.category
                + "\0"
                + item.prompt
            ).encode(
                "utf-8"
            )
        ).digest()
    )
    return tuple(
        selected
    )


def candidate_training_records(
    profile: P4EKCandidateProfile,
    *,
    forbidden_expressions: set[str],
) -> tuple[VN97P4DRecord, ...]:
    seed = (
        P4E_K_SEED
        + profile.seed_offset
    )

    reasoning = _reasoning_records(
        count=
            profile.reasoning_records,
        seed=seed,
        forbidden_expressions=
            forbidden_expressions,
    )
    copies = _copy_records(
        count=
            profile.numeric_copy_records,
        seed=seed + 1000,
    )
    anchors = _anchor_records(
        count=
            profile.anchor_records,
        seed=seed + 2000,
    )

    output = [
        *reasoning,
        *copies,
        *anchors,
    ]
    output.sort(
        key=lambda item: hashlib.sha256(
            (
                item.category
                + "\0"
                + item.prompt
            ).encode(
                "utf-8"
            )
        ).digest()
    )

    if len(output) != (
        profile.reasoning_records
        + profile.numeric_copy_records
        + profile.anchor_records
    ):
        raise RuntimeError(
            "P4E-K candidate record count mismatch"
        )

    return tuple(
        output
    )
