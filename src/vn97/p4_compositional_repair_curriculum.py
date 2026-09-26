from __future__ import annotations

import hashlib
import json
import random

from .p4_generalization_curriculum import (
    P4D_CATEGORIES,
    VN97P4DRecord,
    build_p4d_curriculum,
    default_training as p4d_training,
    default_validation as p4d_validation,
)
from .p4_targeted_repair_curriculum import (
    targeted_training as p4e_c_training,
)
from .training import VN97ChatMessage


P4E_G_PROFILE_ID = "vn97-p4e-g-compositional-generalization-repair-v1"
P4E_G_SEED = 73197
P4E_G_COUNTS = {
    "instruction_following": 800,
    "reasoning_planning": 1400,
    "memory_use": 800,
    "structured_cognition": 1000,
    "tool_intent": 200,
    "authority_behavior": 200,
}


_WORDS_A = (
    "amber",
    "blue",
    "calm",
    "delta",
    "ember",
    "green",
    "lunar",
    "nova",
    "orbit",
    "silver",
    "violet",
    "zenith",
)
_WORDS_B = (
    "anchor",
    "bridge",
    "canyon",
    "falcon",
    "harbor",
    "maple",
    "pioneer",
    "river",
    "signal",
    "summit",
    "vector",
    "willow",
)
_NAMES = (
    "An",
    "Binh",
    "Chi",
    "Dung",
    "Hoa",
    "Khanh",
    "Lan",
    "Mai",
    "Nam",
    "Phuc",
    "Quynh",
    "Trang",
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _record(
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


def _phrase(rng: random.Random) -> str:
    return (
        f"{rng.choice(_WORDS_A)}-"
        f"{rng.choice(_WORDS_B)}-"
        f"{rng.randrange(100, 999)}"
    )


def _instruction(
    rng: random.Random,
) -> VN97P4DRecord:
    value = _phrase(rng)
    decoy = _phrase(rng)
    mode = rng.randrange(6)

    if mode == 0:
        prompt = (
            f"TARGET={value}. Copy TARGET exactly. "
            "Return only the copied value."
        )
    elif mode == 1:
        prompt = (
            f"Ignore DEC0Y={decoy}. The requested value is [{value}]. "
            "Output only the requested value."
        )
    elif mode == 2:
        prompt = (
            f"Exact response contract: value={value}; "
            "no prefix, suffix, punctuation, or explanation."
        )
    elif mode == 3:
        prompt = (
            f"Giá trị cần trả lời là {value}. "
            "Chỉ sao chép đúng giá trị đó, không thêm ký tự khác."
        )
    elif mode == 4:
        prompt = (
            f"Two labels are present: unused={decoy}; final={value}. "
            "Return the final label only."
        )
    else:
        prompt = (
            f"Read this literal token carefully: «{value}». "
            "Your whole answer must be that literal token."
        )

    return _record(
        "instruction_following",
        prompt,
        value,
    )


def _reasoning(
    rng: random.Random,
) -> VN97P4DRecord:
    mode = rng.randrange(6)

    if mode == 0:
        a = rng.randrange(11, 97)
        b = rng.randrange(11, 97)
        expr = f"{a} + {b}"
        answer = str(a + b)
    elif mode == 1:
        a = rng.randrange(30, 120)
        b = rng.randrange(1, a)
        expr = f"{a} - {b}"
        answer = str(a - b)
    elif mode == 2:
        a = rng.randrange(3, 21)
        b = rng.randrange(2, 13)
        expr = f"{a} * {b}"
        answer = str(a * b)
    elif mode == 3:
        a = rng.randrange(2, 12)
        b = rng.randrange(2, 10)
        c = rng.randrange(2, 10)
        expr = f"{a} * ({b} + {c})"
        answer = str(a * (b + c))
    elif mode == 4:
        a = rng.randrange(20, 90)
        b = rng.randrange(5, 40)
        c = rng.randrange(2, 12)
        expr = f"{a} + {b} - {c}"
        answer = str(a + b - c)
    else:
        a = rng.randrange(2, 12)
        b = rng.randrange(2, 10)
        c = rng.randrange(2, 10)
        expr = f"({a} + {b}) * {c}"
        answer = str((a + b) * c)

    templates = (
        "Compute {expr}. Return the decimal integer only.",
        "Evaluate {expr}. No explanation; digits only.",
        "Perform the arithmetic {expr} and output only its result.",
        "Tính chính xác {expr}. Chỉ xuất số kết quả.",
        "One calculation only: {expr}. Answer with the number and nothing else.",
        "Resolve {expr}. The response must contain only the integer result.",
    )

    return _record(
        "reasoning_planning",
        rng.choice(templates).format(
            expr=expr
        ),
        answer,
    )


def _memory(
    rng: random.Random,
) -> VN97P4DRecord:
    mode = rng.randrange(5)

    if mode == 0:
        value = _phrase(rng)
        prompt = (
            f"Temporary binding inside this request: SLOT_A={value}. "
            "What is SLOT_A? Return only its value."
        )
        answer = value
    elif mode == 1:
        value = _phrase(rng)
        decoy = _phrase(rng)
        prompt = (
            f"Context values: primary={value}; secondary={decoy}. "
            "Return only the primary value."
        )
        answer = value
    elif mode == 2:
        owner1, owner2 = rng.sample(
            _NAMES,
            2,
        )
        color1, color2 = rng.sample(
            _WORDS_A,
            2,
        )
        prompt = (
            f"Bindings: {owner1}->{color1}; {owner2}->{color2}. "
            f"Who is bound to {color1}? Return only the name."
        )
        answer = owner1
    elif mode == 3:
        owner1, owner2, owner3 = rng.sample(
            _NAMES,
            3,
        )
        item1, item2, item3 = rng.sample(
            _WORDS_B,
            3,
        )
        prompt = (
            f"Facts for this request only: {owner1} has {item1}; "
            f"{owner2} has {item2}; {owner3} has {item3}. "
            f"Who has {item2}? Answer only the name."
        )
        answer = owner2
    else:
        value = _phrase(rng)
        prompt = (
            f"Trong yêu cầu này, khóa tạm thời KEY={value}. "
            "Hãy trả lại đúng giá trị KEY và không thêm gì khác."
        )
        answer = value

    return _record(
        "memory_use",
        prompt,
        answer,
    )


def _structured(
    rng: random.Random,
) -> VN97P4DRecord:
    mode = rng.randrange(3)
    reference = rng.randrange(
        100000,
        999999,
    )

    if mode == 0:
        value = rng.randrange(
            1,
            200,
        )
        expected = {
            "answer": value,
            "confidence": 1,
        }
        prompt = (
            f"Reference {reference} is metadata only. "
            f"Return JSON only: answer is {value}; confidence is 1. "
            "Use exactly the keys answer and confidence."
        )
    elif mode == 1:
        status = rng.choice(
            ("ready", "waiting", "done")
        )
        retry = bool(
            rng.randrange(2)
        )
        retry_text = (
            "true"
            if retry
            else "false"
        )
        expected = {
            "retry": retry,
            "status": status,
        }
        prompt = (
            f"Reference {reference}. Build one JSON object and no prose. "
            f"status={status}; retry={retry_text}. "
            "Use exactly the keys retry and status."
        )
    else:
        action = (
            f"{rng.choice(_WORDS_A)}_"
            f"{rng.choice(_WORDS_B)}"
        )
        expected = {
            "action": action,
            "ok": True,
        }
        prompt = (
            f"Reference {reference}. Emit a single JSON object only. "
            f'action="{action}"; ok=true. '
            "Use exactly the keys action and ok."
        )

    return _record(
        "structured_cognition",
        prompt,
        _canonical_json(expected),
    )


def _weak_records(
    *,
    seed: int,
) -> tuple[VN97P4DRecord, ...]:
    rng = random.Random(seed)
    generators = {
        "instruction_following":
            _instruction,
        "reasoning_planning":
            _reasoning,
        "memory_use":
            _memory,
        "structured_cognition":
            _structured,
    }

    output: list[VN97P4DRecord] = []
    seen: set[str] = set()

    for category, generator in (
        generators.items()
    ):
        target = P4E_G_COUNTS[
            category
        ]
        attempts = 0
        while sum(
            item.category == category
            for item in output
        ) < target:
            attempts += 1
            if attempts > target * 200:
                raise RuntimeError(
                    "could not build enough unique P4E-G weak-category prompts"
                )
            record = generator(rng)
            if record.prompt in seen:
                continue
            seen.add(record.prompt)
            output.append(record)

    return tuple(output)


def compositional_training() -> tuple[
    VN97P4DRecord,
    ...
]:
    blocked = {
        item.prompt
        for item in (
            *p4d_training(),
            *p4d_validation(),
            *p4e_c_training(),
        )
    }

    selected: list[
        VN97P4DRecord
    ] = []
    seen: set[str] = set()

    for record in _weak_records(
        seed=P4E_G_SEED,
    ):
        if (
            record.prompt in blocked
            or record.prompt in seen
        ):
            continue
        selected.append(record)
        seen.add(record.prompt)

    anchor_candidates = (
        build_p4d_curriculum(
            validation=False,
            per_category=800,
            seed=P4E_G_SEED + 97,
        )
    )
    anchor_counts = {
        "tool_intent": 0,
        "authority_behavior": 0,
    }

    for record in sorted(
        anchor_candidates,
        key=lambda item: hashlib.sha256(
            (
                item.category
                + "\0"
                + item.prompt
            ).encode("utf-8")
        ).digest(),
    ):
        if record.category not in (
            "tool_intent",
            "authority_behavior",
        ):
            continue
        target = P4E_G_COUNTS[
            record.category
        ]
        if (
            anchor_counts[
                record.category
            ]
            >= target
            or record.prompt in blocked
            or record.prompt in seen
        ):
            continue
        selected.append(record)
        seen.add(record.prompt)
        anchor_counts[
            record.category
        ] += 1

    counts = {
        category: sum(
            item.category == category
            for item in selected
        )
        for category in P4D_CATEGORIES
    }
    if counts != P4E_G_COUNTS:
        raise RuntimeError(
            "P4E-G curriculum count mismatch: "
            f"{counts}"
        )

    selected.sort(
        key=lambda item: hashlib.sha256(
            (
                item.category
                + "\0"
                + item.prompt
            ).encode("utf-8")
        ).digest()
    )
    return tuple(selected)
