from __future__ import annotations

import hashlib
import random

from .p4_compositional_repair_curriculum import (
    compositional_training as p4e_g_training,
)
from .p4_generalization_curriculum import (
    VN97P4DRecord,
    default_training as p4d_training,
    default_validation as p4d_validation,
)
from .p4_targeted_repair_curriculum import (
    targeted_training as p4e_c_training,
)
from .training import VN97ChatMessage


P4E_I_PROFILE_ID = "vn97-p4e-i-numeric-arithmetic-repair-v1"
P4E_I_SEED = 94197
P4E_I_COUNTS = {
    "numeric_copy": 3000,
    "place_value": 2200,
    "direct_arithmetic": 6200,
    "anchors": 1000,
}


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


def _copy_record(
    rng: random.Random,
) -> VN97P4DRecord:
    value = rng.randrange(0, 1500)
    templates = (
        "Copy exactly this decimal integer: {value}. Return only that integer.",
        "The target number is {value}. Output exactly the target number and nothing else.",
        "Numeric identity task: value={value}. Respond with digits only.",
        "Sao chép chính xác số {value}. Chỉ trả lời bằng các chữ số của số đó.",
        "Return the literal integer between brackets and no other text: [{value}]",
        "Read NUMBER={value}. Your complete answer must be NUMBER.",
    )
    return _record(
        "reasoning_planning",
        rng.choice(templates).format(
            value=value
        ),
        str(value),
    )


def _place_value_record(
    rng: random.Random,
) -> VN97P4DRecord:
    value = rng.randrange(10, 999)
    digits = str(value)
    mode = rng.randrange(5)

    if mode == 0:
        prompt = (
            f"For the decimal integer {value}, return only its ones digit."
        )
        answer = digits[-1]
    elif mode == 1:
        prompt = (
            f"For the decimal integer {value}, return only its tens digit."
        )
        answer = digits[-2]
    elif mode == 2:
        prompt = (
            f"Number={value}. Return only the first decimal digit."
        )
        answer = digits[0]
    elif mode == 3:
        a = rng.randrange(0, 10)
        b = rng.randrange(0, 10)
        prompt = (
            f"Add the one-digit numbers {a} + {b}. Return only the result."
        )
        answer = str(a + b)
    else:
        a = rng.randrange(0, 10)
        b = rng.randrange(0, 10)
        carry = int(
            a + b >= 10
        )
        prompt = (
            f"When adding one decimal column {a} + {b}, what carry goes to the next column? "
            "Return only 0 or 1."
        )
        answer = str(carry)

    return _record(
        "reasoning_planning",
        prompt,
        answer,
    )


def _arithmetic_record(
    rng: random.Random,
) -> VN97P4DRecord:
    mode = rng.randrange(8)

    if mode in (0, 1):
        a = rng.randrange(0, 100)
        b = rng.randrange(0, 100)
        expr = f"{a} + {b}"
        answer = str(a + b)
    elif mode in (2, 3):
        a = rng.randrange(1, 140)
        b = rng.randrange(0, a + 1)
        expr = f"{a} - {b}"
        answer = str(a - b)
    elif mode in (4, 5):
        a = rng.randrange(0, 21)
        b = rng.randrange(0, 13)
        expr = f"{a} * {b}"
        answer = str(a * b)
    elif mode == 6:
        a = rng.randrange(2, 12)
        b = rng.randrange(2, 10)
        c = rng.randrange(2, 10)
        expr = f"{a} * ({b} + {c})"
        answer = str(
            a * (b + c)
        )
    else:
        a = rng.randrange(2, 12)
        b = rng.randrange(2, 10)
        c = rng.randrange(2, 10)
        expr = f"({a} + {b}) * {c}"
        answer = str(
            (a + b) * c
        )

    templates = (
        "Calculate {expr}. Return only the number.",
        "What is {expr}? No explanation, only the numeric result.",
        "Solve {expr}; answer with digits only.",
        "Evaluate {expr}. Return only the decimal integer.",
        "Tính {expr}. Chỉ trả lời kết quả bằng số.",
        "Perform {expr}. Output only the result and no prose.",
    )
    return _record(
        "reasoning_planning",
        rng.choice(templates).format(
            expr=expr
        ),
        answer,
    )


def numeric_copy_validation() -> tuple[
    VN97P4DRecord,
    ...
]:
    rng = random.Random(
        P4E_I_SEED + 100003
    )
    output: list[
        VN97P4DRecord
    ] = []
    seen: set[
        str
    ] = set()
    while len(output) < 60:
        record = _copy_record(
            rng
        )
        if record.prompt in seen:
            continue
        seen.add(
            record.prompt
        )
        output.append(
            record
        )
    return tuple(
        output
    )


def numeric_arithmetic_training() -> tuple[
    VN97P4DRecord,
    ...
]:
    blocked = {
        item.prompt
        for item in (
            *p4d_training(),
            *p4d_validation(),
            *p4e_c_training(),
            *p4e_g_training(),
            *numeric_copy_validation(),
        )
    }

    rng = random.Random(
        P4E_I_SEED
    )
    output: list[
        VN97P4DRecord
    ] = []
    seen: set[
        str
    ] = set()

    def fill(
        target: int,
        factory,
    ) -> None:
        added = 0
        attempts = 0
        while added < target:
            attempts += 1
            if attempts > target * 300:
                raise RuntimeError(
                    "could not build enough unique P4E-I records"
                )
            record = factory(
                rng
            )
            if (
                record.prompt
                in blocked
                or record.prompt
                in seen
            ):
                continue
            seen.add(
                record.prompt
            )
            output.append(
                record
            )
            added += 1

    fill(
        P4E_I_COUNTS[
            "numeric_copy"
        ],
        _copy_record,
    )
    fill(
        P4E_I_COUNTS[
            "place_value"
        ],
        _place_value_record,
    )
    fill(
        P4E_I_COUNTS[
            "direct_arithmetic"
        ],
        _arithmetic_record,
    )

    anchors = [
        item
        for item
        in p4e_g_training()
        if item.category
        != "reasoning_planning"
        and item.prompt
        not in blocked
        and item.prompt
        not in seen
    ]
    anchors.sort(
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
    anchors = anchors[
        :P4E_I_COUNTS[
            "anchors"
        ]
    ]
    if len(anchors) != (
        P4E_I_COUNTS[
            "anchors"
        ]
    ):
        raise RuntimeError(
            "not enough P4E-I anchor records"
        )
    output.extend(
        anchors
    )

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
    return tuple(
        output
    )
