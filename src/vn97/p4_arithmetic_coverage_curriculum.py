from __future__ import annotations

import hashlib
import re

from .p4_compositional_repair_curriculum import (
    compositional_training as p4e_g_training,
)
from .p4_generalization_curriculum import (
    VN97P4DRecord,
    default_validation,
)
from .training import VN97ChatMessage


P4E_J_PROFILE_ID = "vn97-p4e-j-arithmetic-coverage-repair-v1"
P4E_J_SEED = 115197

_EXPR_RE = re.compile(
    r"(?P<expr>(?:\d+\s*\*\s*\(\s*\d+\s*\+\s*\d+\s*\))|"
    r"(?:\(\s*\d+\s*\+\s*\d+\s*\)\s*\*\s*\d+)|"
    r"(?:\d+\s*[+\-*]\s*\d+))"
)

_TEMPLATES = (
    "Calculate {expr}. Return only the number.",
    "Solve {expr}; answer with digits only.",
    "Evaluate {expr}. Return only the decimal integer.",
    "Perform {expr}. Output only the result and no prose.",
    "Tính {expr}. Chỉ trả lời kết quả bằng số.",
)


def _record(
    prompt: str,
    answer: str,
) -> VN97P4DRecord:
    return VN97P4DRecord(
        category="reasoning_planning",
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


def validation_expressions() -> set[str]:
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
                "could not parse a frozen reasoning validation expression"
            )
        result.add(expression)
    return result


def _fact_rows() -> list[
    tuple[str, str]
]:
    rows: list[
        tuple[str, str]
    ] = []

    for a in range(11, 97):
        for b in range(11, 97):
            rows.append(
                (
                    f"{a} + {b}",
                    str(a + b),
                )
            )

    for a in range(30, 120):
        for b in range(1, a):
            rows.append(
                (
                    f"{a} - {b}",
                    str(a - b),
                )
            )

    for a in range(3, 21):
        for b in range(2, 13):
            rows.append(
                (
                    f"{a} * {b}",
                    str(a * b),
                )
            )

    for a in range(2, 12):
        for b in range(2, 10):
            for c in range(2, 10):
                rows.append(
                    (
                        f"{a} * ({b} + {c})",
                        str(
                            a * (b + c)
                        ),
                    )
                )

    return rows


def arithmetic_coverage_training(
    *,
    forbidden_expressions: set[str] | None = None,
) -> tuple[
    VN97P4DRecord,
    ...
]:
    forbidden = set(
        validation_expressions()
    )
    if (
        forbidden_expressions
        is not None
    ):
        forbidden.update(
            normalize_expression(
                expression
            )
            for expression
            in forbidden_expressions
        )

    output: list[
        VN97P4DRecord
    ] = []
    seen_prompts: set[
        str
    ] = set()

    rows = _fact_rows()
    rows.sort(
        key=lambda item: hashlib.sha256(
            (
                item[0]
                + "\0"
                + item[1]
            ).encode(
                "utf-8"
            )
        ).digest()
    )

    for index, (
        expression,
        answer,
    ) in enumerate(rows):
        normalized = (
            normalize_expression(
                expression
            )
        )
        if normalized in forbidden:
            continue

        template = _TEMPLATES[
            index
            % len(_TEMPLATES)
        ]
        prompt = template.format(
            expr=expression
        )
        if prompt in seen_prompts:
            continue
        seen_prompts.add(
            prompt
        )
        output.append(
            _record(
                prompt,
                answer,
            )
        )

    # Preserve the non-reasoning capability learned in P4E-G.
    anchors = [
        item
        for item
        in p4e_g_training()
        if item.category
        != "reasoning_planning"
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
    output.extend(
        anchors[:1200]
    )

    return tuple(output)
