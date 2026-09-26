from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Iterable

from .training import VN97ChatMessage


P4C_CURRICULUM_SCHEMA = "VN97P4CURR1"
P4C_PROFILE_ID = "vn97-p4-core-instruction-cognition-v1"
P4C_CATEGORIES = (
    "instruction_following",
    "reasoning_planning",
    "memory_use",
    "structured_cognition",
    "tool_intent",
    "authority_behavior",
)

TRAIN_PER_CATEGORY = 1000
VALIDATION_PER_CATEGORY = 100
TRAIN_SEED = 4097
VALIDATION_SEED = 8194


@dataclass(frozen=True)
class VN97P4CurriculumRecord:
    category: str
    messages: tuple[VN97ChatMessage, ...]

    def __post_init__(self) -> None:
        if self.category not in P4C_CATEGORIES:
            raise ValueError("invalid P4C curriculum category")
        if len(self.messages) < 2:
            raise ValueError("P4C record requires at least user + assistant")
        if self.messages[-1].role != "assistant":
            raise ValueError("P4C record must end in an assistant target")

    @property
    def prompt(self) -> str:
        for message in reversed(self.messages[:-1]):
            if message.role == "user":
                return message.content
        raise ValueError("P4C record has no user prompt")

    def jsonl_object(self) -> dict[str, object]:
        return {
            "messages": [
                {
                    "content": message.content,
                    "role": message.role,
                }
                for message in self.messages
            ]
        }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def curriculum_jsonl_bytes(
    records: Iterable[VN97P4CurriculumRecord],
) -> bytes:
    rows = list(records)
    if not rows:
        raise ValueError("P4C curriculum must not be empty")
    return b"".join(
        _canonical_json(row.jsonl_object()) + b"\n"
        for row in rows
    )


def curriculum_sha256(
    records: Iterable[VN97P4CurriculumRecord],
) -> str:
    return hashlib.sha256(
        curriculum_jsonl_bytes(records)
    ).hexdigest()


def _chat(
    category: str,
    prompt: str,
    answer: str,
) -> VN97P4CurriculumRecord:
    return VN97P4CurriculumRecord(
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


def _instruction_record(
    *,
    split: str,
    index: int,
    rng: random.Random,
) -> VN97P4CurriculumRecord:
    token = (
        f"{split.upper()}_TOKEN_"
        f"{index:04d}_{rng.randrange(1000, 9999)}"
    )
    if index % 4 == 0:
        prompt = (
            "Reply with exactly this text and nothing else: "
            f"{token}"
        )
    elif index % 4 == 1:
        prompt = (
            "Return only the following token. Do not add explanation: "
            f"{token}"
        )
    elif index % 4 == 2:
        prompt = (
            "Chỉ trả lời đúng chuỗi sau, không thêm nội dung: "
            f"{token}"
        )
    else:
        prompt = (
            "Hãy xuất duy nhất mã này: "
            f"{token}"
        )
    return _chat(
        "instruction_following",
        prompt,
        token,
    )


def _reasoning_record(
    *,
    split: str,
    index: int,
    rng: random.Random,
) -> VN97P4CurriculumRecord:
    mode = index % 4
    if mode == 0:
        a = rng.randrange(10, 90)
        b = rng.randrange(10, 90)
        prompt = (
            f"Calculate {a} + {b}. Return only the number."
        )
        answer = str(a + b)
    elif mode == 1:
        a = rng.randrange(3, 20)
        b = rng.randrange(2, 12)
        prompt = (
            f"Calculate {a} * {b}. Return only the number."
        )
        answer = str(a * b)
    elif mode == 2:
        a = rng.randrange(10, 80)
        b = rng.randrange(1, a)
        prompt = (
            f"Tính {a} - {b}. Chỉ trả lời bằng số."
        )
        answer = str(a - b)
    else:
        a = rng.randrange(2, 12)
        b = rng.randrange(2, 10)
        c = rng.randrange(2, 10)
        prompt = (
            f"Calculate {a} * ({b} + {c}). Return only the number."
        )
        answer = str(a * (b + c))
    return _chat(
        "reasoning_planning",
        prompt,
        answer,
    )


def _memory_record(
    *,
    split: str,
    index: int,
    rng: random.Random,
) -> VN97P4CurriculumRecord:
    if index % 2 == 0:
        codeword = (
            f"{split.upper()}-MEM-"
            f"{index:04d}-{rng.randrange(100, 999)}"
        )
        prompt = (
            "Remember this fact only for this request: "
            f"codeword = {codeword}. "
            "What is the codeword? Return only the codeword."
        )
        answer = codeword
    else:
        names = [
            "An",
            "Binh",
            "Chi",
            "Dung",
            "Hoa",
            "Khanh",
            "Mai",
            "Nam",
        ]
        colors = [
            "amber",
            "blue",
            "green",
            "orange",
            "purple",
            "silver",
            "white",
            "yellow",
        ]
        first = names[
            (index + rng.randrange(len(names)))
            % len(names)
        ]
        second = names[
            (index + 3 + rng.randrange(len(names)))
            % len(names)
        ]
        if first == second:
            second = names[
                (names.index(first) + 1)
                % len(names)
            ]
        first_color = colors[
            (index + rng.randrange(len(colors)))
            % len(colors)
        ]
        second_color = colors[
            (index + 4 + rng.randrange(len(colors)))
            % len(colors)
        ]
        if first_color == second_color:
            second_color = colors[
                (colors.index(first_color) + 1)
                % len(colors)
            ]
        prompt = (
            f"Facts: {first} owns the {first_color} key. "
            f"{second} owns the {second_color} key. "
            f"Who owns the {first_color} key? Return only the name."
        )
        answer = first
    return _chat(
        "memory_use",
        prompt,
        answer,
    )


def _structured_record(
    *,
    split: str,
    index: int,
    rng: random.Random,
) -> VN97P4CurriculumRecord:
    if index % 3 == 0:
        value = rng.randrange(1, 100)
        prompt = (
            'Return exactly one JSON object with keys "answer" and '
            f'"confidence". The answer is {value} and confidence is 1. '
            "No prose."
        )
        answer = _canonical_json(
            {
                "answer": value,
                "confidence": 1,
            }
        ).decode("utf-8")
    elif index % 3 == 1:
        status = (
            "ready"
            if index % 2
            else "waiting"
        )
        retry = bool(index % 4 == 1)
        retry_text = "true" if retry else "false"
        prompt = (
            "Return exactly one JSON object. "
            f'status is "{status}" and retry is {retry_text}. '
            "No prose."
        )
        answer = _canonical_json(
            {
                "retry": retry,
                "status": status,
            }
        ).decode("utf-8")
    else:
        action = (
            f"{split}-step-{index:04d}"
        )
        prompt = (
            'Return exactly one JSON object with "action" and "ok". '
            f'action is "{action}" and ok is true. No prose.'
        )
        answer = _canonical_json(
            {
                "action": action,
                "ok": True,
            }
        ).decode("utf-8")
    return _chat(
        "structured_cognition",
        prompt,
        answer,
    )


_TOOL_CASES = (
    ("web.fetch", False, "fetch a public webpage without modifying anything"),
    ("file.read", False, "read a local file without modifying it"),
    ("memory.read", False, "read an existing memory record"),
    ("file.write", True, "write new data to a file"),
    ("app.action", True, "launch an external application"),
    ("device.interact", True, "change the device clipboard"),
)


def _tool_record(
    *,
    split: str,
    index: int,
    rng: random.Random,
) -> VN97P4CurriculumRecord:
    capability, approval, description = _TOOL_CASES[
        (index + rng.randrange(len(_TOOL_CASES)))
        % len(_TOOL_CASES)
    ]
    approval_text = (
        "true" if approval else "false"
    )
    prompt = (
        f"A user asks to {description}. "
        "Return only JSON: capability must be "
        f'"{capability}" and requires_approval must be '
        f"{approval_text}."
    )
    answer = _canonical_json(
        {
            "capability": capability,
            "requires_approval": approval,
        }
    ).decode("utf-8")
    return _chat(
        "tool_intent",
        prompt,
        answer,
    )


_AUTH_CASES = (
    (
        "An external write has no user approval.",
        "deny",
        "approval_required",
    ),
    (
        "An unknown external capability is requested.",
        "deny",
        "unknown_capability",
    ),
    (
        "A known read-only capability is requested within its allowed scope.",
        "allow",
        "authorized",
    ),
    (
        "A write capability has a valid explicit approval and allowed scope.",
        "allow",
        "authorized",
    ),
    (
        "A capability request exceeds its granted scope.",
        "deny",
        "scope_denied",
    ),
    (
        "An approval token is expired.",
        "deny",
        "approval_invalid",
    ),
)


def _authority_record(
    *,
    split: str,
    index: int,
    rng: random.Random,
) -> VN97P4CurriculumRecord:
    scenario, decision, reason = _AUTH_CASES[
        (index + rng.randrange(len(_AUTH_CASES)))
        % len(_AUTH_CASES)
    ]
    prompt = (
        f"{scenario} Return only JSON with decision "
        f'"{decision}" and reason "{reason}".'
    )
    answer = _canonical_json(
        {
            "decision": decision,
            "reason": reason,
        }
    ).decode("utf-8")
    return _chat(
        "authority_behavior",
        prompt,
        answer,
    )


_GENERATORS = (
    _instruction_record,
    _reasoning_record,
    _memory_record,
    _structured_record,
    _tool_record,
    _authority_record,
)


def build_p4c_curriculum(
    *,
    split: str,
    per_category: int,
    seed: int,
) -> tuple[VN97P4CurriculumRecord, ...]:
    if split not in {
        "training",
        "validation",
    }:
        raise ValueError(
            "P4C split must be training or validation"
        )
    if per_category <= 0:
        raise ValueError(
            "P4C per_category must be positive"
        )
    if seed < 0:
        raise ValueError(
            "P4C seed must be non-negative"
        )

    rng = random.Random(seed)
    records: list[VN97P4CurriculumRecord] = []
    for generator in _GENERATORS:
        for index in range(per_category):
            records.append(
                generator(
                    split=split,
                    index=index,
                    rng=rng,
                )
            )

    rng.shuffle(records)
    prompts = [
        record.prompt
        for record in records
    ]
    if len(prompts) != len(set(prompts)):
        raise ValueError(
            "P4C curriculum contains duplicate prompts"
        )
    return tuple(records)


def default_p4c_training() -> tuple[
    VN97P4CurriculumRecord,
    ...
]:
    return build_p4c_curriculum(
        split="training",
        per_category=
            TRAIN_PER_CATEGORY,
        seed=TRAIN_SEED,
    )


def default_p4c_validation() -> tuple[
    VN97P4CurriculumRecord,
    ...
]:
    return build_p4c_curriculum(
        split="validation",
        per_category=
            VALIDATION_PER_CATEGORY,
        seed=VALIDATION_SEED,
    )
