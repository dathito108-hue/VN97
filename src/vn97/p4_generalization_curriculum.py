from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Callable

from .training import VN97ChatMessage


P4D_PROFILE_ID = "vn97-p4d-generalization-repair-v1"
P4D_CATEGORIES = (
    "instruction_following",
    "reasoning_planning",
    "memory_use",
    "structured_cognition",
    "tool_intent",
    "authority_behavior",
)

TRAIN_PER_CATEGORY = 1000
VALIDATION_PER_CATEGORY = 150
TRAIN_SEED = 14097
VALIDATION_SEED = 28194


@dataclass(frozen=True)
class VN97P4DRecord:
    category: str
    messages: tuple[VN97ChatMessage, ...]

    @property
    def prompt(self) -> str:
        return self.messages[0].content

    @property
    def answer(self) -> str:
        return self.messages[-1].content


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def curriculum_bytes(
    records: tuple[VN97P4DRecord, ...],
) -> bytes:
    return b"".join(
        (
            _canonical_json(
                {
                    "messages": [
                        {
                            "content": message.content,
                            "role": message.role,
                        }
                        for message in record.messages
                    ]
                }
            )
            + "\n"
        ).encode("utf-8")
        for record in records
    )


def curriculum_sha256(
    records: tuple[VN97P4DRecord, ...],
) -> str:
    return hashlib.sha256(
        curriculum_bytes(records)
    ).hexdigest()


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


def _target_phrase(
    rng: random.Random,
) -> str:
    return (
        f"{rng.choice(_WORDS_A)}-"
        f"{rng.choice(_WORDS_B)}-"
        f"{rng.randrange(100, 999)}"
    )


_TRAIN_INSTRUCTION = (
    "Reply with only: {value}",
    "Output exactly {value}. Add nothing else.",
    "Your entire answer must be {value}.",
    "Chỉ trả lời đúng chuỗi {value}.",
    "Không giải thích; chỉ xuất {value}.",
)
_VALID_INSTRUCTION = (
    "Give no explanation. Respond solely with {value}.",
    "Return one exact value and nothing around it: {value}",
    "Câu trả lời duy nhất phải là {value}.",
)


def _instruction(
    rng: random.Random,
    *,
    validation: bool,
) -> VN97P4DRecord:
    value = _target_phrase(rng)
    templates = (
        _VALID_INSTRUCTION
        if validation
        else _TRAIN_INSTRUCTION
    )
    prompt = rng.choice(
        templates
    ).format(value=value)
    return _record(
        "instruction_following",
        prompt,
        value,
    )


_TRAIN_REASON = (
    "Calculate {expr}. Return only the number.",
    "Solve {expr}; answer with digits only.",
    "Tính {expr}. Chỉ trả lời kết quả.",
)
_VALID_REASON = (
    "What is {expr}? No explanation, only the numeric result.",
    "Hãy cho kết quả của {expr}, không thêm lời giải thích.",
)


def _reasoning(
    rng: random.Random,
    *,
    validation: bool,
) -> VN97P4DRecord:
    mode = rng.randrange(4)
    if mode == 0:
        a, b = rng.randrange(11, 97), rng.randrange(11, 97)
        expr, answer = f"{a} + {b}", str(a + b)
    elif mode == 1:
        a, b = rng.randrange(3, 21), rng.randrange(2, 13)
        expr, answer = f"{a} * {b}", str(a * b)
    elif mode == 2:
        a = rng.randrange(30, 120)
        b = rng.randrange(1, a)
        expr, answer = f"{a} - {b}", str(a - b)
    else:
        a, b, c = rng.randrange(2, 12), rng.randrange(2, 10), rng.randrange(2, 10)
        expr, answer = f"{a} * ({b} + {c})", str(a * (b + c))
    templates = (
        _VALID_REASON
        if validation
        else _TRAIN_REASON
    )
    return _record(
        "reasoning_planning",
        rng.choice(templates).format(expr=expr),
        answer,
    )


_TRAIN_MEMORY = (
    "Remember only for this request: codeword is {value}. What is the codeword? Return only it.",
    "Temporary fact: secret label = {value}. Repeat the secret label only.",
)
_VALID_MEMORY = (
    "Within this message, keep this fact: access word is {value}. What is the access word? Answer only the value.",
    "Use the supplied context only: marker={value}. Return the marker.",
)


def _memory(
    rng: random.Random,
    *,
    validation: bool,
) -> VN97P4DRecord:
    if rng.randrange(2) == 0:
        value = _target_phrase(rng)
        templates = (
            _VALID_MEMORY
            if validation
            else _TRAIN_MEMORY
        )
        return _record(
            "memory_use",
            rng.choice(templates).format(value=value),
            value,
        )

    owner1, owner2 = rng.sample(_NAMES, 2)
    color1, color2 = rng.sample(_WORDS_A, 2)
    if validation:
        prompt = (
            f"Context: {owner1} has the {color1} key while {owner2} has the {color2} key. "
            f"Who has the {color1} key? Answer only the name."
        )
    else:
        prompt = (
            f"Facts: {owner1} owns the {color1} key. {owner2} owns the {color2} key. "
            f"Return only the person who owns the {color1} key."
        )
    return _record(
        "memory_use",
        prompt,
        owner1,
    )


def _structured(
    rng: random.Random,
    *,
    validation: bool,
) -> VN97P4DRecord:
    mode = rng.randrange(3)
    if mode == 0:
        value = rng.randrange(1, 200)
        expected = {
            "answer": value,
            "confidence": 1,
        }
        if validation:
            prompt = (
                f"Produce one JSON object only. Put {value} in answer and 1 in confidence."
            )
        else:
            prompt = (
                f'Return JSON only with keys "answer" and "confidence": answer={value}, confidence=1.'
            )
    elif mode == 1:
        status = rng.choice(("ready", "waiting", "done"))
        retry = bool(rng.randrange(2))
        expected = {
            "retry": retry,
            "status": status,
        }
        retry_text = "true" if retry else "false"
        if validation:
            prompt = (
                f"Respond only as JSON. status must be {status}; retry must be {retry_text}."
            )
        else:
            prompt = (
                f'Output exactly one JSON object containing status="{status}" and retry={retry_text}.'
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
        if validation:
            prompt = (
                f"Return a JSON object only: action is {action}, ok is true."
            )
        else:
            prompt = (
                f'No prose. JSON with "action":"{action}" and "ok":true.'
            )
    return _record(
        "structured_cognition",
        prompt,
        _canonical_json(expected),
    )


_TOOL_CASES = (
    (
        "web.fetch",
        False,
        "fetch a public webpage without changing external state",
    ),
    (
        "file.read",
        False,
        "read an existing local file",
    ),
    (
        "memory.read",
        False,
        "retrieve an existing memory entry",
    ),
    (
        "file.write",
        True,
        "write new contents to a file",
    ),
    (
        "app.action",
        True,
        "launch an external application",
    ),
    (
        "device.interact",
        True,
        "modify the device clipboard",
    ),
)


def _tool(
    rng: random.Random,
    *,
    validation: bool,
) -> VN97P4DRecord:
    capability, approval, action = rng.choice(
        _TOOL_CASES
    )
    expected = {
        "capability": capability,
        "requires_approval": approval,
    }
    approval_text = (
        "true" if approval else "false"
    )
    subject = rng.choice(_NAMES)
    if validation:
        prompt = (
            f"{subject} requests that VN97 {action}. "
            f"Return JSON only with capability={capability} and requires_approval={approval_text}."
        )
    else:
        prompt = (
            f"A user asks to {action}. Classify the tool request. "
            f'JSON only: capability "{capability}", requires_approval {approval_text}.'
        )
    return _record(
        "tool_intent",
        prompt,
        _canonical_json(expected),
    )


_AUTH_CASES = (
    (
        "an external write has no explicit approval",
        "deny",
        "approval_required",
    ),
    (
        "the requested capability is unknown",
        "deny",
        "unknown_capability",
    ),
    (
        "a read-only capability is known and inside scope",
        "allow",
        "authorized",
    ),
    (
        "a write has valid explicit approval and stays in scope",
        "allow",
        "authorized",
    ),
    (
        "the request exceeds its granted scope",
        "deny",
        "scope_denied",
    ),
    (
        "the approval token is expired",
        "deny",
        "approval_invalid",
    ),
)


def _authority(
    rng: random.Random,
    *,
    validation: bool,
) -> VN97P4DRecord:
    scenario, decision, reason = rng.choice(
        _AUTH_CASES
    )
    expected = {
        "decision": decision,
        "reason": reason,
    }
    if validation:
        prompt = (
            f"Policy decision needed because {scenario}. "
            f"Return one JSON object only: decision={decision}, reason={reason}."
        )
    else:
        prompt = (
            f"Scenario: {scenario}. "
            f'Output JSON only with decision "{decision}" and reason "{reason}".'
        )
    return _record(
        "authority_behavior",
        prompt,
        _canonical_json(expected),
    )


_GENERATORS: tuple[
    Callable[..., VN97P4DRecord],
    ...
] = (
    _instruction,
    _reasoning,
    _memory,
    _structured,
    _tool,
    _authority,
)


def build_p4d_curriculum(
    *,
    validation: bool,
    per_category: int,
    seed: int,
) -> tuple[VN97P4DRecord, ...]:
    if per_category <= 0:
        raise ValueError(
            "per_category must be positive"
        )
    rng = random.Random(seed)
    output: list[VN97P4DRecord] = []
    prompts: set[str] = set()

    for generator in _GENERATORS:
        category_count = 0
        attempts = 0
        while category_count < per_category:
            attempts += 1
            if attempts > per_category * 100:
                raise RuntimeError(
                    "could not generate enough unique P4D prompts"
                )
            record = generator(
                rng,
                validation=validation,
            )
            if record.prompt in prompts:
                continue
            prompts.add(record.prompt)
            output.append(record)
            category_count += 1

    rng.shuffle(output)
    return tuple(output)


def default_training() -> tuple[
    VN97P4DRecord,
    ...
]:
    return build_p4d_curriculum(
        validation=False,
        per_category=
            TRAIN_PER_CATEGORY,
        seed=TRAIN_SEED,
    )


def default_validation() -> tuple[
    VN97P4DRecord,
    ...
]:
    return build_p4d_curriculum(
        validation=True,
        per_category=
            VALIDATION_PER_CATEGORY,
        seed=VALIDATION_SEED,
    )
