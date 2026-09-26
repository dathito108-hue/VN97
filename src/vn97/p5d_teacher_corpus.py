from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

from .p5d_distillation import (
    P5D0_PROFILE_ID,
    TEACHER_REPO,
)
from .training import VN97ChatMessage


P5D1_SCHEMA = "VN97P5D1"
P5D1_PROFILE_ID = "vn97-p5d1-falcon3-mamba-teacher-corpus-v1"

P4_PER_CATEGORY = 60
P3_RECORDS = 240

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise helpful assistant. Follow the user's requested output "
    "format exactly. Do not mention the model or training process."
)


@dataclass(frozen=True)
class P5D1Prompt:
    record_id: str
    category: str
    source: str
    messages: tuple[VN97ChatMessage, ...]
    reference_response: str | None
    max_new_tokens: int

    def __post_init__(self) -> None:
        if not self.record_id:
            raise ValueError("record_id must be non-empty")
        if not self.category:
            raise ValueError("category must be non-empty")
        if not self.source:
            raise ValueError("source must be non-empty")
        if not self.messages:
            raise ValueError("messages must be non-empty")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

    def canonical_object(self) -> dict[str, object]:
        return {
            "category": self.category,
            "max_new_tokens": self.max_new_tokens,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in self.messages
            ],
            "record_id": self.record_id,
            "reference_response": self.reference_response,
            "source": self.source,
        }


@dataclass(frozen=True)
class P5D1TeacherRecord:
    prompt: P5D1Prompt
    teacher_response: str
    teacher_revision: str

    def __post_init__(self) -> None:
        if not self.teacher_response:
            raise ValueError("teacher_response must be non-empty")
        if not self.teacher_revision:
            raise ValueError("teacher_revision must be non-empty")

    def canonical_object(self) -> dict[str, object]:
        return {
            "category": self.prompt.category,
            "max_new_tokens": self.prompt.max_new_tokens,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in self.prompt.messages
            ],
            "record_id": self.prompt.record_id,
            "reference_response": self.prompt.reference_response,
            "source": self.prompt.source,
            "teacher_repo": TEACHER_REPO,
            "teacher_response": self.teacher_response,
            "teacher_revision": self.teacher_revision,
        }


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def stable_record_id(
    *,
    category: str,
    source: str,
    messages: Sequence[VN97ChatMessage],
) -> str:
    payload = _canonical_json_bytes(
        {
            "category": category,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in messages
            ],
            "source": source,
        }
    )
    return hashlib.sha256(
        b"VN97P5D1PROMPT\0" + payload
    ).hexdigest()[:24]


def prompt_digest(
    prompts: Iterable[P5D1Prompt],
) -> str:
    rows = sorted(
        (
            prompt.canonical_object()
            for prompt in prompts
        ),
        key=lambda row: str(row["record_id"]),
    )
    return hashlib.sha256(
        b"VN97P5D1MANIFEST\0"
        + b"\n".join(
            _canonical_json_bytes(row)
            for row in rows
        )
    ).hexdigest()


def teacher_corpus_digest(
    records: Iterable[P5D1TeacherRecord],
) -> str:
    rows = sorted(
        (
            record.canonical_object()
            for record in records
        ),
        key=lambda row: str(row["record_id"]),
    )
    return hashlib.sha256(
        b"VN97P5D1CORPUS\0"
        + b"\n".join(
            _canonical_json_bytes(row)
            for row in rows
        )
    ).hexdigest()


def generation_limit_for_category(
    category: str,
) -> int:
    if category in {
        "instruction_following",
        "reasoning_planning",
        "memory_use",
    }:
        return 48
    if category in {
        "structured_cognition",
        "tool_intent",
        "authority_behavior",
    }:
        return 96
    return 192


def normalize_exact_output(
    text: str,
) -> str:
    return " ".join(
        text.strip().split()
    )


def teacher_matches_reference(
    *,
    category: str,
    teacher_response: str,
    reference_response: str | None,
) -> bool | None:
    if reference_response is None:
        return None

    teacher = normalize_exact_output(
        teacher_response
    )
    reference = normalize_exact_output(
        reference_response
    )

    if category in {
        "structured_cognition",
        "tool_intent",
        "authority_behavior",
    }:
        try:
            teacher_json = json.loads(teacher)
            reference_json = json.loads(reference)
        except json.JSONDecodeError:
            return False
        return teacher_json == reference_json

    return teacher == reference


def report_object(
    *,
    teacher_revision: str,
    manifest_sha256: str,
    corpus_sha256: str,
    total_records: int,
    completed_records: int,
    category_counts: dict[str, int],
    exact_reference_counts: dict[str, int],
) -> dict[str, object]:
    return {
        "category_counts": dict(
            sorted(category_counts.items())
        ),
        "completed_records": completed_records,
        "corpus_sha256": corpus_sha256,
        "exact_reference_counts": dict(
            sorted(exact_reference_counts.items())
        ),
        "manifest_sha256": manifest_sha256,
        "p5d0_profile_id": P5D0_PROFILE_ID,
        "profile_id": P5D1_PROFILE_ID,
        "schema": P5D1_SCHEMA,
        "teacher": {
            "repo": TEACHER_REPO,
            "revision": teacher_revision,
        },
        "total_records": total_records,
    }
