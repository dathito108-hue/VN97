from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable


P5D3A_SCHEMA = "VN97P5D3A"
P5D3A_PROFILE_ID = "vn97-p5d3a-relational-teacher-features-v1"

TEACHER_HIDDEN_STATE_INDICES = (16, 32, 48, 64)
RELATIONAL_SEGMENTS = 4
EXCERPT_UTF8_BYTES = 384


def _safe_utf8_prefix(
    data: bytes,
    limit: int,
) -> str:
    return data[:limit].decode(
        "utf-8",
        errors="ignore",
    )


def _safe_utf8_suffix(
    data: bytes,
    limit: int,
) -> str:
    return data[-limit:].decode(
        "utf-8",
        errors="ignore",
    )


def canonical_transcript(
    row: dict[str, object],
) -> str:
    raw_messages = row.get(
        "messages"
    )
    if not isinstance(
        raw_messages,
        list,
    ):
        raise ValueError(
            "P5D3A row messages must be a list"
        )

    lines: list[str] = []
    for item in raw_messages:
        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(
                "P5D3A message must be an object"
            )
        role = str(
            item.get(
                "role",
                "",
            )
        ).strip()
        content = str(
            item.get(
                "content",
                "",
            )
        ).strip()
        if not role or not content:
            raise ValueError(
                "P5D3A message role/content must be non-empty"
            )
        lines.append(
            f"{role.upper()}: {content}"
        )

    teacher_response = str(
        row.get(
            "teacher_response",
            "",
        )
    ).strip()
    if not teacher_response:
        raise ValueError(
            "P5D3A teacher_response must be non-empty"
        )
    lines.append(
        f"ASSISTANT: {teacher_response}"
    )
    return "\n".join(
        lines
    )


def canonical_excerpt(
    row: dict[str, object],
) -> str:
    text = canonical_transcript(
        row
    )
    encoded = text.encode(
        "utf-8"
    )
    if len(encoded) <= EXCERPT_UTF8_BYTES:
        return text

    half = EXCERPT_UTF8_BYTES // 2
    head = _safe_utf8_prefix(
        encoded,
        half,
    ).rstrip()
    tail = _safe_utf8_suffix(
        encoded,
        half,
    ).lstrip()
    return (
        head
        + "\n[...semantic bridge...]\n"
        + tail
    )


def excerpt_sha256(
    row: dict[str, object],
) -> str:
    return hashlib.sha256(
        canonical_excerpt(
            row
        ).encode(
            "utf-8"
        )
    ).hexdigest()


@dataclass(frozen=True)
class P5D3ATarget:
    record_id: str
    category: str
    split: str
    excerpt_sha256: str
    teacher_revision: str
    teacher_token_count: int
    layer_grams: dict[
        str,
        list[list[float]],
    ]
    layer_relative_norms: dict[
        str,
        list[float],
    ]

    def __post_init__(self) -> None:
        if not self.record_id:
            raise ValueError(
                "record_id must be non-empty"
            )
        if not self.category:
            raise ValueError(
                "category must be non-empty"
            )
        if self.split not in {
            "train",
            "holdout",
        }:
            raise ValueError(
                "split must be train or holdout"
            )
        if len(self.excerpt_sha256) != 64:
            raise ValueError(
                "excerpt_sha256 must be SHA-256 hex"
            )
        if not self.teacher_revision:
            raise ValueError(
                "teacher_revision must be non-empty"
            )
        if self.teacher_token_count < RELATIONAL_SEGMENTS:
            raise ValueError(
                "teacher_token_count is too small"
            )

    def canonical_object(
        self,
    ) -> dict[str, object]:
        return {
            "category":
                self.category,
            "excerpt_sha256":
                self.excerpt_sha256,
            "layer_grams":
                self.layer_grams,
            "layer_relative_norms":
                self.layer_relative_norms,
            "record_id":
                self.record_id,
            "split":
                self.split,
            "teacher_revision":
                self.teacher_revision,
            "teacher_token_count":
                self.teacher_token_count,
        }


def target_digest(
    targets: Iterable[
        P5D3ATarget
    ],
) -> str:
    rows = sorted(
        (
            target.canonical_object()
            for target in targets
        ),
        key=lambda row:
            str(
                row[
                    "record_id"
                ]
            ),
    )
    payload = b"\n".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode(
            "utf-8"
        )
        for row in rows
    )
    return hashlib.sha256(
        b"VN97P5D3A\0"
        + payload
    ).hexdigest()


def profile_object() -> dict[str, object]:
    return {
        "excerpt_utf8_bytes":
            EXCERPT_UTF8_BYTES,
        "hidden_state_indices":
            list(
                TEACHER_HIDDEN_STATE_INDICES
            ),
        "profile_id":
            P5D3A_PROFILE_ID,
        "relational_segments":
            RELATIONAL_SEGMENTS,
        "schema":
            P5D3A_SCHEMA,
        "target_kind":
            "cross-tokenizer segment-cosine-gram",
    }


def profile_sha256() -> str:
    payload = json.dumps(
        profile_object(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode(
        "utf-8"
    )
    return hashlib.sha256(
        b"VN97P5D3APROFILE\0"
        + payload
    ).hexdigest()
