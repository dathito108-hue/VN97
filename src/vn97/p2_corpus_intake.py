from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Sequence


class VN97P2CorpusIntakeError(RuntimeError):
    pass


_MAX_MESSAGE_BYTES = 8 * 1024
_MAX_CONVERSATION_BYTES = 16 * 1024
_SPLITS = ("training", "validation", "release")
_IDENTITY_CONTAMINATION_TERMS = (
    "hypermamba",
    "vimind",
    "chatgpt",
    "claude",
    "gemini",
    "copilot",
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class VN97P2SourceSpec:
    source_id: str
    origin: str
    license: str
    adapter: str
    max_records: int

    def __post_init__(self) -> None:
        for value, label, bound in (
            (self.source_id, "source_id", 128),
            (self.origin, "origin", 1024),
            (self.license, "license", 256),
            (self.adapter, "adapter", 64),
        ):
            if (
                not isinstance(value, str)
                or not value
                or len(value.encode("utf-8")) > bound
            ):
                raise VN97P2CorpusIntakeError(
                    f"{label} is outside bounds"
                )
        if self.max_records <= 0:
            raise VN97P2CorpusIntakeError(
                "max_records must be positive"
            )


CANONICAL_SOURCES = (
    VN97P2SourceSpec(
        source_id="vi-dialogue-hoanghai2110",
        origin=(
            "https://huggingface.co/datasets/"
            "hoanghai2110/vietnamese-dataset"
        ),
        license="Apache-2.0",
        adapter="messages",
        max_records=800,
    ),
    VN97P2SourceSpec(
        source_id="databricks-dolly-15k",
        origin=(
            "https://huggingface.co/datasets/"
            "databricks/databricks-dolly-15k"
        ),
        license="CC-BY-SA-3.0",
        adapter="dolly",
        max_records=600,
    ),
    VN97P2SourceSpec(
        source_id="openai-gsm8k-train",
        origin=(
            "https://huggingface.co/datasets/"
            "openai/gsm8k"
        ),
        license="MIT",
        adapter="gsm8k",
        max_records=600,
    ),
)


def source_spec(source_id: str) -> VN97P2SourceSpec:
    for value in CANONICAL_SOURCES:
        if value.source_id == source_id:
            return value
    raise VN97P2CorpusIntakeError(
        f"unknown P2 source: {source_id}"
    )


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise VN97P2CorpusIntakeError(
            f"{label} must be text"
        )
    value = value.strip()
    size = len(value.encode("utf-8"))
    if not value or size > _MAX_MESSAGE_BYTES:
        raise VN97P2CorpusIntakeError(
            f"{label} must be 1..{_MAX_MESSAGE_BYTES} UTF-8 bytes"
        )
    return value


def _validate_messages(
    messages: Sequence[dict[str, object]],
) -> tuple[dict[str, str], ...]:
    if not messages:
        raise VN97P2CorpusIntakeError(
            "conversation must not be empty"
        )
    out: list[dict[str, str]] = []
    total = 0
    for raw in messages:
        if (
            not isinstance(raw, dict)
            or set(raw) != {"role", "content"}
        ):
            raise VN97P2CorpusIntakeError(
                "chat messages must contain exactly role/content"
            )
        role = raw["role"]
        if role not in {"system", "user", "assistant"}:
            raise VN97P2CorpusIntakeError(
                "chat role is invalid"
            )
        content = _text(
            raw["content"],
            label="message content",
        )
        total += len(content.encode("utf-8"))
        if total > _MAX_CONVERSATION_BYTES:
            raise VN97P2CorpusIntakeError(
                "conversation exceeds byte bound"
            )
        out.append(
            {"role": str(role), "content": content}
        )
    if out[-1]["role"] != "assistant":
        raise VN97P2CorpusIntakeError(
            "P2 conversation must end with assistant target"
        )
    if not any(item["role"] == "user" for item in out):
        raise VN97P2CorpusIntakeError(
            "P2 conversation must contain a user turn"
        )
    return tuple(out)


def adapt_record(
    source_id: str,
    raw: object,
) -> tuple[dict[str, str], ...]:
    spec = source_spec(source_id)
    if not isinstance(raw, dict):
        raise VN97P2CorpusIntakeError(
            "raw P2 record must be an object"
        )

    if spec.adapter == "messages":
        if set(raw) != {"messages"}:
            raise VN97P2CorpusIntakeError(
                "messages source record must be exactly {messages}"
            )
        messages = raw["messages"]
        if not isinstance(messages, list):
            raise VN97P2CorpusIntakeError(
                "messages must be an array"
            )
        return _validate_messages(messages)

    if spec.adapter == "dolly":
        required = {
            "instruction",
            "context",
            "response",
            "category",
        }
        if set(raw) != required:
            raise VN97P2CorpusIntakeError(
                "Dolly record fields are invalid"
            )
        instruction = _text(
            raw["instruction"],
            label="Dolly instruction",
        )
        context_raw = raw["context"]
        if not isinstance(context_raw, str):
            raise VN97P2CorpusIntakeError(
                "Dolly context must be text"
            )
        context = context_raw.strip()
        if context:
            if len(context.encode("utf-8")) > _MAX_MESSAGE_BYTES:
                raise VN97P2CorpusIntakeError(
                    "Dolly context exceeds byte bound"
                )
            user = f"{instruction}\n\nContext:\n{context}"
        else:
            user = instruction
        response = _text(
            raw["response"],
            label="Dolly response",
        )
        return _validate_messages(
            [
                {"role": "user", "content": user},
                {
                    "role": "assistant",
                    "content": response,
                },
            ]
        )

    if spec.adapter == "gsm8k":
        if set(raw) != {"question", "answer"}:
            raise VN97P2CorpusIntakeError(
                "GSM8K record fields are invalid"
            )
        question = _text(
            raw["question"],
            label="GSM8K question",
        )
        answer = _text(
            raw["answer"],
            label="GSM8K answer",
        )
        return _validate_messages(
            [
                {
                    "role": "user",
                    "content": question,
                },
                {
                    "role": "assistant",
                    "content": answer,
                },
            ]
        )

    raise VN97P2CorpusIntakeError(
        "P2 source adapter is unsupported"
    )


def has_identity_contamination(
    messages: Sequence[dict[str, str]],
) -> bool:
    text = "\n".join(
        message["content"]
        for message in messages
    ).casefold()
    return any(
        term in text
        for term in _IDENTITY_CONTAMINATION_TERMS
    )


def conversation_fingerprint(
    source_id: str,
    messages: Sequence[dict[str, str]],
) -> str:
    source_spec(source_id)
    digest = hashlib.sha256()
    digest.update(b"VN97P2REC1\0")
    digest.update(source_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(
        _canonical_json(
            {"messages": list(messages)}
        )
    )
    return digest.hexdigest()


def assigned_split(
    source_id: str,
    messages: Sequence[dict[str, str]],
) -> str:
    # Validate the declared source, but do not include it in split identity.
    # Identical conversations mirrored by two datasets must always land in
    # the same split so a mirror cannot create train/holdout leakage.
    source_spec(source_id)
    digest = hashlib.sha256()
    digest.update(b"VN97P2SPLIT1\0")
    digest.update(
        _canonical_json(
            {"messages": list(messages)}
        )
    )
    value = int.from_bytes(
        digest.digest()[:8],
        "big",
    ) % 1000
    if value < 50:
        return "release"
    if value < 100:
        return "validation"
    return "training"


@dataclass(frozen=True)
class VN97P2PreparedSource:
    spec: VN97P2SourceSpec
    raw_sha256: str
    raw_bytes: int
    input_records: int
    accepted_records: int
    rejected_records: int
    split_records: dict[
        str,
        tuple[tuple[str, tuple[dict[str, str], ...]], ...],
    ]

    def __post_init__(self) -> None:
        if (
            len(self.raw_sha256) != 64
            or any(
                ch not in "0123456789abcdef"
                for ch in self.raw_sha256
            )
        ):
            raise VN97P2CorpusIntakeError(
                "raw source SHA-256 is invalid"
            )
        if self.raw_bytes <= 0:
            raise VN97P2CorpusIntakeError(
                "raw source byte count must be positive"
            )
        if self.input_records <= 0:
            raise VN97P2CorpusIntakeError(
                "input_records must be positive"
            )
        if (
            self.accepted_records < 0
            or self.rejected_records < 0
            or self.accepted_records
            + self.rejected_records
            != self.input_records
        ):
            raise VN97P2CorpusIntakeError(
                "P2 source counts are inconsistent"
            )
        if set(self.split_records) != set(_SPLITS):
            raise VN97P2CorpusIntakeError(
                "P2 split map is invalid"
            )


def prepare_source(
    source_id: str,
    records: Iterable[object],
    *,
    raw_sha256: str,
    raw_bytes: int,
    max_records: int | None = None,
) -> VN97P2PreparedSource:
    spec = source_spec(source_id)
    selection_limit = (
        spec.max_records
        if max_records is None
        else max_records
    )
    if selection_limit <= 0:
        raise VN97P2CorpusIntakeError(
            "max_records override must be positive"
        )
    accepted: list[
        tuple[str, str, tuple[dict[str, str], ...]]
    ] = []
    seen: set[str] = set()
    input_records = 0
    rejected = 0

    for raw in records:
        input_records += 1
        try:
            messages = adapt_record(
                source_id,
                raw,
            )
        except VN97P2CorpusIntakeError:
            rejected += 1
            continue
        if has_identity_contamination(messages):
            rejected += 1
            continue
        fingerprint = conversation_fingerprint(
            source_id,
            messages,
        )
        if fingerprint in seen:
            rejected += 1
            continue
        seen.add(fingerprint)
        split = assigned_split(
            source_id,
            messages,
        )
        accepted.append(
            (fingerprint, split, messages)
        )

    if input_records <= 0:
        raise VN97P2CorpusIntakeError(
            "P2 source contains no records"
        )

    accepted.sort(key=lambda item: item[0])
    accepted = accepted[: selection_limit]
    accepted_fp = {
        item[0] for item in accepted
    }

    # Records that were valid/unique but fell beyond the deterministic source
    # quota are intentionally counted as rejected/not-selected.
    rejected = input_records - len(accepted)

    split_map: dict[
        str,
        list[
            tuple[
                str,
                tuple[dict[str, str], ...],
            ]
        ],
    ] = {
        "training": [],
        "validation": [],
        "release": [],
    }
    for fingerprint, split, messages in accepted:
        if fingerprint not in accepted_fp:
            raise AssertionError(
                "accepted fingerprint accounting mismatch"
            )
        split_map[split].append(
            (fingerprint, messages)
        )

    return VN97P2PreparedSource(
        spec=spec,
        raw_sha256=raw_sha256,
        raw_bytes=raw_bytes,
        input_records=input_records,
        accepted_records=len(accepted),
        rejected_records=rejected,
        split_records={
            split: tuple(split_map[split])
            for split in _SPLITS
        },
    )


def render_chat_jsonl(
    rows: Sequence[
        tuple[
            str,
            tuple[dict[str, str], ...],
        ]
    ],
) -> bytes:
    return b"".join(
        _canonical_json(
            {"messages": list(messages)}
        )
        + b"\n"
        for _, messages in rows
    )


def source_summary(
    prepared: Sequence[VN97P2PreparedSource],
) -> dict[str, object]:
    if not prepared:
        raise VN97P2CorpusIntakeError(
            "P2 source summary requires sources"
        )
    values = []
    for item in sorted(
        prepared,
        key=lambda value: value.spec.source_id,
    ):
        values.append(
            {
                "accepted_records": item.accepted_records,
                "adapter": item.spec.adapter,
                "input_records": item.input_records,
                "license": item.spec.license,
                "max_records": item.spec.max_records,
                "raw_bytes": item.raw_bytes,
                "raw_sha256": item.raw_sha256,
                "origin": item.spec.origin,
                "rejected_records": item.rejected_records,
                "source_id": item.spec.source_id,
                "splits": {
                    split: len(
                        item.split_records[split]
                    )
                    for split in _SPLITS
                },
            }
        )
    payload = {
        "schema": "VN97P2SRC1",
        "sources": values,
    }
    payload["summary_id"] = hashlib.sha256(
        b"VN97P2SRC1\0"
        + _canonical_json(payload)
    ).hexdigest()
    return payload
