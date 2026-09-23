from __future__ import annotations

import hashlib
import json
from typing import Sequence

from .training import VN97ChatMessage


class VN97DatasetSplitError(RuntimeError):
    pass


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def dataset_record_fingerprints(
    records: Sequence[object],
    *,
    mode: str,
) -> frozenset[str]:
    if mode not in {"text", "chat"}:
        raise ValueError("dataset mode must be text or chat")
    if not records:
        raise ValueError("dataset split must contain records")

    fingerprints: set[str] = set()
    for record in records:
        if mode == "text":
            if not isinstance(record, str) or not record:
                raise ValueError("text split records must be non-empty strings")
            canonical = _canonical_json({"text": record})
        else:
            if (
                not isinstance(record, tuple)
                or not record
                or any(not isinstance(message, VN97ChatMessage) for message in record)
            ):
                raise ValueError(
                    "chat split records must be non-empty VN97ChatMessage tuples"
                )
            canonical = _canonical_json(
                {
                    "messages": [
                        {
                            "content": message.content,
                            "role": message.role,
                        }
                        for message in record
                    ]
                }
            )

        digest = hashlib.sha256()
        digest.update(b"VN97DATAREC1\0")
        digest.update(mode.encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical)
        fingerprints.add(digest.hexdigest())

    return frozenset(fingerprints)


def require_disjoint_dataset_splits(
    training_records: Sequence[object],
    *,
    training_mode: str,
    validation_records: Sequence[object],
    validation_mode: str,
    release_records: Sequence[object],
    release_mode: str,
) -> None:
    training = dataset_record_fingerprints(
        training_records,
        mode=training_mode,
    )
    validation = dataset_record_fingerprints(
        validation_records,
        mode=validation_mode,
    )
    release = dataset_record_fingerprints(
        release_records,
        mode=release_mode,
    )

    checks = (
        ("training/validation", training.intersection(validation)),
        ("training/release", training.intersection(release)),
        ("validation/release", validation.intersection(release)),
    )
    for label, overlap in checks:
        if overlap:
            raise VN97DatasetSplitError(
                f"{label} dataset records overlap"
            )
