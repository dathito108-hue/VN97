from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

from .dataset_split import (
    VN97DatasetSplitError,
    dataset_record_fingerprints,
)
from .training import VN97ChatMessage


class VN97ProductionCorpusError(RuntimeError):
    pass


_SPLITS = ("training", "validation", "release")
_MODES = ("text", "chat")
_MAX_SOURCE_ID_BYTES = 128
_MAX_ORIGIN_BYTES = 1024
_MAX_LICENSE_BYTES = 256


def _bounded_text(value: object, *, label: str, max_bytes: int) -> str:
    if not isinstance(value, str):
        raise VN97ProductionCorpusError(f"{label} must be a string")
    encoded = value.encode("utf-8")
    if not encoded or len(encoded) > max_bytes:
        raise VN97ProductionCorpusError(
            f"{label} must be 1..{max_bytes} UTF-8 bytes"
        )
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_record(record: object, *, mode: str) -> bytes:
    if mode == "text":
        if not isinstance(record, str) or not record:
            raise VN97ProductionCorpusError(
                "text corpus records must be non-empty strings"
            )
        return _canonical_json({"text": record})

    if mode != "chat":
        raise VN97ProductionCorpusError("corpus mode must be text or chat")
    if (
        not isinstance(record, tuple)
        or not record
        or any(not isinstance(item, VN97ChatMessage) for item in record)
    ):
        raise VN97ProductionCorpusError(
            "chat corpus records must be non-empty VN97ChatMessage tuples"
        )
    return _canonical_json(
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


@dataclass(frozen=True)
class VN97CorpusSource:
    source_id: str
    origin: str
    license: str
    split: str
    mode: str
    content_sha256: str
    content_bytes: int
    input_records: int
    kept_records: int
    duplicate_records: int

    def __post_init__(self) -> None:
        _bounded_text(
            self.source_id,
            label="source_id",
            max_bytes=_MAX_SOURCE_ID_BYTES,
        )
        _bounded_text(
            self.origin,
            label="origin",
            max_bytes=_MAX_ORIGIN_BYTES,
        )
        _bounded_text(
            self.license,
            label="license",
            max_bytes=_MAX_LICENSE_BYTES,
        )
        if self.split not in _SPLITS:
            raise VN97ProductionCorpusError("source split is invalid")
        if self.mode not in _MODES:
            raise VN97ProductionCorpusError("source mode is invalid")
        if (
            len(self.content_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in self.content_sha256)
        ):
            raise VN97ProductionCorpusError(
                "source content_sha256 must be lowercase SHA-256"
            )
        if self.content_bytes <= 0 or self.input_records <= 0:
            raise VN97ProductionCorpusError(
                "source content bytes/records must be positive"
            )
        if not 0 <= self.kept_records <= self.input_records:
            raise VN97ProductionCorpusError("source kept_records is invalid")
        if self.duplicate_records != self.input_records - self.kept_records:
            raise VN97ProductionCorpusError(
                "source duplicate_records does not match input-kept"
            )

    def canonical_object(self) -> dict[str, object]:
        return {
            "content_bytes": self.content_bytes,
            "content_sha256": self.content_sha256,
            "duplicate_records": self.duplicate_records,
            "input_records": self.input_records,
            "kept_records": self.kept_records,
            "license": self.license,
            "mode": self.mode,
            "origin": self.origin,
            "source_id": self.source_id,
            "split": self.split,
        }


@dataclass(frozen=True)
class VN97CorpusSplit:
    mode: str
    records: int
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise VN97ProductionCorpusError("split mode is invalid")
        if self.records <= 0 or self.bytes <= 0:
            raise VN97ProductionCorpusError(
                "split records/bytes must be positive"
            )
        if (
            len(self.sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in self.sha256)
        ):
            raise VN97ProductionCorpusError(
                "split sha256 must be lowercase SHA-256"
            )

    def canonical_object(self) -> dict[str, object]:
        return {
            "bytes": self.bytes,
            "mode": self.mode,
            "records": self.records,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class VN97ProductionCorpusManifest:
    profile_id: str
    sources: tuple[VN97CorpusSource, ...]
    training: VN97CorpusSplit
    validation: VN97CorpusSplit
    release: VN97CorpusSplit
    manifest_id: str

    def _identity_object(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "schema": "VN97CORPUS1",
            "sources": [
                item.canonical_object()
                for item in self.sources
            ],
            "splits": {
                "release": self.release.canonical_object(),
                "training": self.training.canonical_object(),
                "validation": self.validation.canonical_object(),
            },
        }

    def __post_init__(self) -> None:
        _bounded_text(
            self.profile_id,
            label="profile_id",
            max_bytes=128,
        )
        if not self.sources:
            raise VN97ProductionCorpusError(
                "production corpus must contain at least one source"
            )
        ids = [item.source_id for item in self.sources]
        if len(ids) != len(set(ids)):
            raise VN97ProductionCorpusError(
                "production corpus source IDs must be unique"
            )
        expected = hashlib.sha256(
            b"VN97CORPUS1\0" + _canonical_json(self._identity_object())
        ).hexdigest()
        if self.manifest_id != expected:
            raise VN97ProductionCorpusError(
                "VN97CORPUS1 manifest identity mismatch"
            )

    def to_bytes(self) -> bytes:
        payload = self._identity_object()
        payload["manifest_id"] = self.manifest_id
        return _canonical_json(payload) + b"\n"


@dataclass(frozen=True)
class VN97CorpusPreparedSplit:
    mode: str
    records: tuple[bytes, ...]

    @property
    def bytes(self) -> bytes:
        return b"".join(record + b"\n" for record in self.records)

    @property
    def split(self) -> VN97CorpusSplit:
        blob = self.bytes
        return VN97CorpusSplit(
            mode=self.mode,
            records=len(self.records),
            bytes=len(blob),
            sha256=hashlib.sha256(blob).hexdigest(),
        )


def prepare_corpus(
    source_rows: Sequence[
        tuple[
            str,
            str,
            str,
            str,
            str,
            bytes,
            Sequence[object],
        ]
    ],
    *,
    profile_id: str,
) -> tuple[
    VN97ProductionCorpusManifest,
    dict[str, VN97CorpusPreparedSplit],
]:
    if not source_rows:
        raise VN97ProductionCorpusError(
            "production corpus source list is empty"
        )

    seen_source_ids: set[str] = set()
    split_modes: dict[str, str] = {}
    split_records: dict[str, dict[str, bytes]] = {
        key: {} for key in _SPLITS
    }
    source_objects: list[VN97CorpusSource] = []
    global_fp_to_split: dict[str, str] = {}

    for (
        source_id,
        origin,
        license_name,
        split,
        mode,
        raw_bytes,
        records,
    ) in source_rows:
        if source_id in seen_source_ids:
            raise VN97ProductionCorpusError(
                "production corpus source IDs must be unique"
            )
        seen_source_ids.add(source_id)
        _bounded_text(
            source_id,
            label="source_id",
            max_bytes=_MAX_SOURCE_ID_BYTES,
        )
        _bounded_text(
            origin,
            label="origin",
            max_bytes=_MAX_ORIGIN_BYTES,
        )
        _bounded_text(
            license_name,
            label="license",
            max_bytes=_MAX_LICENSE_BYTES,
        )
        if split not in _SPLITS:
            raise VN97ProductionCorpusError("source split is invalid")
        if mode not in _MODES:
            raise VN97ProductionCorpusError("source mode is invalid")
        if not raw_bytes:
            raise VN97ProductionCorpusError("source bytes must be non-empty")
        if not records:
            raise VN97ProductionCorpusError("source records must be non-empty")

        existing_mode = split_modes.setdefault(split, mode)
        if existing_mode != mode:
            raise VN97ProductionCorpusError(
                f"{split} sources must use one dataset mode"
            )

        fingerprints = tuple(
            dataset_record_fingerprints([record], mode=mode).pop()
            for record in records
        )
        for fp in fingerprints:
            prior_split = global_fp_to_split.get(fp)
            if prior_split is not None and prior_split != split:
                raise VN97DatasetSplitError(
                    f"{prior_split}/{split} dataset records overlap"
                )
            global_fp_to_split[fp] = split

        kept_before = len(split_records[split])
        for fp, record in zip(fingerprints, records):
            split_records[split].setdefault(
                fp,
                _canonical_record(record, mode=mode),
            )
        kept_after = len(split_records[split])
        kept = kept_after - kept_before

        source_objects.append(
            VN97CorpusSource(
                source_id=source_id,
                origin=origin,
                license=license_name,
                split=split,
                mode=mode,
                content_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                content_bytes=len(raw_bytes),
                input_records=len(records),
                kept_records=kept,
                duplicate_records=len(records) - kept,
            )
        )

    missing = [split for split in _SPLITS if not split_records[split]]
    if missing:
        raise VN97ProductionCorpusError(
            "production corpus is missing split(s): " + ",".join(missing)
        )

    prepared: dict[str, VN97CorpusPreparedSplit] = {}
    for split in _SPLITS:
        ordered = tuple(
            split_records[split][fp]
            for fp in sorted(split_records[split])
        )
        prepared[split] = VN97CorpusPreparedSplit(
            mode=split_modes[split],
            records=ordered,
        )

    source_tuple = tuple(
        sorted(
            source_objects,
            key=lambda item: (item.split, item.source_id),
        )
    )
    identity = {
        "profile_id": profile_id,
        "schema": "VN97CORPUS1",
        "sources": [item.canonical_object() for item in source_tuple],
        "splits": {
            split: prepared[split].split.canonical_object()
            for split in ("release", "training", "validation")
        },
    }
    manifest_id = hashlib.sha256(
        b"VN97CORPUS1\0" + _canonical_json(identity)
    ).hexdigest()
    manifest = VN97ProductionCorpusManifest(
        profile_id=profile_id,
        sources=source_tuple,
        training=prepared["training"].split,
        validation=prepared["validation"].split,
        release=prepared["release"].split,
        manifest_id=manifest_id,
    )
    return manifest, prepared
