from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
import hashlib
import math
import os
from pathlib import Path
import struct
import tempfile
import time
from typing import Iterable, Sequence
import zlib


MAGIC = b"VN97MEM1"
VERSION = 1
_FILE_HEADER = struct.Struct("<8sII")
_FRAME_PREFIX = struct.Struct("<II")
_RECORD_FIXED = struct.Struct("<QQBBHfIIIQ32s")
MAX_VECTOR_DIM = 8192
MAX_SOURCE_BYTES = 1 << 20
MAX_CONTENT_BYTES = 16 << 20
MAX_FRAME_BODY = 64 << 20


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


class MemoryKind(IntEnum):
    EPISODIC = 1
    SEMANTIC = 2


class MemoryFormatError(ValueError):
    pass


class MemoryCorruptionError(MemoryFormatError):
    pass


class MemoryTruncatedTailError(MemoryFormatError):
    pass


@dataclass(frozen=True)
class MemoryRecord:
    record_id: int
    timestamp_ns: int
    kind: MemoryKind
    importance: float
    source: str
    content: str
    vector: tuple[float, ...] | None = None
    parent_id: int = 0
    content_sha256: bytes = b""

    def __post_init__(self) -> None:
        if self.record_id <= 0:
            raise ValueError("record_id must be positive")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if not math.isfinite(self.importance) or not 0.0 <= self.importance <= 1.0:
            raise ValueError("importance must be finite and in [0, 1]")
        if self.parent_id != 0 and (self.parent_id < 0 or self.parent_id >= self.record_id):
            raise ValueError("parent_id must be zero or reference an earlier record")
        if self.vector is not None and not all(math.isfinite(v) for v in self.vector):
            raise ValueError("memory vector must contain only finite values")
        source_bytes = self.source.encode("utf-8")
        content_bytes = self.content.encode("utf-8")
        if len(source_bytes) > MAX_SOURCE_BYTES:
            raise ValueError("source is too large")
        if len(content_bytes) > MAX_CONTENT_BYTES:
            raise ValueError("content is too large")
        digest = hashlib.sha256(content_bytes).digest()
        if self.content_sha256:
            if len(self.content_sha256) != 32 or self.content_sha256 != digest:
                raise ValueError("content_sha256 does not match content")
        else:
            object.__setattr__(self, "content_sha256", digest)


@dataclass(frozen=True)
class MemoryHit:
    record: MemoryRecord
    score: float
    semantic_score: float
    recency_score: float
    importance_score: float


@dataclass(frozen=True)
class RetentionPolicy:
    max_records: int | None = None
    max_age_ns: int | None = None
    min_importance: float = 0.0

    def __post_init__(self) -> None:
        if self.max_records is not None and self.max_records <= 0:
            raise ValueError("max_records must be positive when set")
        if self.max_age_ns is not None and self.max_age_ns <= 0:
            raise ValueError("max_age_ns must be positive when set")
        if not math.isfinite(self.min_importance) or not 0.0 <= self.min_importance <= 1.0:
            raise ValueError("min_importance must be finite and in [0, 1]")


@dataclass(frozen=True)
class WorkingMemoryItem:
    sequence: int
    timestamp_ns: int
    source: str
    content: str
    importance: float

    @property
    def utf8_bytes(self) -> int:
        return len(self.source.encode("utf-8")) + len(self.content.encode("utf-8"))


class WorkingMemory:
    """Bounded FIFO working memory with deterministic oldest-first eviction."""

    def __init__(self, *, max_items: int = 64, max_utf8_bytes: int = 64 * 1024) -> None:
        if max_items <= 0 or max_utf8_bytes <= 0:
            raise ValueError("working-memory limits must be positive")
        self.max_items = max_items
        self.max_utf8_bytes = max_utf8_bytes
        self._items: deque[WorkingMemoryItem] = deque()
        self._bytes = 0
        self._next_sequence = 1

    @property
    def total_utf8_bytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> tuple[WorkingMemoryItem, ...]:
        return tuple(self._items)

    def clear(self) -> None:
        self._items.clear()
        self._bytes = 0

    def push(
        self,
        content: str,
        *,
        source: str = "",
        importance: float = 0.5,
        timestamp_ns: int | None = None,
    ) -> WorkingMemoryItem:
        if not math.isfinite(importance) or not 0.0 <= importance <= 1.0:
            raise ValueError("importance must be finite and in [0, 1]")
        item = WorkingMemoryItem(
            sequence=self._next_sequence,
            timestamp_ns=time.time_ns() if timestamp_ns is None else timestamp_ns,
            source=source,
            content=content,
            importance=importance,
        )
        if item.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if item.utf8_bytes > self.max_utf8_bytes:
            raise ValueError("working-memory item exceeds byte budget")
        self._next_sequence += 1
        self._items.append(item)
        self._bytes += item.utf8_bytes
        while len(self._items) > self.max_items or self._bytes > self.max_utf8_bytes:
            evicted = self._items.popleft()
            self._bytes -= evicted.utf8_bytes
        return item


@dataclass(frozen=True)
class JournalScan:
    vector_dim: int
    records: tuple[MemoryRecord, ...]
    valid_bytes: int
    tail_truncated: bool


def _pack_record(record: MemoryRecord, vector_dim: int) -> bytes:
    source = record.source.encode("utf-8")
    content = record.content.encode("utf-8")
    vector = record.vector or ()
    if len(vector) not in (0, vector_dim):
        raise ValueError(
            f"vector length must be 0 or journal vector_dim={vector_dim}, got {len(vector)}"
        )
    vector_blob = struct.pack(f"<{len(vector)}f", *vector) if vector else b""
    fixed = _RECORD_FIXED.pack(
        record.record_id,
        record.timestamp_ns,
        int(record.kind),
        0,
        0,
        record.importance,
        len(source),
        len(content),
        len(vector),
        record.parent_id,
        record.content_sha256,
    )
    body = fixed + source + content + vector_blob
    if len(body) > MAX_FRAME_BODY:
        raise ValueError("memory record frame is too large")
    return _FRAME_PREFIX.pack(len(body), zlib.crc32(body) & 0xFFFFFFFF) + body


def _parse_record(body: bytes, vector_dim: int, previous_id: int) -> MemoryRecord:
    if len(body) < _RECORD_FIXED.size:
        raise MemoryCorruptionError("record body is shorter than fixed header")
    (
        record_id,
        timestamp_ns,
        kind_raw,
        flags,
        reserved,
        importance,
        source_len,
        content_len,
        vector_count,
        parent_id,
        digest,
    ) = _RECORD_FIXED.unpack_from(body)
    if flags != 0 or reserved != 0:
        raise MemoryCorruptionError("unsupported memory record flags")
    if record_id <= previous_id:
        raise MemoryCorruptionError("memory record IDs must be strictly increasing")
    if parent_id != 0 and parent_id >= record_id:
        raise MemoryCorruptionError("memory parent_id must reference an earlier record")
    try:
        kind = MemoryKind(kind_raw)
    except ValueError as exc:
        raise MemoryCorruptionError(f"unknown memory kind {kind_raw}") from exc
    if not math.isfinite(importance) or not 0.0 <= importance <= 1.0:
        raise MemoryCorruptionError("memory importance is invalid")
    if source_len > MAX_SOURCE_BYTES or content_len > MAX_CONTENT_BYTES:
        raise MemoryCorruptionError("memory text field exceeds format limits")
    if vector_count not in (0, vector_dim):
        raise MemoryCorruptionError("memory vector dimension does not match journal")
    expected = _RECORD_FIXED.size + source_len + content_len + vector_count * 4
    if expected != len(body):
        raise MemoryCorruptionError("memory record length does not match fields")

    offset = _RECORD_FIXED.size
    source_bytes = body[offset : offset + source_len]
    offset += source_len
    content_bytes = body[offset : offset + content_len]
    offset += content_len
    try:
        source = source_bytes.decode("utf-8")
        content = content_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MemoryCorruptionError("memory source/content is not valid UTF-8") from exc
    if hashlib.sha256(content_bytes).digest() != digest:
        raise MemoryCorruptionError("memory content SHA-256 does not match")

    vector: tuple[float, ...] | None = None
    if vector_count:
        values = struct.unpack_from(f"<{vector_count}f", body, offset)
        if not all(math.isfinite(v) for v in values):
            raise MemoryCorruptionError("memory vector contains non-finite values")
        vector = tuple(float(v) for v in values)
    return MemoryRecord(
        record_id=record_id,
        timestamp_ns=timestamp_ns,
        kind=kind,
        importance=float(importance),
        source=source,
        content=content,
        vector=vector,
        parent_id=parent_id,
        content_sha256=digest,
    )


def scan_journal(
    path: os.PathLike[str] | str,
    *,
    allow_torn_tail: bool = False,
) -> JournalScan:
    file_path = Path(path)
    file_size = file_path.stat().st_size
    with file_path.open("rb") as handle:
        header = handle.read(_FILE_HEADER.size)
        if len(header) != _FILE_HEADER.size:
            raise MemoryFormatError("journal is shorter than VN97MEM1 header")
        magic, version, vector_dim = _FILE_HEADER.unpack(header)
        if magic != MAGIC:
            raise MemoryFormatError("bad VN97MEM1 magic")
        if version != VERSION:
            raise MemoryFormatError(f"unsupported VN97MEM1 version: {version}")
        if vector_dim > MAX_VECTOR_DIM:
            raise MemoryFormatError("journal vector dimension exceeds format limit")

        records: list[MemoryRecord] = []
        valid_bytes = _FILE_HEADER.size
        previous_id = 0
        seen_ids: set[int] = set()
        tail_truncated = False

        while valid_bytes < file_size:
            frame_start = valid_bytes
            prefix = handle.read(_FRAME_PREFIX.size)
            if len(prefix) != _FRAME_PREFIX.size:
                if allow_torn_tail:
                    tail_truncated = True
                    break
                raise MemoryTruncatedTailError("journal ends inside frame prefix")
            body_len, expected_crc = _FRAME_PREFIX.unpack(prefix)
            if body_len < _RECORD_FIXED.size or body_len > MAX_FRAME_BODY:
                raise MemoryCorruptionError("memory frame length is invalid")
            body = handle.read(body_len)
            if len(body) != body_len:
                if allow_torn_tail:
                    tail_truncated = True
                    break
                raise MemoryTruncatedTailError("journal ends inside record body")
            if (zlib.crc32(body) & 0xFFFFFFFF) != expected_crc:
                raise MemoryCorruptionError("memory frame CRC32 mismatch")
            record = _parse_record(body, vector_dim, previous_id)
            if record.parent_id and record.parent_id not in seen_ids:
                raise MemoryCorruptionError(
                    "memory parent_id does not reference an existing record"
                )
            records.append(record)
            seen_ids.add(record.record_id)
            previous_id = record.record_id
            valid_bytes = frame_start + _FRAME_PREFIX.size + body_len

    return JournalScan(
        vector_dim=vector_dim,
        records=tuple(records),
        valid_bytes=valid_bytes,
        tail_truncated=tail_truncated,
    )


class MemoryJournal:
    """Single-writer append-only VN97MEM1 journal reference implementation."""

    def __init__(self, path: Path, scan: JournalScan) -> None:
        self.path = path
        self.vector_dim = scan.vector_dim
        self._records = list(scan.records)
        self._last_id = self._records[-1].record_id if self._records else 0

    @classmethod
    def create(
        cls,
        path: os.PathLike[str] | str,
        *,
        vector_dim: int,
        overwrite: bool = False,
    ) -> "MemoryJournal":
        if vector_dim < 0 or vector_dim > MAX_VECTOR_DIM:
            raise ValueError(f"vector_dim must be in [0, {MAX_VECTOR_DIM}]")
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("wb" if overwrite else "xb") as handle:
            handle.write(_FILE_HEADER.pack(MAGIC, VERSION, vector_dim))
            handle.flush()
            os.fsync(handle.fileno())
        return cls.open(file_path)

    @classmethod
    def open(
        cls,
        path: os.PathLike[str] | str,
        *,
        recover_torn_tail: bool = False,
    ) -> "MemoryJournal":
        file_path = Path(path)
        scan = scan_journal(file_path, allow_torn_tail=recover_torn_tail)
        if recover_torn_tail and scan.tail_truncated:
            with file_path.open("r+b") as handle:
                handle.truncate(scan.valid_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            scan = scan_journal(file_path)
        return cls(file_path, scan)

    def records(self) -> tuple[MemoryRecord, ...]:
        return tuple(self._records)

    def append(
        self,
        kind: MemoryKind,
        content: str,
        *,
        source: str = "",
        importance: float = 0.5,
        vector: Sequence[float] | None = None,
        parent_id: int = 0,
        timestamp_ns: int | None = None,
        durable: bool = True,
    ) -> MemoryRecord:
        if parent_id < 0 or parent_id > self._last_id:
            raise ValueError("parent_id must reference an existing earlier record")
        vector_tuple = None if vector is None else tuple(_f32(v) for v in vector)
        record = MemoryRecord(
            record_id=self._last_id + 1,
            timestamp_ns=time.time_ns() if timestamp_ns is None else timestamp_ns,
            kind=MemoryKind(kind),
            importance=_f32(importance),
            source=source,
            content=content,
            vector=vector_tuple,
            parent_id=parent_id,
        )
        frame = _pack_record(record, self.vector_dim)
        with self.path.open("ab") as handle:
            handle.write(frame)
            handle.flush()
            if durable:
                os.fsync(handle.fileno())
        self._records.append(record)
        self._last_id = record.record_id
        return record

    def retrieve(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 5,
        kinds: Iterable[MemoryKind] | None = None,
        semantic_weight: float = 1.0,
        recency_weight: float = 0.0,
        importance_weight: float = 0.0,
        now_ns: int | None = None,
        recency_half_life_ns: int = 86_400_000_000_000,
    ) -> tuple[MemoryHit, ...]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        weights = (semantic_weight, recency_weight, importance_weight)
        if not all(math.isfinite(w) and w >= 0.0 for w in weights):
            raise ValueError("retrieval weights must be finite and non-negative")
        if sum(weights) <= 0.0:
            raise ValueError("at least one retrieval weight must be positive")
        query = tuple(float(v) for v in query_vector)
        if len(query) != self.vector_dim:
            raise ValueError(
                f"query vector must have length {self.vector_dim}, got {len(query)}"
            )
        if not all(math.isfinite(v) for v in query):
            raise ValueError("query vector must contain only finite values")
        query_norm = math.sqrt(sum(v * v for v in query))
        if query_norm == 0.0:
            raise ValueError("query vector norm must be non-zero")
        if recency_half_life_ns <= 0:
            raise ValueError("recency_half_life_ns must be positive")
        allowed = None if kinds is None else {MemoryKind(kind) for kind in kinds}
        if now_ns is None:
            now_ns = max((r.timestamp_ns for r in self._records), default=0)
        if now_ns < 0:
            raise ValueError("now_ns must be non-negative")

        hits: list[MemoryHit] = []
        for record in self._records:
            if record.vector is None:
                continue
            if allowed is not None and record.kind not in allowed:
                continue
            norm = math.sqrt(sum(v * v for v in record.vector))
            semantic = 0.0 if norm == 0.0 else sum(
                a * b for a, b in zip(query, record.vector)
            ) / (query_norm * norm)
            semantic = max(-1.0, min(1.0, semantic))
            age = max(0, now_ns - record.timestamp_ns)
            recency = math.exp(-math.log(2.0) * age / recency_half_life_ns)
            score = (
                semantic_weight * semantic
                + recency_weight * recency
                + importance_weight * record.importance
            )
            hits.append(
                MemoryHit(
                    record=record,
                    score=score,
                    semantic_score=semantic,
                    recency_score=recency,
                    importance_score=record.importance,
                )
            )
        hits.sort(
            key=lambda hit: (
                -hit.score,
                -hit.record.timestamp_ns,
                -hit.record.record_id,
            )
        )
        return tuple(hits[:top_k])

    def compact(
        self,
        policy: RetentionPolicy,
        *,
        now_ns: int | None = None,
    ) -> tuple[int, int]:
        """Atomically rewrite retained records while preserving IDs and parent ancestry."""
        if not self._records:
            return 0, 0
        if now_ns is None:
            now_ns = max(record.timestamp_ns for record in self._records)
        if now_ns < 0:
            raise ValueError("now_ns must be non-negative")

        by_id = {record.record_id: record for record in self._records}
        keep = {
            record.record_id
            for record in self._records
            if record.importance >= policy.min_importance
            and (
                policy.max_age_ns is None
                or max(0, now_ns - record.timestamp_ns) <= policy.max_age_ns
            )
        }
        # Preserve the highest ID so compaction cannot cause ID reuse.
        keep.add(self._records[-1].record_id)

        if policy.max_records is not None and len(keep) > policy.max_records:
            ranked = sorted(
                (by_id[record_id] for record_id in keep),
                key=lambda record: (
                    -record.importance,
                    -record.timestamp_ns,
                    -record.record_id,
                ),
            )
            keep = {record.record_id for record in ranked[: policy.max_records]}
            keep.add(self._records[-1].record_id)

        # Preserve provenance ancestry even when it exceeds the target max_records.
        pending = list(keep)
        while pending:
            record = by_id[pending.pop()]
            if record.parent_id and record.parent_id not in keep:
                keep.add(record.parent_id)
                pending.append(record.parent_id)

        retained = [record for record in self._records if record.record_id in keep]
        removed = len(self._records) - len(retained)

        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(_FILE_HEADER.pack(MAGIC, VERSION, self.vector_dim))
                for record in retained:
                    handle.write(_pack_record(record, self.vector_dim))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
            try:
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
            except (AttributeError, OSError):
                directory_fd = None
            if directory_fd is not None:
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        scan = scan_journal(self.path)
        self._records = list(scan.records)
        self._last_id = self._records[-1].record_id if self._records else 0
        return len(retained), removed
