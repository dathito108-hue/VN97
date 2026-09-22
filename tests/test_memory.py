from __future__ import annotations

from pathlib import Path
import struct
import zlib

import pytest

from vn97.memory import (
    MAGIC,
    MemoryCorruptionError,
    MemoryJournal,
    MemoryKind,
    MemoryTruncatedTailError,
    RetentionPolicy,
    WorkingMemory,
    scan_journal,
)


def test_working_memory_evicts_oldest_by_item_limit():
    memory = WorkingMemory(max_items=2, max_utf8_bytes=1000)
    memory.push("one", timestamp_ns=1)
    memory.push("two", timestamp_ns=2)
    memory.push("three", timestamp_ns=3)
    assert [item.content for item in memory.items()] == ["two", "three"]


def test_working_memory_evicts_oldest_by_byte_budget():
    memory = WorkingMemory(max_items=10, max_utf8_bytes=7)
    memory.push("abc", timestamp_ns=1)
    memory.push("def", timestamp_ns=2)
    memory.push("ghi", timestamp_ns=3)
    assert [item.content for item in memory.items()] == ["def", "ghi"]
    assert memory.total_utf8_bytes == 6


def test_journal_roundtrip_provenance_vector_and_digest(tmp_path: Path):
    path = tmp_path / "memory.vn97mem"
    journal = MemoryJournal.create(path, vector_dim=3)
    first = journal.append(
        MemoryKind.EPISODIC,
        "User asked about VN97 memory",
        source="chat:42",
        importance=0.7,
        vector=[1.0, 0.0, 0.0],
        timestamp_ns=100,
    )
    second = journal.append(
        MemoryKind.SEMANTIC,
        "VN97 uses an append-only sovereign memory journal",
        source="derived:summary",
        importance=0.9,
        vector=[0.9, 0.1, 0.0],
        parent_id=first.record_id,
        timestamp_ns=200,
    )

    reopened = MemoryJournal.open(path)
    assert reopened.records() == (first, second)
    assert len(first.content_sha256) == 32
    assert second.parent_id == first.record_id
    assert path.read_bytes().startswith(MAGIC)


def test_retrieval_is_deterministic_and_filterable(tmp_path: Path):
    path = tmp_path / "memory.vn97mem"
    journal = MemoryJournal.create(path, vector_dim=3)
    journal.append(
        MemoryKind.EPISODIC,
        "old exact",
        vector=[1.0, 0.0, 0.0],
        importance=0.2,
        timestamp_ns=100,
    )
    journal.append(
        MemoryKind.SEMANTIC,
        "new near",
        vector=[0.9, 0.1, 0.0],
        importance=1.0,
        timestamp_ns=200,
    )
    journal.append(
        MemoryKind.SEMANTIC,
        "opposite",
        vector=[-1.0, 0.0, 0.0],
        importance=1.0,
        timestamp_ns=300,
    )

    hits = journal.retrieve([1.0, 0.0, 0.0], top_k=3)
    assert [hit.record.content for hit in hits] == [
        "old exact",
        "new near",
        "opposite",
    ]

    hybrid = journal.retrieve(
        [1.0, 0.0, 0.0],
        top_k=2,
        kinds=[MemoryKind.SEMANTIC],
        semantic_weight=1.0,
        recency_weight=0.2,
        importance_weight=0.2,
        now_ns=300,
        recency_half_life_ns=100,
    )
    assert hybrid[0].record.content == "new near"
    assert all(hit.record.kind == MemoryKind.SEMANTIC for hit in hybrid)


def test_torn_tail_recovery_only_truncates_incomplete_frame(tmp_path: Path):
    path = tmp_path / "memory.vn97mem"
    journal = MemoryJournal.create(path, vector_dim=2)
    journal.append(
        MemoryKind.EPISODIC,
        "safe",
        vector=[1.0, 0.0],
        timestamp_ns=1,
    )
    valid_size = path.stat().st_size
    with path.open("ab") as handle:
        handle.write(b"\x40\x00\x00")

    with pytest.raises(MemoryTruncatedTailError):
        MemoryJournal.open(path)

    recovered = MemoryJournal.open(path, recover_torn_tail=True)
    assert [record.content for record in recovered.records()] == ["safe"]
    assert path.stat().st_size == valid_size


def test_crc_corruption_fails_closed_even_with_recovery_enabled(tmp_path: Path):
    path = tmp_path / "memory.vn97mem"
    journal = MemoryJournal.create(path, vector_dim=2)
    journal.append(
        MemoryKind.EPISODIC,
        "integrity",
        vector=[1.0, 0.0],
        timestamp_ns=1,
    )
    blob = bytearray(path.read_bytes())
    blob[-1] ^= 0x01
    path.write_bytes(blob)

    with pytest.raises(MemoryCorruptionError):
        MemoryJournal.open(path, recover_torn_tail=True)


def test_content_digest_corruption_fails_after_crc_is_recomputed(tmp_path: Path):
    path = tmp_path / "memory.vn97mem"
    journal = MemoryJournal.create(path, vector_dim=0)
    journal.append(
        MemoryKind.SEMANTIC,
        "hello",
        source="unit",
        timestamp_ns=1,
    )
    blob = bytearray(path.read_bytes())
    body_len, _ = struct.unpack_from("<II", blob, 16)
    body_start = 24
    content_start = body_start + 76 + 4
    blob[content_start] ^= 0x01
    body = bytes(blob[body_start : body_start + body_len])
    struct.pack_into("<I", blob, 20, zlib.crc32(body) & 0xFFFFFFFF)
    path.write_bytes(blob)

    with pytest.raises(MemoryCorruptionError, match="SHA-256"):
        scan_journal(path)


def test_retention_compaction_preserves_latest_and_parent_chain(tmp_path: Path):
    path = tmp_path / "memory.vn97mem"
    journal = MemoryJournal.create(path, vector_dim=2)
    root = journal.append(
        MemoryKind.EPISODIC,
        "root",
        importance=0.1,
        vector=[1.0, 0.0],
        timestamp_ns=100,
    )
    child = journal.append(
        MemoryKind.SEMANTIC,
        "important child",
        importance=1.0,
        vector=[1.0, 0.0],
        parent_id=root.record_id,
        timestamp_ns=200,
    )
    journal.append(
        MemoryKind.EPISODIC,
        "discard me",
        importance=0.0,
        vector=[0.0, 1.0],
        timestamp_ns=300,
    )
    latest = journal.append(
        MemoryKind.EPISODIC,
        "latest",
        importance=0.0,
        vector=[0.0, 1.0],
        timestamp_ns=400,
    )

    retained, removed = journal.compact(
        RetentionPolicy(max_records=1, min_importance=0.9),
        now_ns=400,
    )
    assert retained == 3
    assert removed == 1
    records = journal.records()
    assert [record.record_id for record in records] == [
        root.record_id,
        child.record_id,
        latest.record_id,
    ]

    appended = journal.append(
        MemoryKind.EPISODIC,
        "after compact",
        vector=[1.0, 1.0],
        timestamp_ns=500,
    )
    assert appended.record_id == 5


def test_invalid_vector_dimensions_and_parent_fail_before_write(tmp_path: Path):
    path = tmp_path / "memory.vn97mem"
    journal = MemoryJournal.create(path, vector_dim=3)
    size = path.stat().st_size
    with pytest.raises(ValueError):
        journal.append(
            MemoryKind.SEMANTIC,
            "bad vector",
            vector=[1.0, 2.0],
            timestamp_ns=1,
        )
    with pytest.raises(ValueError):
        journal.append(
            MemoryKind.SEMANTIC,
            "bad parent",
            vector=[1.0, 2.0, 3.0],
            parent_id=1,
            timestamp_ns=1,
        )
    assert path.stat().st_size == size


def test_missing_parent_reference_is_corruption(tmp_path: Path):
    path = tmp_path / "memory.vn97mem"
    journal = MemoryJournal.create(path, vector_dim=0)
    journal.append(MemoryKind.EPISODIC, "one", timestamp_ns=1)
    journal.append(
        MemoryKind.SEMANTIC,
        "two",
        parent_id=1,
        timestamp_ns=2,
    )
    blob = bytearray(path.read_bytes())
    first_body_len = struct.unpack_from("<I", blob, 16)[0]
    second_prefix = 16 + 8 + first_body_len
    second_body_len = struct.unpack_from("<I", blob, second_prefix)[0]
    second_body = second_prefix + 8
    struct.pack_into("<Q", blob, second_body + 36, 9)
    body = bytes(blob[second_body : second_body + second_body_len])
    struct.pack_into(
        "<I",
        blob,
        second_prefix + 4,
        zlib.crc32(body) & 0xFFFFFFFF,
    )
    path.write_bytes(blob)
    with pytest.raises(MemoryCorruptionError, match="parent_id"):
        scan_journal(path)
