from __future__ import annotations

import ctypes
import math
from pathlib import Path
import sys
import tempfile

from vn97.memory import (
    MemoryJournal,
    MemoryKind,
)


U8P = ctypes.POINTER(ctypes.c_uint8)
F32P = ctypes.POINTER(ctypes.c_float)
U64P = ctypes.POINTER(ctypes.c_uint64)
SIZEP = ctypes.POINTER(ctypes.c_size_t)


class NativeStore:
    def __init__(self, lib, path: Path, vector_dim: int) -> None:
        self.lib = lib
        self.handle = ctypes.c_void_p()
        status = lib.vn97_memory_store_create(
            str(path).encode(),
            vector_dim,
            0,
            ctypes.byref(self.handle),
        )
        assert status == 0
        assert self.handle.value is not None

    def close(self) -> None:
        if self.handle.value is not None:
            self.lib.vn97_memory_store_close(self.handle)
            self.handle = ctypes.c_void_p()

    def append(
        self,
        kind: int,
        timestamp_ns: int,
        importance: float,
        source: bytes,
        content: bytes,
        vector: list[float] | None,
        parent_id: int = 0,
    ) -> int:
        source_buf = (
            (ctypes.c_uint8 * len(source)).from_buffer_copy(source)
            if source
            else None
        )
        content_buf = (
            (ctypes.c_uint8 * len(content)).from_buffer_copy(content)
            if content
            else None
        )
        vector_buf = (
            (ctypes.c_float * len(vector))(*vector)
            if vector is not None
            else None
        )
        record_id = ctypes.c_uint64()
        status = self.lib.vn97_memory_store_append(
            self.handle,
            kind,
            timestamp_ns,
            importance,
            source_buf,
            len(source),
            content_buf,
            len(content),
            vector_buf,
            0 if vector is None else len(vector),
            parent_id,
            1,
            ctypes.byref(record_id),
        )
        assert status == 0
        return record_id.value


def configure(lib) -> None:
    lib.vn97_memory_store_create.argtypes = [
        ctypes.c_char_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.vn97_memory_store_create.restype = ctypes.c_int
    lib.vn97_memory_store_open.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.vn97_memory_store_open.restype = ctypes.c_int
    lib.vn97_memory_store_close.argtypes = [ctypes.c_void_p]
    lib.vn97_memory_store_close.restype = None
    lib.vn97_memory_store_append.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint8,
        ctypes.c_uint64,
        ctypes.c_float,
        U8P,
        ctypes.c_size_t,
        U8P,
        ctypes.c_size_t,
        F32P,
        ctypes.c_size_t,
        ctypes.c_uint64,
        ctypes.c_int,
        U64P,
    ]
    lib.vn97_memory_store_append.restype = ctypes.c_int
    lib.vn97_memory_store_retrieve.argtypes = [
        ctypes.c_void_p,
        F32P,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_uint64,
        ctypes.c_uint64,
        U64P,
        F32P,
        F32P,
        F32P,
        F32P,
        ctypes.c_size_t,
        SIZEP,
    ]
    lib.vn97_memory_store_retrieve.restype = ctypes.c_int
    lib.vn97_memory_store_compact.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint64,
        ctypes.c_float,
        ctypes.c_uint64,
        SIZEP,
        SIZEP,
    ]
    lib.vn97_memory_store_compact.restype = ctypes.c_int
    lib.vn97_memory_store_stats.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        SIZEP,
        U64P,
        SIZEP,
    ]
    lib.vn97_memory_store_stats.restype = ctypes.c_int


def retrieve(lib, handle, query: list[float], top_k: int):
    query_buf = (ctypes.c_float * len(query))(*query)
    ids = (ctypes.c_uint64 * top_k)()
    scores = (ctypes.c_float * top_k)()
    semantic = (ctypes.c_float * top_k)()
    recency = (ctypes.c_float * top_k)()
    importance = (ctypes.c_float * top_k)()
    count = ctypes.c_size_t()
    status = lib.vn97_memory_store_retrieve(
        handle,
        query_buf,
        len(query),
        top_k,
        0x3,
        1.0,
        0.2,
        0.2,
        400,
        100,
        ids,
        scores,
        semantic,
        recency,
        importance,
        top_k,
        ctypes.byref(count),
    )
    assert status == 0
    return [
        (
            ids[i],
            scores[i],
            semantic[i],
            recency[i],
            importance[i],
        )
        for i in range(count.value)
    ]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: memory_store_python_equivalence.py <shared-library>"
        )
    lib = ctypes.CDLL(str(Path(sys.argv[1]).resolve()))
    configure(lib)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "native.vn97mem"
        store = NativeStore(lib, path, 3)
        try:
            root = store.append(
                1, 100, 0.2, b"chat:1", b"root exact", [1.0, 0.0, 0.0]
            )
            child = store.append(
                2,
                200,
                1.0,
                b"derived",
                b"important child",
                [0.9, 0.1, 0.0],
                root,
            )
            discarded = store.append(
                2, 300, 0.0, b"test", b"opposite", [-1.0, 0.0, 0.0]
            )
            latest = store.append(
                1, 400, 0.0, b"test", b"latest", [0.0, 1.0, 0.0]
            )
            assert (root, child, discarded, latest) == (1, 2, 3, 4)

            vector_dim = ctypes.c_uint32()
            record_count = ctypes.c_size_t()
            last_record_id = ctypes.c_uint64()
            valid_bytes = ctypes.c_size_t()
            assert (
                lib.vn97_memory_store_stats(
                    store.handle,
                    ctypes.byref(vector_dim),
                    ctypes.byref(record_count),
                    ctypes.byref(last_record_id),
                    ctypes.byref(valid_bytes),
                )
                == 0
            )
            assert vector_dim.value == 3
            assert record_count.value == 4
            assert last_record_id.value == 4
            assert valid_bytes.value == path.stat().st_size

            python_journal = MemoryJournal.open(path)
            records = python_journal.records()
            assert [record.record_id for record in records] == [1, 2, 3, 4]
            assert records[1].parent_id == 1
            assert records[1].content == "important child"
            assert records[1].source == "derived"
            assert records[1].vector is not None
            assert len(records[1].content_sha256) == 32

            native_hits = retrieve(lib, store.handle, [1.0, 0.0, 0.0], 4)
            python_hits = python_journal.retrieve(
                [1.0, 0.0, 0.0],
                top_k=4,
                semantic_weight=1.0,
                recency_weight=0.2,
                importance_weight=0.2,
                now_ns=400,
                recency_half_life_ns=100,
            )
            assert [hit[0] for hit in native_hits] == [
                hit.record.record_id for hit in python_hits
            ]
            for native, oracle in zip(native_hits, python_hits):
                assert math.isclose(
                    native[1], oracle.score, rel_tol=3e-5, abs_tol=3e-6
                )
                assert math.isclose(
                    native[2], oracle.semantic_score, rel_tol=3e-5, abs_tol=3e-6
                )
                assert math.isclose(
                    native[3], oracle.recency_score, rel_tol=3e-5, abs_tol=3e-6
                )
                assert math.isclose(
                    native[4], oracle.importance_score, rel_tol=0, abs_tol=2e-7
                )

            query_buf = (ctypes.c_float * 3)(1.0, 0.0, 0.0)
            one_id = (ctypes.c_uint64 * 1)()
            one_score = (ctypes.c_float * 1)()
            count = ctypes.c_size_t()
            status = lib.vn97_memory_store_retrieve(
                store.handle,
                query_buf,
                3,
                4,
                0x3,
                1.0,
                0.0,
                0.0,
                400,
                100,
                one_id,
                one_score,
                one_score,
                one_score,
                one_score,
                1,
                ctypes.byref(count),
            )
            assert status == 11
            assert count.value == 4

            retained = ctypes.c_size_t()
            removed = ctypes.c_size_t()
            assert (
                lib.vn97_memory_store_compact(
                    store.handle,
                    1,
                    0,
                    0.9,
                    400,
                    ctypes.byref(retained),
                    ctypes.byref(removed),
                )
                == 0
            )
            assert (retained.value, removed.value) == (3, 1)
            compacted = MemoryJournal.open(path)
            assert [record.record_id for record in compacted.records()] == [1, 2, 4]

            after = store.append(
                1, 500, 0.5, b"test", b"after compact", [1.0, 1.0, 0.0]
            )
            assert after == 5
        finally:
            store.close()

        reopened = MemoryJournal.open(path)
        assert [record.record_id for record in reopened.records()] == [1, 2, 4, 5]
        assert reopened.records()[1].parent_id == 1

        recovery_path = Path(directory) / "recover.vn97mem"
        oracle = MemoryJournal.create(recovery_path, vector_dim=2)
        oracle.append(
            MemoryKind.EPISODIC,
            "safe",
            source="python",
            vector=[1.0, 0.0],
            timestamp_ns=1,
        )
        valid_size = recovery_path.stat().st_size
        with recovery_path.open("ab") as handle:
            handle.write(b"\x20\x00\x00")

        handle = ctypes.c_void_p()
        status = lib.vn97_memory_store_open(
            str(recovery_path).encode(),
            1,
            ctypes.byref(handle),
        )
        assert status == 0
        lib.vn97_memory_store_close(handle)
        assert recovery_path.stat().st_size == valid_size
        assert [
            record.content
            for record in MemoryJournal.open(recovery_path).records()
        ] == ["safe"]

    print("M4B native memory store equivalence PASS")


if __name__ == "__main__":
    main()
