from __future__ import annotations

import ctypes
from pathlib import Path
import sys
import tempfile

from vn97.memory import (
    MemoryJournal,
    MemoryKind,
)


U8P = ctypes.POINTER(
    ctypes.c_uint8
)


def scan(
    lib,
    blob: bytes,
    allow_torn_tail: bool,
):
    array = (
        ctypes.c_uint8
        * len(blob)
    ).from_buffer_copy(blob)
    vector_dim = ctypes.c_uint32()
    record_count = ctypes.c_size_t()
    last_record_id = ctypes.c_uint64()
    valid_bytes = ctypes.c_size_t()
    tail_truncated = ctypes.c_int()

    status = lib.vn97_memory_scan(
        array,
        len(blob),
        int(allow_torn_tail),
        ctypes.byref(vector_dim),
        ctypes.byref(record_count),
        ctypes.byref(last_record_id),
        ctypes.byref(valid_bytes),
        ctypes.byref(tail_truncated),
    )
    return (
        status,
        vector_dim.value,
        record_count.value,
        last_record_id.value,
        valid_bytes.value,
        bool(tail_truncated.value),
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: memory_python_equivalence.py "
            "<shared-library>"
        )

    lib = ctypes.CDLL(
        str(
            Path(
                sys.argv[1]
            ).resolve()
        )
    )
    lib.vn97_memory_scan.argtypes = [
        U8P,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.POINTER(
            ctypes.c_uint32),
        ctypes.POINTER(
            ctypes.c_size_t),
        ctypes.POINTER(
            ctypes.c_uint64),
        ctypes.POINTER(
            ctypes.c_size_t),
        ctypes.POINTER(
            ctypes.c_int),
    ]
    lib.vn97_memory_scan.restype = (
        ctypes.c_int
    )

    with tempfile.TemporaryDirectory() as directory:
        path = (
            Path(directory)
            / "memory.vn97mem"
        )
        journal = MemoryJournal.create(
            path,
            vector_dim=4,
        )
        root = journal.append(
            MemoryKind.EPISODIC,
            "VN97 memory event",
            source="test",
            importance=0.6,
            vector=[
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            timestamp_ns=100,
        )
        journal.append(
            MemoryKind.SEMANTIC,
            "VN97 persistent semantic memory",
            source="derived",
            importance=0.9,
            vector=[
                0.9,
                0.1,
                0.0,
                0.0,
            ],
            parent_id=root.record_id,
            timestamp_ns=200,
        )
        blob = path.read_bytes()

    result = scan(
        lib,
        blob,
        False,
    )
    assert result == (
        0,
        4,
        2,
        2,
        len(blob),
        False,
    )

    torn = (
        blob
        + b"\x20\x00\x00"
    )
    strict = scan(
        lib,
        torn,
        False,
    )
    assert strict[0] == 9
    recovered = scan(
        lib,
        torn,
        True,
    )
    assert recovered == (
        0,
        4,
        2,
        2,
        len(blob),
        True,
    )

    corrupt = bytearray(blob)
    corrupt[-1] ^= 0x01
    failed = scan(
        lib,
        bytes(corrupt),
        True,
    )
    assert failed[0] == 7

    print(
        "M4A native memory scan equivalence PASS"
    )


if __name__ == "__main__":
    main()
