from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vn97.r2.production_acquisition import load_r2d9_definition
from vn97.r2.production_source_adapter import (
    R2D10_LOCK_SCHEMA,
    R2D10SourceLock,
    adapt_r2d10_record,
    build_r2d10_pack,
    load_r2d10_lock,
    verify_r2d10_pack,
)


def _jsonl(path: Path, rows: list[dict[str, object]]) -> tuple[str, int]:
    data = b"".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest(), len(rows)


def _source(
    *,
    source_id: str,
    family: str,
    adapter: str,
    path: str,
    sha256: str,
    records: int,
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "origin": f"https://example.test/{source_id}",
        "revision": "release-2026-09-27",
        "license": "MIT",
        "license_approved": True,
        "family": family,
        "adapter": adapter,
        "path": path,
        "expected_sha256": sha256,
        "expected_records": records,
        "max_bytes": 1024 * 1024,
    }


def _lock(
    path: Path,
    sources: list[dict[str, object]],
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": R2D10_LOCK_SCHEMA,
                "profile_id": "vn97-production-intelligence-v1",
                "sources": sources,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("adapter", "raw", "first_role", "last_role"),
    [
        (
            "messages",
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ]
            },
            "user",
            "assistant",
        ),
        (
            "dolly",
            {
                "instruction": "Explain",
                "context": "Context",
                "response": "Answer",
                "category": "closed_qa",
            },
            "user",
            "assistant",
        ),
        (
            "gsm8k",
            {"question": "2+2?", "answer": "4"},
            "user",
            "assistant",
        ),
        (
            "instruction_io",
            {
                "instruction": "Translate",
                "input": "hello",
                "output": "xin chào",
            },
            "user",
            "assistant",
        ),
        (
            "question_answer",
            {"question": "Capital?", "answer": "Hanoi"},
            "user",
            "assistant",
        ),
        (
            "prompt_response",
            {"prompt": "Say hi", "response": "Hi"},
            "user",
            "assistant",
        ),
        (
            "tool_trace",
            {
                "prompt": "Find the weather",
                "tool_name": "weather.lookup",
                "arguments": {"city": "Hanoi"},
                "result": {"temp_c": 29},
                "response": "It is 29 C.",
            },
            "user",
            "assistant",
        ),
        (
            "action_trace",
            {
                "observation": "App is closed",
                "action": {"type": "launch", "package": "example.app"},
                "result": {"status": "opened"},
                "response": "Opened.",
            },
            "user",
            "assistant",
        ),
        (
            "capability_demo",
            {
                "capability": "summarization",
                "instruction": "Summarize this.",
                "response": "Summary.",
            },
            "system",
            "assistant",
        ),
    ],
)
def test_adapter_pack_supported_shapes(
    adapter: str,
    raw: dict[str, object],
    first_role: str,
    last_role: str,
) -> None:
    messages = adapt_r2d10_record(adapter, raw)
    assert messages[0]["role"] == first_role
    assert messages[-1]["role"] == last_role
    assert any(item["role"] == "user" for item in messages)


def test_tool_trace_keeps_result_as_context() -> None:
    messages = adapt_r2d10_record(
        "tool_trace",
        {
            "prompt": "Look up account state",
            "tool_name": "state.read",
            "arguments": {"id": "abc"},
            "result": {"status": "ok"},
            "response": "The account is active.",
        },
    )
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"].startswith("[TOOL_CALL]\n")
    assert messages[2]["role"] == "system"
    assert messages[2]["content"].startswith("[TOOL_RESULT]\n")
    assert messages[3]["role"] == "assistant"


def test_action_trace_keeps_result_as_context() -> None:
    messages = adapt_r2d10_record(
        "action_trace",
        {
            "observation": "Need to open an app",
            "action": {"type": "launch", "package": "example"},
            "result": {"status": "ok"},
            "response": "Done.",
        },
    )
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"].startswith("[ACTION]\n")
    assert messages[2]["role"] == "system"
    assert messages[2]["content"].startswith("[ACTION_RESULT]\n")


def test_source_lock_rejects_adapter_family_mismatch() -> None:
    with pytest.raises(ValueError, match="not compatible"):
        R2D10SourceLock(
            source_id="bad",
            origin="https://example.test/bad",
            revision="v1",
            license="MIT",
            family="language",
            adapter="tool_trace",
            path="bad.jsonl",
            expected_sha256="a" * 64,
            expected_records=1,
            max_bytes=1024,
        )


def test_r2d10_builds_normalized_pack_and_d9_handoff(
    tmp_path: Path,
) -> None:
    instruction_sha, instruction_records = _jsonl(
        tmp_path / "instruction.jsonl",
        [
            {
                "instruction": "Translate",
                "input": "hello",
                "output": "xin chào",
            },
            {
                "instruction": "Answer",
                "input": "2+2",
                "output": "4",
            },
        ],
    )
    reasoning_sha, reasoning_records = _jsonl(
        tmp_path / "reasoning.jsonl",
        [
            {"question": "1+1?", "answer": "2"},
            {"question": "3+4?", "answer": "7"},
        ],
    )
    tool_sha, tool_records = _jsonl(
        tmp_path / "tool.jsonl",
        [
            {
                "prompt": "Search",
                "tool_name": "search",
                "arguments": {"q": "VN97"},
                "result": {"hits": 1},
                "response": "Found one result.",
            }
        ],
    )

    lock = _lock(
        tmp_path / "lock.json",
        [
            _source(
                source_id="instruction-source",
                family="instruction",
                adapter="instruction_io",
                path="instruction.jsonl",
                sha256=instruction_sha,
                records=instruction_records,
            ),
            _source(
                source_id="reasoning-source",
                family="reasoning",
                adapter="question_answer",
                path="reasoning.jsonl",
                sha256=reasoning_sha,
                records=reasoning_records,
            ),
            _source(
                source_id="tool-source",
                family="tool",
                adapter="tool_trace",
                path="tool.jsonl",
                sha256=tool_sha,
                records=tool_records,
            ),
        ],
    )

    pack_dir = tmp_path / "pack"
    pack = build_r2d10_pack(
        lock_path=lock,
        output_dir=pack_dir,
    )
    assert len(pack["receipts"]) == 3

    verified = verify_r2d10_pack(pack_dir)
    assert verified["pack_id"] == pack["pack_id"]

    d9 = load_r2d9_definition(
        pack_dir / "r2d9-definition.json"
    )
    assert {item.family for item in d9.sources} == {
        "instruction",
        "reasoning",
        "tool",
    }
    assert {
        item.source_id for item in d9.sources
    } == {
        "instruction-source",
        "reasoning-source",
        "tool-source",
    }

    for receipt in pack["receipts"]:
        normalized = pack_dir / receipt["normalized_path"]
        rows = [
            json.loads(line)
            for line in normalized.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        assert len(rows) == receipt["normalized_records"]
        assert all(set(row) == {"messages"} for row in rows)


def test_r2d10_pack_rejects_normalized_tamper(
    tmp_path: Path,
) -> None:
    sha, records = _jsonl(
        tmp_path / "source.jsonl",
        [{"question": "Q?", "answer": "A."}],
    )
    lock = _lock(
        tmp_path / "lock.json",
        [
            _source(
                source_id="reasoning-source",
                family="reasoning",
                adapter="question_answer",
                path="source.jsonl",
                sha256=sha,
                records=records,
            )
        ],
    )
    pack_dir = tmp_path / "pack"
    pack = build_r2d10_pack(
        lock_path=lock,
        output_dir=pack_dir,
    )
    normalized = pack_dir / pack["receipts"][0]["normalized_path"]
    with normalized.open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(ValueError, match="normalized"):
        verify_r2d10_pack(pack_dir)


def test_r2d10_pack_rejects_d9_handoff_tamper(
    tmp_path: Path,
) -> None:
    sha, records = _jsonl(
        tmp_path / "source.jsonl",
        [{"question": "Q?", "answer": "A."}],
    )
    lock = _lock(
        tmp_path / "lock.json",
        [
            _source(
                source_id="reasoning-source",
                family="reasoning",
                adapter="question_answer",
                path="source.jsonl",
                sha256=sha,
                records=records,
            )
        ],
    )
    pack_dir = tmp_path / "pack"
    build_r2d10_pack(
        lock_path=lock,
        output_dir=pack_dir,
    )
    d9_path = pack_dir / "r2d9-definition.json"
    d9 = json.loads(d9_path.read_text(encoding="utf-8"))
    d9["sources"][0]["family"] = "language"
    d9_path.write_text(json.dumps(d9), encoding="utf-8")

    with pytest.raises(ValueError, match="D9 definition"):
        verify_r2d10_pack(pack_dir)


def test_r2d10_source_sha_mismatch_fails_before_output(
    tmp_path: Path,
) -> None:
    _, records = _jsonl(
        tmp_path / "source.jsonl",
        [{"question": "Q?", "answer": "A."}],
    )
    lock = _lock(
        tmp_path / "lock.json",
        [
            _source(
                source_id="reasoning-source",
                family="reasoning",
                adapter="question_answer",
                path="source.jsonl",
                sha256="0" * 64,
                records=records,
            )
        ],
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_r2d10_pack(
            lock_path=lock,
            output_dir=tmp_path / "pack",
        )


def test_r2d10_source_symlink_is_rejected(
    tmp_path: Path,
) -> None:
    sha, records = _jsonl(
        tmp_path / "real.jsonl",
        [{"question": "Q?", "answer": "A."}],
    )
    link = tmp_path / "link.jsonl"
    try:
        link.symlink_to(tmp_path / "real.jsonl")
    except OSError:
        pytest.skip("symlink unavailable on this platform")

    lock = _lock(
        tmp_path / "lock.json",
        [
            _source(
                source_id="reasoning-source",
                family="reasoning",
                adapter="question_answer",
                path="link.jsonl",
                sha256=sha,
                records=records,
            )
        ],
    )
    with pytest.raises(ValueError, match="symlink"):
        build_r2d10_pack(
            lock_path=lock,
            output_dir=tmp_path / "pack",
        )


def test_lock_source_order_does_not_change_fingerprint(
    tmp_path: Path,
) -> None:
    sha_a, rec_a = _jsonl(
        tmp_path / "a.jsonl",
        [{"question": "A?", "answer": "A."}],
    )
    sha_b, rec_b = _jsonl(
        tmp_path / "b.jsonl",
        [{"question": "B?", "answer": "B."}],
    )
    a = _source(
        source_id="a-source",
        family="reasoning",
        adapter="question_answer",
        path="a.jsonl",
        sha256=sha_a,
        records=rec_a,
    )
    b = _source(
        source_id="b-source",
        family="reasoning",
        adapter="question_answer",
        path="b.jsonl",
        sha256=sha_b,
        records=rec_b,
    )
    one = _lock(tmp_path / "one.json", [a, b])
    two = _lock(tmp_path / "two.json", [b, a])

    assert (
        load_r2d10_lock(one).fingerprint()
        == load_r2d10_lock(two).fingerprint()
    )
