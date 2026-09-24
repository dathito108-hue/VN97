from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "vn97"

pkg = types.ModuleType("vn97")
pkg.__path__ = [str(SRC)]
sys.modules["vn97"] = pkg

spec = importlib.util.spec_from_file_location(
    "vn97.p2_corpus_intake",
    SRC / "p2_corpus_intake.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load p2_corpus_intake.py")
module = importlib.util.module_from_spec(spec)
sys.modules["vn97.p2_corpus_intake"] = module
spec.loader.exec_module(module)


def expect_failure(label: str, fn) -> None:
    try:
        fn()
    except Exception:
        return
    raise AssertionError(f"expected failure: {label}")


def main() -> None:
    ids = tuple(
        item.source_id
        for item in module.CANONICAL_SOURCES
    )
    assert ids == (
        "vi-dialogue-hoanghai2110",
        "databricks-dolly-15k",
        "openai-gsm8k-train",
    )

    vi = module.adapt_record(
        "vi-dialogue-hoanghai2110",
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Giải thích pin điện thoại.",
                },
                {
                    "role": "assistant",
                    "content": "Pin lưu trữ năng lượng hóa học.",
                },
            ]
        },
    )
    assert vi[-1]["role"] == "assistant"

    dolly = module.adapt_record(
        "databricks-dolly-15k",
        {
            "instruction": "Summarize the note.",
            "context": "A short context.",
            "response": "A short summary.",
            "category": "summarization",
        },
    )
    assert "Context:" in dolly[0]["content"]
    assert dolly[1]["content"] == "A short summary."

    gsm = module.adapt_record(
        "openai-gsm8k-train",
        {
            "question": "If 2 apples are added to 3, how many?",
            "answer": "5",
        },
    )
    assert gsm[0]["role"] == "user"
    assert gsm[1]["content"] == "5"

    assert module.has_identity_contamination(
        (
            {
                "role": "user",
                "content": "Bạn là ai?",
            },
            {
                "role": "assistant",
                "content": "Tôi là HyperMamba.",
            },
        )
    )
    assert not module.has_identity_contamination(vi)

    # Split identity deliberately ignores source ID so mirrored conversations
    # cannot land on opposite sides of the train/holdout boundary.
    assert (
        module.assigned_split(
            "vi-dialogue-hoanghai2110",
            vi,
        )
        == module.assigned_split(
            "databricks-dolly-15k",
            vi,
        )
    )

    vi_records = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"Câu hỏi {index}",
                },
                {
                    "role": "assistant",
                    "content": f"Câu trả lời {index}",
                },
            ]
        }
        for index in range(120)
    ]
    vi_records.append(vi_records[0])
    vi_records.append(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Bạn là ai?",
                },
                {
                    "role": "assistant",
                    "content": "Tôi là Vimind HyperMamba.",
                },
            ]
        }
    )
    prepared = module.prepare_source(
        "vi-dialogue-hoanghai2110",
        vi_records,
        raw_sha256="a" * 64,
        raw_bytes=1000,
    )
    assert prepared.input_records == 122
    assert prepared.accepted_records == 120
    assert prepared.rejected_records == 2
    assert sum(
        len(prepared.split_records[split])
        for split in (
            "training",
            "validation",
            "release",
        )
    ) == 120

    # Determinism is independent of source input order.
    prepared2 = module.prepare_source(
        "vi-dialogue-hoanghai2110",
        list(reversed(vi_records)),
        raw_sha256="a" * 64,
        raw_bytes=1000,
    )
    assert prepared.split_records == prepared2.split_records

    rendered = module.render_chat_jsonl(
        prepared.split_records["training"]
    )
    assert rendered
    assert rendered.endswith(b"\n")
    assert b'"messages"' in rendered

    summary = module.source_summary([prepared])
    assert summary["schema"] == "VN97P2SRC1"
    assert len(summary["summary_id"]) == 64
    row = summary["sources"][0]
    assert row["raw_sha256"] == "a" * 64
    assert row["raw_bytes"] == 1000
    assert row["accepted_records"] == 120

    expect_failure(
        "invalid Dolly shape",
        lambda: module.adapt_record(
            "databricks-dolly-15k",
            {
                "instruction": "x",
                "response": "y",
            },
        ),
    )
    expect_failure(
        "conversation without assistant target",
        lambda: module.adapt_record(
            "vi-dialogue-hoanghai2110",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "only user",
                    }
                ]
            },
        ),
    )
    expect_failure(
        "invalid raw hash",
        lambda: module.prepare_source(
            "openai-gsm8k-train",
            [
                {
                    "question": "1+1?",
                    "answer": "2",
                }
            ],
            raw_sha256="bad",
            raw_bytes=10,
        ),
    )

    print(
        "VN97 P2 corpus intake host contract PASS "
        f"sources={len(module.CANONICAL_SOURCES)}"
    )


if __name__ == "__main__":
    main()
