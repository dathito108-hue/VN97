from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vn97.production_corpus import (  # noqa: E402
    VN97ProductionCorpusError,
    prepare_corpus,
)


@dataclass(frozen=True)
class Msg:
    role: str
    content: str


def chat(user: str, assistant: str):
    return (
        Msg("user", user),
        Msg("assistant", assistant),
    )


def expect_failure(label: str, fn) -> None:
    try:
        fn()
    except Exception:
        return
    raise AssertionError(f"expected failure: {label}")


def source(
    source_id: str,
    split: str,
    records,
    *,
    raw: bytes | None = None,
):
    raw_bytes = raw or (
        source_id.encode("utf-8") + b"\n"
    )
    return (
        source_id,
        f"https://example.invalid/{source_id}",
        "CC-BY-4.0",
        split,
        "chat",
        raw_bytes,
        records,
    )


def main() -> None:
    train_a = [
        chat("hello", "world"),
        chat("2+2?", "4"),
    ]
    train_b = [
        chat("2+2?", "4"),
        chat("plan", "step one"),
    ]
    validation = [
        chat("validation", "answer"),
    ]
    release = [
        chat("sealed", "holdout"),
    ]

    rows = [
        source("train-b", "training", train_b),
        source("release-a", "release", release),
        source("train-a", "training", train_a),
        source("validation-a", "validation", validation),
    ]
    manifest, prepared = prepare_corpus(
        rows,
        profile_id="vn97-production-intelligence-v1",
    )

    assert manifest.training.records == 3
    assert manifest.validation.records == 1
    assert manifest.release.records == 1
    assert prepared["training"].bytes.endswith(b"\n")
    assert (
        hashlib.sha256(
            prepared["training"].bytes
        ).hexdigest()
        == manifest.training.sha256
    )
    sources = {
        item.source_id: item
        for item in manifest.sources
    }
    assert sources["train-a"].kept_records == 2
    assert sources["train-b"].kept_records == 1
    assert sources["train-b"].duplicate_records == 1

    manifest2, prepared2 = prepare_corpus(
        list(reversed(rows)),
        profile_id="vn97-production-intelligence-v1",
    )
    assert manifest2.to_bytes() == manifest.to_bytes()
    for split in ("training", "validation", "release"):
        assert prepared2[split].bytes == prepared[split].bytes

    expect_failure(
        "cross-split overlap",
        lambda: prepare_corpus(
            [
                source(
                    "train",
                    "training",
                    [chat("same", "record")],
                ),
                source(
                    "validation",
                    "validation",
                    [chat("different", "record")],
                ),
                source(
                    "release",
                    "release",
                    [chat("same", "record")],
                ),
            ],
            profile_id="vn97-production-intelligence-v1",
        ),
    )

    expect_failure(
        "missing release",
        lambda: prepare_corpus(
            [
                source(
                    "train",
                    "training",
                    [chat("a", "b")],
                ),
                source(
                    "validation",
                    "validation",
                    [chat("c", "d")],
                ),
            ],
            profile_id="vn97-production-intelligence-v1",
        ),
    )

    expect_failure(
        "mixed mode within split",
        lambda: prepare_corpus(
            [
                source(
                    "train-chat",
                    "training",
                    [chat("a", "b")],
                ),
                (
                    "train-text",
                    "https://example.invalid/text",
                    "CC0-1.0",
                    "training",
                    "text",
                    b"text\n",
                    ["plain text"],
                ),
                source(
                    "validation",
                    "validation",
                    [chat("c", "d")],
                ),
                source(
                    "release",
                    "release",
                    [chat("e", "f")],
                ),
            ],
            profile_id="vn97-production-intelligence-v1",
        ),
    )

    payload = manifest.to_bytes()
    assert b'"schema":"VN97CORPUS1"' in payload
    assert manifest.manifest_id.encode("ascii") in payload

    print(
        "VN97 production corpus v1 host contract PASS "
        f"id={manifest.manifest_id}"
    )


if __name__ == "__main__":
    main()
