from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

from .corpus_io import atomic_write
from .p2_corpus_intake import (
    CANONICAL_SOURCES,
    VN97P2CorpusIntakeError,
    prepare_source,
    render_chat_jsonl,
)


class VN97P3CorpusIntakeError(RuntimeError):
    pass


P3_LIMITS = {
    "vi-dialogue-hoanghai2110": 2009,
    "databricks-dolly-15k": 15011,
    "openai-gsm8k-train": 7473,
}

_MAX_RAW_SOURCE_BYTES = 128 * 1024 * 1024


def _read_regular(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise VN97P3CorpusIntakeError(
            f"could not open P3 source safely: {path}"
        ) from exc
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or not 0 < info.st_size <= _MAX_RAW_SOURCE_BYTES
        ):
            raise VN97P3CorpusIntakeError(
                f"P3 source type/size is invalid: {path}"
            )
        data = bytearray()
        while len(data) < info.st_size:
            chunk = os.read(
                fd,
                min(1024 * 1024, info.st_size - len(data)),
            )
            if not chunk:
                break
            data.extend(chunk)
        after = os.fstat(fd)
        if (
            len(data) != info.st_size
            or after.st_size != info.st_size
            or after.st_ino != info.st_ino
            or after.st_dev != info.st_dev
        ):
            raise VN97P3CorpusIntakeError(
                f"P3 source changed while reading: {path}"
            )
        return bytes(data)
    finally:
        os.close(fd)


def _load_jsonl(data: bytes, *, label: str) -> list[object]:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise VN97P3CorpusIntakeError(
            f"{label} must be UTF-8 JSONL"
        ) from exc
    rows: list[object] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(
                line,
                parse_constant=lambda raw: (
                    _ for _ in ()
                ).throw(ValueError(raw)),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise VN97P3CorpusIntakeError(
                f"invalid JSON at {label}:{line_number}"
            ) from exc
        rows.append(value)
    if not rows:
        raise VN97P3CorpusIntakeError(
            f"{label} contains no records"
        )
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the expanded P3 language corpus from the same "
            "three reviewed/pinned public sources used by P2."
        )
    )
    parser.add_argument("--vi-dialogue", required=True)
    parser.add_argument("--dolly", required=True)
    parser.add_argument("--gsm8k", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--approve-source-license",
        action="append",
        default=[],
        metavar="SOURCE_ID",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    required = {item.source_id for item in CANONICAL_SOURCES}
    approvals = set(args.approve_source_license)
    if approvals != required:
        raise VN97P3CorpusIntakeError(
            "P3 source-license approvals must exactly match "
            f"canonical sources; missing={sorted(required-approvals)} "
            f"extra={sorted(approvals-required)}"
        )

    paths = {
        "vi-dialogue-hoanghai2110": Path(args.vi_dialogue),
        "databricks-dolly-15k": Path(args.dolly),
        "openai-gsm8k-train": Path(args.gsm8k),
    }
    identities: set[tuple[int, int]] = set()
    prepared = []

    for spec in CANONICAL_SOURCES:
        path = paths[spec.source_id]
        info = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):
            raise VN97P3CorpusIntakeError(
                f"P3 source must be a regular non-symlink file: {path}"
            )
        identity = (int(info.st_dev), int(info.st_ino))
        if identity in identities:
            raise VN97P3CorpusIntakeError(
                "P3 raw sources must be physically distinct"
            )
        identities.add(identity)

        data = _read_regular(path)
        prepared.append(
            prepare_source(
                spec.source_id,
                _load_jsonl(data, label=spec.source_id),
                raw_sha256=hashlib.sha256(data).hexdigest(),
                raw_bytes=len(data),
                max_records=P3_LIMITS[spec.source_id],
            )
        )

    split_counts = {
        split: sum(
            len(item.split_records[split])
            for item in prepared
        )
        for split in ("training", "validation", "release")
    }
    if (
        split_counts["training"] < 10_000
        or split_counts["validation"] < 500
        or split_counts["release"] < 500
    ):
        raise VN97P3CorpusIntakeError(
            "expanded P3 corpus is below minimum split sizes"
        )

    output = Path(args.output_dir)
    if output.exists():
        if output.is_symlink() or not output.is_dir() or any(output.iterdir()):
            raise VN97P3CorpusIntakeError(
                "P3 output-dir must be a real empty directory"
            )
    else:
        output.mkdir(parents=True, exist_ok=False)

    definition_sources = []
    summary_sources = []
    for item in prepared:
        summary_sources.append(
            {
                "accepted_records": item.accepted_records,
                "input_records": item.input_records,
                "license": item.spec.license,
                "max_records": P3_LIMITS[item.spec.source_id],
                "origin": item.spec.origin,
                "raw_bytes": item.raw_bytes,
                "raw_sha256": item.raw_sha256,
                "rejected_records": item.rejected_records,
                "source_id": item.spec.source_id,
                "splits": {
                    split: len(item.split_records[split])
                    for split in ("training", "validation", "release")
                },
            }
        )
        for split in ("training", "validation", "release"):
            rows = item.split_records[split]
            if not rows:
                continue
            filename = f"{item.spec.source_id}.{split}.jsonl"
            atomic_write(output / filename, render_chat_jsonl(rows))
            definition_sources.append(
                {
                    "license": item.spec.license,
                    "license_approved": True,
                    "mode": "chat",
                    "origin": item.spec.origin,
                    "path": filename,
                    "source_id": f"{item.spec.source_id}-{split}",
                    "split": split,
                }
            )

    definition = {
        "profile_id": "vn97-production-intelligence-v1",
        "schema": "VN97CORPUSDEF1",
        "sources": sorted(
            definition_sources,
            key=lambda item: (item["split"], item["source_id"]),
        ),
    }
    definition_bytes = (
        json.dumps(
            definition,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    atomic_write(
        output / "corpus-definition.vn97corpusdef1.json",
        definition_bytes,
    )

    summary_body = {
        "definition_sha256": hashlib.sha256(
            definition_bytes
        ).hexdigest(),
        "schema": "VN97P3SRC1",
        "sources": sorted(
            summary_sources,
            key=lambda item: item["source_id"],
        ),
        "split_records": split_counts,
    }
    summary = dict(summary_body)
    summary["summary_id"] = hashlib.sha256(
        b"VN97P3SRC1\0"
        + json.dumps(
            summary_body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    atomic_write(
        output / "p3-source-summary.vn97p3src1.json",
        json.dumps(
            summary,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n",
    )

    print(
        "VN97P3SRC1 "
        f"id={summary['summary_id']} "
        f"training={split_counts['training']} "
        f"validation={split_counts['validation']} "
        f"release={split_counts['release']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-p3-corpus-intake: {exc}", file=sys.stderr)
        raise
