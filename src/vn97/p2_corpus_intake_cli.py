from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

from .p2_corpus_intake import (
    CANONICAL_SOURCES,
    VN97P2CorpusIntakeError,
    prepare_source,
    render_chat_jsonl,
    source_summary,
)
from .corpus_io import atomic_write as _atomic_write


_MAX_RAW_SOURCE_BYTES = 128 * 1024 * 1024


def _read_regular_file(
    path: Path,
    *,
    max_bytes: int = _MAX_RAW_SOURCE_BYTES,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise VN97P2CorpusIntakeError(
            f"could not open P2 source safely: {path}"
        ) from exc
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or not 0 < info.st_size <= max_bytes
        ):
            raise VN97P2CorpusIntakeError(
                f"P2 source size/type is invalid: {path}"
            )
        out = bytearray()
        while len(out) < info.st_size:
            chunk = os.read(
                fd,
                min(
                    1024 * 1024,
                    info.st_size - len(out),
                ),
            )
            if not chunk:
                break
            out.extend(chunk)
        after = os.fstat(fd)
        if (
            len(out) != info.st_size
            or after.st_size != info.st_size
            or after.st_ino != info.st_ino
            or after.st_dev != info.st_dev
        ):
            raise VN97P2CorpusIntakeError(
                f"P2 source changed while being read: {path}"
            )
        return bytes(out)
    finally:
        os.close(fd)


def _load_jsonl(data: bytes, *, label: str) -> list[object]:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise VN97P2CorpusIntakeError(
            f"{label} must be UTF-8 JSONL"
        ) from exc

    records: list[object] = []
    for line_number, line in enumerate(
        text.split("\n"),
        start=1,
    ):
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
            raise VN97P2CorpusIntakeError(
                f"invalid JSON at {label}:{line_number}"
            ) from exc
        records.append(value)
    if not records:
        raise VN97P2CorpusIntakeError(
            f"{label} contains no records"
        )
    return records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert three reviewed public P2 datasets into deterministic "
            "VN97 chat JSONL sources plus VN97CORPUSDEF1."
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
        help=(
            "Explicitly confirm review/acceptance of one source license. "
            "Repeat for every canonical P2 source."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    required_approvals = {
        item.source_id for item in CANONICAL_SOURCES
    }
    approvals = set(args.approve_source_license)
    if approvals != required_approvals:
        missing = sorted(
            required_approvals - approvals
        )
        extra = sorted(
            approvals - required_approvals
        )
        raise VN97P2CorpusIntakeError(
            "P2 source-license approvals must exactly match canonical "
            f"sources; missing={missing} extra={extra}"
        )

    raw_paths = {
        "vi-dialogue-hoanghai2110":
            Path(args.vi_dialogue),
        "databricks-dolly-15k":
            Path(args.dolly),
        "openai-gsm8k-train":
            Path(args.gsm8k),
    }
    identities: set[tuple[int, int]] = set()
    prepared = []
    raw_digests: dict[str, str] = {}

    for spec in CANONICAL_SOURCES:
        path = raw_paths[spec.source_id]
        try:
            info = os.stat(
                path,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise VN97P2CorpusIntakeError(
                f"P2 source is unavailable: {path}"
            ) from exc
        if not stat.S_ISREG(info.st_mode):
            raise VN97P2CorpusIntakeError(
                f"P2 source must be a regular non-symlink file: {path}"
            )
        identity = (
            int(info.st_dev),
            int(info.st_ino),
        )
        if identity in identities:
            raise VN97P2CorpusIntakeError(
                "P2 raw sources must be physically distinct files"
            )
        identities.add(identity)

        data = _read_regular_file(path)
        digest = hashlib.sha256(data).hexdigest()
        raw_digests[spec.source_id] = digest
        records = _load_jsonl(
            data,
            label=spec.source_id,
        )
        value = prepare_source(
            spec.source_id,
            records,
            raw_sha256=digest,
            raw_bytes=len(data),
        )
        if value.accepted_records <= 0:
            raise VN97P2CorpusIntakeError(
                f"P2 source produced no accepted records: {spec.source_id}"
            )
        prepared.append(value)

    total_split_counts = {
        split: sum(
            len(item.split_records[split])
            for item in prepared
        )
        for split in (
            "training",
            "validation",
            "release",
        )
    }
    if any(
        total_split_counts[split] < 16
        for split in (
            "training",
            "validation",
            "release",
        )
    ):
        raise VN97P2CorpusIntakeError(
            "P2 corpus must contain at least 16 records in every split"
        )

    output = Path(args.output_dir)
    if output.exists():
        if output.is_symlink() or not output.is_dir():
            raise VN97P2CorpusIntakeError(
                "P2 intake output-dir must be a real directory"
            )
        if any(output.iterdir()):
            raise VN97P2CorpusIntakeError(
                "P2 intake output-dir must be new or empty"
            )
    else:
        output.mkdir(
            parents=True,
            exist_ok=False,
        )
    if output.is_symlink():
        raise VN97P2CorpusIntakeError(
            "P2 intake output-dir must not be a symlink"
        )

    definition_sources = []
    for item in prepared:
        for split in (
            "training",
            "validation",
            "release",
        ):
            rows = item.split_records[split]
            if not rows:
                continue
            filename = (
                f"{item.spec.source_id}.{split}.jsonl"
            )
            _atomic_write(
                output / filename,
                render_chat_jsonl(rows),
            )
            definition_sources.append(
                {
                    "license": item.spec.license,
                    "license_approved": True,
                    "mode": "chat",
                    "origin": item.spec.origin,
                    "path": filename,
                    "source_id": (
                        f"{item.spec.source_id}-{split}"
                    ),
                    "split": split,
                }
            )

    definition = {
        "profile_id":
            "vn97-production-intelligence-v1",
        "schema": "VN97CORPUSDEF1",
        "sources": sorted(
            definition_sources,
            key=lambda item: (
                item["split"],
                item["source_id"],
            ),
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
    _atomic_write(
        output / "corpus-definition.vn97corpusdef1.json",
        definition_bytes,
    )

    summary = source_summary(prepared)
    summary["definition_sha256"] = hashlib.sha256(
        definition_bytes
    ).hexdigest()
    summary["split_records"] = total_split_counts
    summary["raw_source_sha256"] = {
        key: raw_digests[key]
        for key in sorted(raw_digests)
    }
    summary_without_id = dict(summary)
    summary_without_id.pop(
        "summary_id",
        None,
    )
    summary["summary_id"] = hashlib.sha256(
        b"VN97P2SRC1\0"
        + json.dumps(
            summary_without_id,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    summary_bytes = (
        json.dumps(
            summary,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    _atomic_write(
        output / "p2-source-summary.vn97p2src1.json",
        summary_bytes,
    )

    print(
        "VN97P2SRC1 "
        f"id={summary['summary_id']} "
        f"training={total_split_counts['training']} "
        f"validation={total_split_counts['validation']} "
        f"release={total_split_counts['release']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"vn97-p2-corpus-intake: {exc}",
            file=sys.stderr,
        )
        raise
