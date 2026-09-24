from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
import sys

from .p2_corpus_bundle import (
    VN97P2CorpusBundleError,
    verify_p2_corpus_chain,
)


def _create_only(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise VN97P2CorpusBundleError(
            "P2 bundle receipt output must not already exist"
        )
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink():
        raise VN97P2CorpusBundleError(
            "P2 bundle receipt parent must not be a symlink"
        )
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=parent,
    )
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "wb", closefd=True) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp, path)
        except FileExistsError as exc:
            raise VN97P2CorpusBundleError(
                "P2 bundle receipt raced with another writer"
            ) from exc
        dir_fd = os.open(
            parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        temp.unlink(missing_ok=True)
    if path.read_bytes() != data:
        raise VN97P2CorpusBundleError(
            "P2 bundle receipt post-write verification failed"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the complete P2 fetch -> intake -> VN97CORPUS1 "
            "chain and emit VN97P2BUNDLE1."
        )
    )
    parser.add_argument(
        "--fetch-receipt",
        required=True,
    )
    parser.add_argument(
        "--source-summary",
        required=True,
    )
    parser.add_argument(
        "--definition",
        required=True,
    )
    parser.add_argument(
        "--corpus-dir",
        required=True,
    )
    parser.add_argument(
        "--repository-commit",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = verify_p2_corpus_chain(
        fetch_receipt_path=Path(args.fetch_receipt),
        source_summary_path=Path(args.source_summary),
        definition_path=Path(args.definition),
        corpus_dir=Path(args.corpus_dir),
        repository_commit=args.repository_commit,
    )
    output = Path(args.output)
    _create_only(
        output,
        receipt.to_bytes(),
    )
    payload = receipt.canonical_object()
    print(
        "VN97P2BUNDLE1 "
        f"id={payload['bundle_id']} "
        f"corpus={receipt.corpus_manifest_id}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"vn97-p2-corpus-bundle: {exc}",
            file=sys.stderr,
        )
        raise
