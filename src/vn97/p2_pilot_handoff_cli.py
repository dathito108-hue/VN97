from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
import sys

from .p2_pilot_handoff import (
    VN97P2PilotHandoffError,
    build_pilot_run_receipt,
    verify_corpus_artifact,
)


def _create_only(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise VN97P2PilotHandoffError(
            "P2 run receipt output must not already exist"
        )
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink():
        raise VN97P2PilotHandoffError(
            "P2 run receipt parent must not be a symlink"
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
            raise VN97P2PilotHandoffError(
                "P2 run receipt raced with another writer"
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
        raise VN97P2PilotHandoffError(
            "P2 run receipt post-write verification failed"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the P2 sealed corpus artifact handoff and seal "
            "the resulting medium-pilot run evidence."
        )
    )
    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    verify = sub.add_parser(
        "verify-corpus",
        help="Verify the downloaded VN97-P2-Corpus artifact.",
    )
    verify.add_argument("--artifact-root", required=True)
    verify.add_argument("--corpus-commit", required=True)

    seal = sub.add_parser(
        "seal-run",
        help="Bind a successful VN97PILOT1 output to the exact corpus artifact.",
    )
    seal.add_argument("--artifact-root", required=True)
    seal.add_argument("--corpus-commit", required=True)
    seal.add_argument("--pilot-output", required=True)
    seal.add_argument("--training-commit", required=True)
    seal.add_argument("--corpus-run-id", required=True)
    seal.add_argument("--training-run-id", required=True)
    seal.add_argument("--python-version", required=True)
    seal.add_argument("--torch-version", required=True)
    seal.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    corpus = verify_corpus_artifact(
        Path(args.artifact_root),
        expected_corpus_commit=args.corpus_commit,
    )
    if args.command == "verify-corpus":
        print(
            "VN97P2CORPUSHANDOFF "
            f"bundle={corpus.bundle_id} "
            f"corpus={corpus.corpus_manifest_id}"
        )
        return 0

    receipt = build_pilot_run_receipt(
        corpus=corpus,
        pilot_output=Path(args.pilot_output),
        training_commit=args.training_commit,
        corpus_run_id=args.corpus_run_id,
        training_run_id=args.training_run_id,
        python_version=args.python_version,
        torch_version=args.torch_version,
    )
    _create_only(
        Path(args.output),
        receipt.to_bytes(),
    )
    payload = receipt.canonical_object()
    print(
        "VN97P2RUN1 "
        f"id={payload['run_identity']} "
        f"checkpoint={receipt.checkpoint_sha256}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"vn97-p2-pilot-handoff: {exc}",
            file=sys.stderr,
        )
        raise
