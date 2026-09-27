from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .production_source_adapter import (
    build_r2d10_pack,
    verify_r2d10_pack,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build or verify a VN97-R2D10 pinned production source "
            "adapter pack and its direct R2-D9 handoff."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--lock", required=True)
    build.add_argument("--output-dir", required=True)

    verify = sub.add_parser("verify")
    verify.add_argument("--pack-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "build":
        payload = build_r2d10_pack(
            lock_path=Path(args.lock),
            output_dir=Path(args.output_dir),
        )
        print(
            "VN97R2D10 "
            f"command=build pack={payload['pack_id']} "
            f"sources={len(payload['receipts'])}",
            flush=True,
        )
        return 0

    payload = verify_r2d10_pack(
        Path(args.pack_dir)
    )
    print(
        "VN97R2D10 "
        f"command=verify pack={payload['pack_id']} "
        f"sources={len(payload['receipts'])} "
        "status=PASS",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-r2-source-pack: {exc}", file=sys.stderr)
        raise
