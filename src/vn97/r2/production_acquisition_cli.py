from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .production_acquisition import (
    build_r2d9_campaign,
    verify_r2d9_campaign,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build or verify a VN97-R2D9 production corpus acquisition "
            "and sealing campaign from pinned, license-approved chat JSONL."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--definition", required=True)
    build.add_argument("--output-dir", required=True)

    verify = sub.add_parser("verify")
    verify.add_argument("--campaign-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "build":
        payload = build_r2d9_campaign(
            definition_path=Path(args.definition),
            output_dir=Path(args.output_dir),
        )
        print(
            "VN97R2D9 "
            f"command=build campaign={payload['campaign_id']} "
            f"unique_records={payload['global_unique_records']} "
            f"seals={len(payload['seals'])} "
            f"families={len(payload['family_totals'])}",
            flush=True,
        )
        return 0

    payload = verify_r2d9_campaign(
        Path(args.campaign_dir)
    )
    print(
        "VN97R2D9 "
        f"command=verify campaign={payload['campaign_id']} "
        f"unique_records={payload['global_unique_records']} "
        f"seals={len(payload['seals'])} "
        "status=PASS",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-r2-corpus-campaign: {exc}", file=sys.stderr)
        raise
