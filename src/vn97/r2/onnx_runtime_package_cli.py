from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .onnx_runtime_package import (
    R2F1_RUNTIME_FILENAME,
    build_r2f1_runtime_package,
    verify_r2f1_runtime_package,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build or verify the portable VN97-R2F1 Android ONNX "
            "production runtime package descriptor."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--bundle-dir", required=True)
    build.add_argument("--e4-profile", required=True)
    build.add_argument("--output")

    verify = sub.add_parser("verify")
    verify.add_argument("--package", required=True)
    verify.add_argument("--bundle-id")
    verify.add_argument("--tuning-id")

    status = sub.add_parser("status")
    status.add_argument("--package", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        output = (
            Path(args.output)
            if args.output is not None
            else Path(args.bundle_dir) / R2F1_RUNTIME_FILENAME
        )
        payload = build_r2f1_runtime_package(
            bundle_dir=Path(args.bundle_dir),
            e4_profile_path=Path(args.e4_profile),
            output_path=output,
        )
        print(
            "VN97R2F1 "
            f"command=build runtime={payload['runtime_id']} "
            f"bundle={payload['bundle_id']} "
            f"tuning={payload['tuning_id']}",
            flush=True,
        )
        return 0

    payload = verify_r2f1_runtime_package(
        Path(args.package),
        expected_bundle_id=getattr(args, "bundle_id", None),
        expected_tuning_id=getattr(args, "tuning_id", None),
    )
    if args.command == "verify":
        print(
            "VN97R2F1 "
            f"command=verify runtime={payload['runtime_id']} status=PASS",
            flush=True,
        )
        return 0

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-r2-onnx-runtime: {exc}", file=sys.stderr)
        raise
