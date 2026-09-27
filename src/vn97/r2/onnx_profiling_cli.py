from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .onnx_export import load_r2_onnx_manifest
from .onnx_profiling import verify_r2e3_receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify or inspect a device-measured VN97-R2E3 ONNX "
            "provider profiling receipt."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    verify = sub.add_parser("verify")
    verify.add_argument("--bundle-dir", required=True)
    verify.add_argument("--receipt", required=True)

    status = sub.add_parser("status")
    status.add_argument("--bundle-dir", required=True)
    status.add_argument("--receipt", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = load_r2_onnx_manifest(Path(args.bundle_dir))
    receipt = verify_r2e3_receipt(
        Path(args.receipt),
        expected_bundle_id=str(manifest["bundle_id"]),
    )
    if receipt.get("architecture_fingerprint") != manifest.get(
        "architecture_fingerprint"
    ):
        raise ValueError(
            "E3 receipt architecture differs from E2 bundle"
        )
    if receipt.get("profile") != manifest.get("profile"):
        raise ValueError("E3 receipt profile differs from E2 bundle")

    if args.command == "verify":
        print(
            "VN97R2E3 "
            f"command=verify receipt={receipt['receipt_id']} "
            f"bundle={receipt['bundle_id']} "
            "status=PASS",
            flush=True,
        )
        return 0

    print(
        json.dumps(
            {
                "receipt_id": receipt["receipt_id"],
                "bundle_id": receipt["bundle_id"],
                "profile": receipt["profile"],
                "device": receipt["device"],
                "graph_results": receipt["graph_results"],
                "failures": receipt["failures"],
            },
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
        print(f"vn97-r2-onnx-profile: {exc}", file=sys.stderr)
        raise
