from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .onnx_autotune import (
    build_r2e4_profile_from_files,
    verify_r2e4_profile,
)
from .onnx_export import load_r2_onnx_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build or verify a VN97-R2E4 dynamic ONNX runtime tuning "
            "profile from one verified E3 device receipt."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--bundle-dir", required=True)
    build.add_argument("--receipt", required=True)
    build.add_argument("--output", required=True)

    verify = sub.add_parser("verify")
    verify.add_argument("--bundle-dir", required=True)
    verify.add_argument("--profile", required=True)

    status = sub.add_parser("status")
    status.add_argument("--bundle-dir", required=True)
    status.add_argument("--profile", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "build":
        profile = build_r2e4_profile_from_files(
            bundle_dir=Path(args.bundle_dir),
            receipt_path=Path(args.receipt),
            output_path=Path(args.output),
        )
        print(
            "VN97R2E4 "
            f"command=build tuning={profile['tuning_id']} "
            f"bundle={profile['bundle_id']} "
            f"device={profile['device_key']}",
            flush=True,
        )
        return 0

    manifest = load_r2_onnx_manifest(Path(args.bundle_dir))
    profile = verify_r2e4_profile(
        Path(args.profile),
        expected_bundle_id=str(manifest["bundle_id"]),
    )
    if profile.get("architecture_fingerprint") != manifest.get(
        "architecture_fingerprint"
    ):
        raise ValueError(
            "E4 profile architecture differs from E2 bundle"
        )
    if profile.get("profile") != manifest.get("profile"):
        raise ValueError("E4 profile fast/deep mode differs from E2 bundle")

    if args.command == "verify":
        print(
            "VN97R2E4 "
            f"command=verify tuning={profile['tuning_id']} "
            f"bundle={profile['bundle_id']} "
            "status=PASS",
            flush=True,
        )
        return 0

    print(
        json.dumps(
            {
                "tuning_id": profile["tuning_id"],
                "bundle_id": profile["bundle_id"],
                "e3_receipt_id": profile["e3_receipt_id"],
                "device_key": profile["device_key"],
                "device": profile["device"],
                "preferred_chunk_sizes": profile[
                    "preferred_chunk_sizes"
                ],
                "graph_policies": profile["graph_policies"],
                "control_policy": profile["control_policy"],
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
        print(f"vn97-r2-onnx-autotune: {exc}", file=sys.stderr)
        raise
