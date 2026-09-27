from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .onnx_export import load_r2_onnx_manifest
from .onnx_hardening import (
    seal_r2e5_hardening,
    verify_r2e5_hardening,
    verify_r2e5_run,
)
from .onnx_profiling import verify_r2e3_receipt
from .onnx_autotune import verify_r2e4_profile


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify Galaxy S21 FE VN97-R2E5 device evidence and seal "
            "a benchmark/hardening result."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    verify_run = sub.add_parser("verify-run")
    verify_run.add_argument("--bundle-dir", required=True)
    verify_run.add_argument("--e3-receipt", required=True)
    verify_run.add_argument("--e4-profile", required=True)
    verify_run.add_argument("--run", required=True)

    seal = sub.add_parser("seal")
    seal.add_argument("--bundle-dir", required=True)
    seal.add_argument("--e3-receipt", required=True)
    seal.add_argument("--e4-profile", required=True)
    seal.add_argument("--run", required=True)
    seal.add_argument("--output", required=True)
    seal.add_argument(
        "--allow-non-s21fe",
        action="store_true",
        help="Allow a non-S21-FE device only for development evidence.",
    )

    verify = sub.add_parser("verify")
    verify.add_argument("--bundle-dir", required=True)
    verify.add_argument("--hardening", required=True)

    status = sub.add_parser("status")
    status.add_argument("--bundle-dir", required=True)
    status.add_argument("--hardening", required=True)
    return parser


def _bound_inputs(args) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    manifest = load_r2_onnx_manifest(Path(args.bundle_dir))
    bundle_id = str(manifest["bundle_id"])
    e3 = verify_r2e3_receipt(
        Path(args.e3_receipt),
        expected_bundle_id=bundle_id,
    )
    e4 = verify_r2e4_profile(
        Path(args.e4_profile),
        expected_bundle_id=bundle_id,
    )
    if e4.get("e3_receipt_id") != e3.get("receipt_id"):
        raise ValueError(
            "E5 E4 profile is not bound to the supplied E3 receipt"
        )
    return manifest, e3, e4


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "verify-run":
        manifest, e3, e4 = _bound_inputs(args)
        run = verify_r2e5_run(
            Path(args.run),
            expected_bundle_id=str(manifest["bundle_id"]),
            expected_e3_receipt_id=str(e3["receipt_id"]),
            expected_tuning_id=str(e4["tuning_id"]),
        )
        print(
            "VN97R2E5 "
            f"command=verify-run run={run['run_id']} "
            f"model={run['device']['model']} status=PASS",
            flush=True,
        )
        return 0

    if args.command == "seal":
        sealed = seal_r2e5_hardening(
            bundle_dir=Path(args.bundle_dir),
            e3_receipt_path=Path(args.e3_receipt),
            e4_profile_path=Path(args.e4_profile),
            run_path=Path(args.run),
            output_path=Path(args.output),
            require_s21_fe=not args.allow_non_s21fe,
        )
        print(
            "VN97R2E5 "
            f"command=seal hardening={sealed['hardening_id']} "
            f"target_match={str(sealed['target_device_match']).lower()} "
            f"passed={str(sealed['hardening_passed']).lower()}",
            flush=True,
        )
        return 0

    manifest = load_r2_onnx_manifest(Path(args.bundle_dir))
    sealed = verify_r2e5_hardening(
        Path(args.hardening),
        expected_bundle_id=str(manifest["bundle_id"]),
    )
    if sealed.get("architecture_fingerprint") != manifest.get(
        "architecture_fingerprint"
    ):
        raise ValueError(
            "E5 hardening architecture differs from E2 bundle"
        )

    if args.command == "verify":
        print(
            "VN97R2E5 "
            f"command=verify hardening={sealed['hardening_id']} "
            f"passed={str(sealed['hardening_passed']).lower()} "
            "status=PASS",
            flush=True,
        )
        return 0

    print(
        json.dumps(
            {
                "hardening_id": sealed["hardening_id"],
                "bundle_id": sealed["bundle_id"],
                "e3_receipt_id": sealed["e3_receipt_id"],
                "e4_tuning_id": sealed["e4_tuning_id"],
                "run_id": sealed["run_id"],
                "device": sealed["device"],
                "warm_step_p95_ns": sealed["warm_step_p95_ns"],
                "sustained_p95_ns": sealed["sustained_p95_ns"],
                "sustained_p95_over_warm_ppm": sealed[
                    "sustained_p95_over_warm_ppm"
                ],
                "failure_count": sealed["failure_count"],
                "minimal_apk_graphs": sealed["minimal_apk_graphs"],
                "hardening_passed": sealed["hardening_passed"],
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
        print(f"vn97-r2-onnx-harden: {exc}", file=sys.stderr)
        raise
