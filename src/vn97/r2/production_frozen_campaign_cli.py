from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .production_frozen_campaign import (
    assert_r2d13_preflight_allowed,
    build_r2d13_campaign,
    seal_r2d13_ready,
    verify_r2d13_campaign,
    verify_r2d13_ready,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build, verify and gate a VN97-R2D13 frozen production "
            "campaign for measured T4 preflight and quota-bounded D7."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--view-dir", required=True)
    build.add_argument("--workspace-root", required=True)
    build.add_argument("--curriculum-plan", required=True)
    build.add_argument("--preflight-package", required=True)
    build.add_argument("--repository-commit", required=True)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--learning-rate", type=float, default=1e-4)
    build.add_argument("--weight-decay", type=float, default=0.01)
    build.add_argument("--max-grad-norm", type=float, default=1.0)
    build.add_argument("--checkpoint-every", type=int, default=25)
    build.add_argument("--max-run-seconds", type=float, default=3000.0)

    verify = sub.add_parser("verify")
    verify.add_argument("--package-dir", required=True)
    verify.add_argument("--workspace-root", required=True)

    gate = sub.add_parser("gate-preflight")
    gate.add_argument("--package-dir", required=True)
    gate.add_argument("--workspace-root", required=True)

    seal = sub.add_parser("seal-ready")
    seal.add_argument("--package-dir", required=True)
    seal.add_argument("--workspace-root", required=True)
    seal.add_argument("--preflight-receipt", required=True)
    seal.add_argument("--output", required=True)

    ready = sub.add_parser("verify-ready")
    ready.add_argument("--package-dir", required=True)
    ready.add_argument("--workspace-root", required=True)
    ready.add_argument("--ready-receipt", required=True)
    ready.add_argument("--preflight-receipt", required=True)

    status = sub.add_parser("status")
    status.add_argument("--package-dir", required=True)
    status.add_argument("--workspace-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "build":
        campaign = build_r2d13_campaign(
            view_dir=Path(args.view_dir),
            workspace_root=Path(args.workspace_root),
            curriculum_plan_path=Path(args.curriculum_plan),
            preflight_package_dir=Path(args.preflight_package),
            repository_commit=args.repository_commit,
            output_dir=Path(args.output_dir),
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            checkpoint_every_optimizer_steps=args.checkpoint_every,
            max_run_seconds=args.max_run_seconds,
        )
        print(
            "VN97R2D13 "
            f"command=build campaign={campaign['campaign_id']} "
            f"view={campaign['virtual_view_id']} "
            f"projection={campaign['projected_index_id']} "
            f"scale_floor_passed="
            f"{str(campaign['scale_floor_passed']).lower()} "
            f"gpu_preflight_allowed="
            f"{str(campaign['gpu_preflight_allowed']).lower()}",
            flush=True,
        )
        return 0

    if args.command == "verify":
        verified = verify_r2d13_campaign(
            Path(args.package_dir),
            workspace_root=Path(args.workspace_root),
        )
        campaign = verified["campaign"]
        print(
            "VN97R2D13 "
            f"command=verify campaign={campaign['campaign_id']} "
            f"scale_floor_passed="
            f"{str(campaign['scale_floor_passed']).lower()} "
            "status=PASS",
            flush=True,
        )
        return 0

    if args.command == "gate-preflight":
        verified = assert_r2d13_preflight_allowed(
            Path(args.package_dir),
            workspace_root=Path(args.workspace_root),
        )
        campaign = verified["campaign"]
        print(
            "VN97R2D13 "
            f"command=gate-preflight campaign={campaign['campaign_id']} "
            "gpu_preflight_allowed=true",
            flush=True,
        )
        return 0

    if args.command == "seal-ready":
        ready = seal_r2d13_ready(
            package_dir=Path(args.package_dir),
            workspace_root=Path(args.workspace_root),
            preflight_receipt_path=Path(args.preflight_receipt),
            output_path=Path(args.output),
        )
        print(
            "VN97R2D13 "
            f"command=seal-ready ready={ready['ready_id']} "
            f"device={json.dumps(ready['device_name'])} "
            "training_allowed=true",
            flush=True,
        )
        return 0

    if args.command == "verify-ready":
        verified = verify_r2d13_ready(
            package_dir=Path(args.package_dir),
            workspace_root=Path(args.workspace_root),
            ready_receipt_path=Path(args.ready_receipt),
            preflight_receipt_path=Path(args.preflight_receipt),
        )
        ready = verified["ready"]
        print(
            "VN97R2D13 "
            f"command=verify-ready ready={ready['ready_id']} "
            "training_allowed=true status=PASS",
            flush=True,
        )
        return 0

    verified = verify_r2d13_campaign(
        Path(args.package_dir),
        workspace_root=Path(args.workspace_root),
    )
    campaign = verified["campaign"]
    print(
        json.dumps(
            {
                "campaign_id": campaign["campaign_id"],
                "repository_commit": campaign["repository_commit"],
                "stage": campaign["stage"],
                "virtual_view_id": campaign["virtual_view_id"],
                "registry_generation": campaign["registry_generation"],
                "projected_index_id": campaign["projected_index_id"],
                "curriculum_plan_id": campaign["curriculum_plan_id"],
                "preflight_campaign_id": campaign[
                    "preflight_campaign_id"
                ],
                "recipe_fingerprint": campaign["recipe_fingerprint"],
                "scale": campaign["scale"],
                "scale_floor_passed": campaign["scale_floor_passed"],
                "gpu_preflight_allowed": campaign[
                    "gpu_preflight_allowed"
                ],
                "training_launch_allowed": campaign[
                    "training_launch_allowed"
                ],
                "max_run_seconds": campaign["max_run_seconds"],
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
        print(
            f"vn97-r2-production-campaign: {exc}",
            file=sys.stderr,
        )
        raise
