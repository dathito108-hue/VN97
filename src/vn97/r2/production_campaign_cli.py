from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .production_campaign import (
    R2D5_MAX_SPLIT_BYTES,
    build_r2d5_preflight_package,
    seal_r2d5_preflight,
    verify_r2d5_package,
)
from .production_contract import R2ProductionTrainingRecipe


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build, verify, or seal a self-contained VN97-R2D5 "
            "measured-T4-preflight campaign package."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--corpus-dir", required=True)
    build.add_argument("--tokenizer", required=True)
    build.add_argument("--repository-commit", required=True)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--task-family", action="append", required=True)
    build.add_argument("--sequence-length", type=int, default=128)
    build.add_argument("--micro-batch-size", type=int, default=1)
    build.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
    )
    build.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="fp16",
    )
    build.add_argument("--safety-fraction", type=float, default=0.90)
    build.add_argument(
        "--max-split-bytes",
        type=int,
        default=R2D5_MAX_SPLIT_BYTES,
    )

    verify = sub.add_parser("verify")
    verify.add_argument("--package-dir", required=True)

    seal = sub.add_parser("seal-preflight")
    seal.add_argument("--package-dir", required=True)
    seal.add_argument("--preflight-bundle", required=True)
    seal.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "build":
        recipe = R2ProductionTrainingRecipe(
            sequence_length=args.sequence_length,
            micro_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=(
                args.gradient_accumulation_steps
            ),
            precision=args.precision,
            activation_checkpointing=True,
            memory_efficient_scan=True,
            optimizer_state_offload=True,
            full_parameter_training=True,
            quantization_used=False,
        )
        payload = build_r2d5_preflight_package(
            corpus_dir=Path(args.corpus_dir),
            tokenizer_path=Path(args.tokenizer),
            repository_commit=args.repository_commit,
            output_dir=Path(args.output_dir),
            task_families=args.task_family,
            recipe=recipe,
            safety_fraction=args.safety_fraction,
            max_split_bytes=args.max_split_bytes,
        )
        print(
            "VN97R2D5 "
            f"command=build campaign={payload['campaign_id']} "
            f"parameters={payload['model_parameter_count']} "
            "training_allowed=false",
            flush=True,
        )
        return 0

    if args.command == "verify":
        payload = verify_r2d5_package(
            Path(args.package_dir)
        )
        print(
            "VN97R2D5 "
            f"command=verify campaign={payload['campaign_id']} "
            "status=PASS training_allowed=false",
            flush=True,
        )
        return 0

    receipt = seal_r2d5_preflight(
        package_dir=Path(args.package_dir),
        preflight_bundle_path=Path(args.preflight_bundle),
        output_path=Path(args.output),
    )
    print(
        "VN97R2D5 "
        f"command=seal-preflight receipt={receipt['receipt_id']} "
        f"device={json.dumps(receipt['device_name'])} "
        f"peak_reserved={receipt['peak_reserved_bytes']} "
        "training_allowed=false",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-r2-preflight-package: {exc}", file=sys.stderr)
        raise
