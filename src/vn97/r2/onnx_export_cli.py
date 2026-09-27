from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .onnx_export import (
    R2_ONNX_SUPPORTED_CHUNKS,
    export_r2_checkpoint_onnx_bundle,
    validate_ort_parity,
    verify_r2_onnx_bundle,
)
from .checkpoint import load_r2_checkpoint


def _chunks(value: str) -> tuple[int, ...]:
    try:
        items = tuple(
            sorted(
                {
                    int(part.strip())
                    for part in value.split(",")
                    if part.strip()
                }
            )
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "chunk sizes must be comma-separated integers"
        ) from exc
    if not items:
        raise argparse.ArgumentTypeError(
            "at least one chunk size is required"
        )
    unsupported = [
        item
        for item in items
        if item not in R2_ONNX_SUPPORTED_CHUNKS
    ]
    if unsupported:
        raise argparse.ArgumentTypeError(
            "unsupported chunk sizes: "
            + ",".join(str(item) for item in unsupported)
        )
    return items


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export or verify VN97-R2 explicit-state ONNX step/chunk "
            "graphs for Android ONNX Runtime."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output-dir", required=True)
    export.add_argument(
        "--profile",
        choices=("fast", "deep"),
        default="deep",
    )
    export.add_argument(
        "--chunks",
        type=_chunks,
        default=(32,),
        help="comma-separated subset of 8,16,32,64,128",
    )
    export.add_argument(
        "--example-batch-size",
        type=int,
        default=1,
    )

    verify = sub.add_parser("verify")
    verify.add_argument("--bundle-dir", required=True)

    parity = sub.add_parser("parity")
    parity.add_argument("--checkpoint", required=True)
    parity.add_argument("--bundle-dir", required=True)
    parity.add_argument(
        "--profile",
        choices=("fast", "deep"),
        default="deep",
    )
    parity.add_argument("--chunk-size", type=int, default=None)
    parity.add_argument("--batch-size", type=int, default=1)
    parity.add_argument("--seed", type=int, default=9702)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "export":
        manifest = export_r2_checkpoint_onnx_bundle(
            Path(args.checkpoint),
            Path(args.output_dir),
            profile=args.profile,
            chunk_sizes=args.chunks,
            example_batch_size=args.example_batch_size,
        )
        print(
            "VN97R2E2 "
            f"command=export bundle={manifest['bundle_id']} "
            f"profile={manifest['profile']} "
            f"active_layers={manifest['active_layers']} "
            f"chunks={','.join(map(str, manifest['supported_chunk_sizes']))}",
            flush=True,
        )
        return 0

    if args.command == "verify":
        manifest = verify_r2_onnx_bundle(
            Path(args.bundle_dir)
        )
        print(
            "VN97R2E2 "
            f"command=verify bundle={manifest['bundle_id']} "
            f"profile={manifest['profile']} "
            "status=PASS",
            flush=True,
        )
        return 0

    model, evidence = load_r2_checkpoint(
        Path(args.checkpoint),
        map_location="cpu",
    )
    metrics = validate_ort_parity(
        model,
        Path(args.bundle_dir),
        profile=args.profile,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "checkpoint_sha256": evidence["sha256"],
                "profile": args.profile,
                "chunk_size": args.chunk_size,
                "batch_size": args.batch_size,
                "metrics": metrics,
                "status": "PASS",
            },
            sort_keys=True,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-r2-onnx-export: {exc}", file=sys.stderr)
        raise
