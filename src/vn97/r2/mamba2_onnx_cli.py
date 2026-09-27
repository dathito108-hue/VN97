from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .mamba2_onnx import (
    export_g03_capsule_step_onnx,
    validate_ort_step_parity,
    verify_mamba2_g04_bundle,
)
from .mamba2_g03_capsule import load_g03_capsule
from .mamba2_onnx import Mamba2OnnxConfig, VN97Mamba2StepOnnx


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "VN97 R2-G0.4 explicit-state Mamba-2 ONNX step lowering."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export-step")
    export.add_argument("--capsule-root", required=True, type=Path)
    export.add_argument("--output-dir", required=True, type=Path)
    export.add_argument("--example-batch-size", type=int, default=1)

    verify = sub.add_parser("verify")
    verify.add_argument("--bundle-dir", required=True, type=Path)

    parity = sub.add_parser("parity")
    parity.add_argument("--capsule-root", required=True, type=Path)
    parity.add_argument("--bundle-dir", required=True, type=Path)
    parity.add_argument("--batch-size", type=int, default=1)
    parity.add_argument("--seed", type=int, default=9704)
    parity.add_argument("--max-abs-error", type=float, default=1.0e-3)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "export-step":
        manifest = export_g03_capsule_step_onnx(
            args.capsule_root,
            args.output_dir,
            example_batch_size=args.example_batch_size,
        )
        print(
            json.dumps(
                {
                    "status": "VN97_G04_STEP_ONNX_EXPORTED",
                    "manifest_id": manifest["manifest_id"],
                    "capsule_id": manifest["capsule_id"],
                    "graph_kind": manifest["graph_kind"],
                    "state_contract": manifest["state_contract"],
                    "parallel_prefill_ready":
                        manifest["parallel_prefill_ready"],
                    "production_activation_authorized":
                        manifest["production_activation_authorized"],
                },
                sort_keys=True,
            )
        )
        return 0

    if args.command == "verify":
        manifest = verify_mamba2_g04_bundle(args.bundle_dir)
        print(
            json.dumps(
                {
                    "status": "VN97_G04_ONNX_BUNDLE_VERIFIED",
                    "manifest_id": manifest["manifest_id"],
                    "capsule_id": manifest["capsule_id"],
                    "graph_files": manifest["graph_files"],
                    "production_activation_authorized": False,
                },
                sort_keys=True,
            )
        )
        return 0

    if args.max_abs_error <= 0.0:
        raise ValueError("--max-abs-error must be positive")
    capsule = load_g03_capsule(
        args.capsule_root,
        verify_large_weight_sha256=True,
    )
    model = VN97Mamba2StepOnnx(
        Mamba2OnnxConfig.from_source_spec(capsule.spec),
        capsule.tensors,
    ).eval()
    metrics = validate_ort_step_parity(
        model,
        args.bundle_dir,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    worst = max(metrics.values())
    if worst > args.max_abs_error:
        raise ValueError(
            f"G0.4 ORT parity failed: max error {worst} "
            f"> {args.max_abs_error}"
        )
    print(
        json.dumps(
            {
                "status": "VN97_G04_ORT_STEP_PARITY_PASS",
                "metrics": metrics,
                "max_abs_error_limit": args.max_abs_error,
                "production_activation_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-r2-mamba2-g04: {exc}", file=sys.stderr)
        raise
