from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .mamba2_g03_capsule import load_g03_capsule
from .mamba2_onnx import (
    Mamba2OnnxConfig,
    VN97Mamba2StepOnnx,
)
from .mamba2_parallel_onnx import (
    VN97_MAMBA2_G05_CHUNKS,
    VN97Mamba2ParallelChunkOnnx,
    export_g03_capsule_parallel_onnx,
    validate_ort_parallel_parity,
    validate_parallel_parity,
    verify_g05_bundle,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "VN97 R2-G0.5 single-graph Mamba-2 parallel SSD prefill."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export")
    export.add_argument("--capsule-root", required=True, type=Path)
    export.add_argument("--output-dir", required=True, type=Path)
    export.add_argument(
        "--chunk-size",
        type=int,
        choices=VN97_MAMBA2_G05_CHUNKS,
        default=32,
    )

    verify = sub.add_parser("verify")
    verify.add_argument("--bundle-dir", required=True, type=Path)

    parity = sub.add_parser("parity")
    parity.add_argument("--capsule-root", required=True, type=Path)
    parity.add_argument("--bundle-dir", required=True, type=Path)
    parity.add_argument(
        "--valid-length",
        type=int,
        required=True,
    )
    parity.add_argument("--seed", type=int, default=9706)
    parity.add_argument("--max-logit-error", type=float, default=2.0e-3)
    parity.add_argument("--max-state-error", type=float, default=2.0e-3)

    native = sub.add_parser("native-parity")
    native.add_argument("--capsule-root", required=True, type=Path)
    native.add_argument(
        "--chunk-size",
        type=int,
        choices=VN97_MAMBA2_G05_CHUNKS,
        default=32,
    )
    native.add_argument("--valid-length", type=int, required=True)
    native.add_argument("--seed", type=int, default=9705)

    return parser


def _model(
    capsule_root: Path,
    chunk_size: int,
) -> VN97Mamba2ParallelChunkOnnx:
    capsule = load_g03_capsule(
        capsule_root,
        verify_large_weight_sha256=True,
    )
    step = VN97Mamba2StepOnnx(
        Mamba2OnnxConfig.from_source_spec(capsule.spec),
        capsule.tensors,
    ).eval()
    return VN97Mamba2ParallelChunkOnnx(
        step,
        chunk_size=chunk_size,
    ).eval()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "export":
        manifest = export_g03_capsule_parallel_onnx(
            args.capsule_root,
            args.output_dir,
            chunk_size=args.chunk_size,
        )
        print(
            json.dumps(
                {
                    "status": "VN97_G05_PARALLEL_ONNX_EXPORTED",
                    "manifest_id": manifest["manifest_id"],
                    "capsule_id": manifest["capsule_id"],
                    "max_chunk_size": manifest["max_chunk_size"],
                    "single_weight_graph":
                        manifest["single_weight_graph"],
                    "decode_via_valid_length_one":
                        manifest["decode_via_valid_length_one"],
                    "parallel_prefill_ready":
                        manifest["parallel_prefill_ready"],
                    "production_activation_authorized": False,
                },
                sort_keys=True,
            )
        )
        return 0

    if args.command == "verify":
        manifest = verify_g05_bundle(args.bundle_dir)
        print(
            json.dumps(
                {
                    "status": "VN97_G05_BUNDLE_VERIFIED",
                    "manifest_id": manifest["manifest_id"],
                    "max_chunk_size": manifest["max_chunk_size"],
                    "parallel_algorithm": manifest["parallel_algorithm"],
                    "production_activation_authorized": False,
                },
                sort_keys=True,
            )
        )
        return 0

    if args.command == "native-parity":
        model = _model(args.capsule_root, args.chunk_size)
        metrics = validate_parallel_parity(
            model,
            valid_length=args.valid_length,
            seed=args.seed,
        )
        print(
            json.dumps(
                {
                    "status": "VN97_G05_NATIVE_PARITY_MEASURED",
                    "valid_length": args.valid_length,
                    "metrics": metrics,
                    "production_activation_authorized": False,
                },
                sort_keys=True,
            )
        )
        return 0

    manifest = verify_g05_bundle(args.bundle_dir)
    chunk_size = int(manifest["max_chunk_size"])
    model = _model(args.capsule_root, chunk_size)
    metrics = validate_ort_parallel_parity(
        model,
        args.bundle_dir,
        valid_length=args.valid_length,
        seed=args.seed,
    )
    if metrics["max_logits_abs_error"] > args.max_logit_error:
        raise ValueError("G0.5 logits parity exceeds configured tolerance")
    if max(
        metrics["max_conv_state_abs_error"],
        metrics["max_ssm_state_abs_error"],
    ) > args.max_state_error:
        raise ValueError("G0.5 state parity exceeds configured tolerance")
    print(
        json.dumps(
            {
                "status": "VN97_G05_ORT_PARITY_PASS",
                "valid_length": args.valid_length,
                "metrics": metrics,
                "max_logit_error": args.max_logit_error,
                "max_state_error": args.max_state_error,
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
        print(f"vn97-r2-mamba2-g05: {exc}", file=sys.stderr)
        raise
