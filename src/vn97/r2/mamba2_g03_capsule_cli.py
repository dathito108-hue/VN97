from __future__ import annotations

import argparse
import json
from pathlib import Path

from .mamba2_g03_capsule import (
    VN97_MAMBA2_G03_CAPSULE_MANIFEST,
    load_g03_capsule,
    materialize_g03_capsule,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "VN97 R2-G0.3 zero-copy Mamba-2 2.7B capsule materialization."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    materialize = sub.add_parser("materialize")
    materialize.add_argument("--source-root", required=True, type=Path)
    materialize.add_argument("--output-root", required=True, type=Path)

    verify = sub.add_parser("verify")
    verify.add_argument("--capsule-root", required=True, type=Path)
    verify.add_argument(
        "--skip-large-weight-sha256",
        action="store_true",
        help="Development only. Production verification must hash the 5.4GB payload.",
    )

    inspect = sub.add_parser("inspect")
    inspect.add_argument("--capsule-root", required=True, type=Path)

    args = parser.parse_args()

    if args.command == "materialize":
        manifest = materialize_g03_capsule(
            args.source_root,
            args.output_root,
        )
        print(
            json.dumps(
                {
                    "status": "VN97_G03_ZERO_COPY_CAPSULE_CREATED",
                    "capsule_root": str(args.output_root),
                    "capsule_manifest": str(
                        args.output_root / VN97_MAMBA2_G03_CAPSULE_MANIFEST
                    ),
                    "capsule_id": manifest.capsule_id(),
                    "source_weight_sha256": manifest.source_weight_sha256,
                    "source_weight_size_bytes":
                        manifest.source_weight_size_bytes,
                    "transfer_semantics": manifest.transfer_semantics,
                    "logical_namespace": manifest.logical_namespace,
                    "zero_copy_materialization":
                        manifest.zero_copy_materialization,
                    "source_runtime_required":
                        manifest.source_runtime_required,
                    "production_parity_required": True,
                    "production_activation_authorized": False,
                },
                sort_keys=True,
            )
        )
        return

    loaded = load_g03_capsule(
        args.capsule_root,
        verify_large_weight_sha256=(
            False
            if args.command == "inspect"
            else not args.skip_large_weight_sha256
        ),
    )
    result = {
        "status": (
            "VN97_G03_CAPSULE_VERIFIED"
            if args.command == "verify"
            else "VN97_G03_CAPSULE_INSPECTED"
        ),
        "capsule_root": str(loaded.root),
        "capsule_id": loaded.manifest.capsule_id(),
        "manifest_sha256": loaded.manifest_sha256,
        "source_weight_sha256":
            loaded.manifest.source_weight_sha256,
        "source_weight_size_bytes":
            loaded.manifest.source_weight_size_bytes,
        "logical_tensor_count": len(loaded.tensors),
        "d_model": loaded.spec.d_model,
        "n_layers": loaded.spec.n_layers,
        "n_heads": loaded.spec.n_heads,
        "d_state": loaded.spec.d_state,
        "source_runtime_required":
            loaded.manifest.source_runtime_required,
        "large_weight_sha256_verified": (
            args.command == "verify"
            and not args.skip_large_weight_sha256
        ),
        "production_parity_required": True,
        "production_activation_authorized": False,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
