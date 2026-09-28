from __future__ import annotations

import argparse
import json
from pathlib import Path

from .mamba2_onnx import (
    build_mamba2_onnx_manifest,
    capsule_identity_for_onnx,
    load_capsule_for_onnx,
    official_27b_state_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "VN97 R2-G0.4 Mamba-2 explicit-state ONNX contract tooling."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    contract = sub.add_parser("contract")

    bind = sub.add_parser("bind-capsule")
    bind.add_argument("--capsule-root", required=True, type=Path)

    args = parser.parse_args()

    if args.command == "contract":
        print(
            json.dumps(
                {
                    "status": "VN97_MAMBA2_ONNX_CONTRACT_LOCKED",
                    "state_contract":
                        official_27b_state_contract().canonical_object(),
                    "production_activation_authorized": False,
                },
                sort_keys=True,
            )
        )
        return

    capsule = load_capsule_for_onnx(args.capsule_root)
    capsule_id, manifest_sha = capsule_identity_for_onnx(capsule)
    print(
        json.dumps(
            {
                "status": "VN97_MAMBA2_CAPSULE_BOUND_FOR_ONNX",
                "capsule_id": capsule_id,
                "capsule_manifest_sha256": manifest_sha,
                "source_weight_sha256":
                    capsule.manifest.source_weight_sha256,
                "d_model": capsule.spec.d_model,
                "n_layers": capsule.spec.n_layers,
                "n_heads": capsule.spec.n_heads,
                "d_state": capsule.spec.d_state,
                "production_activation_authorized": False,
                "next_gate": "EXPORT_REAL_STEP_AND_CHUNK_GRAPHS",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
