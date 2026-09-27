from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .mamba2_dynamic_onnx import (
    export_g03_capsule_dynamic_onnx,
    verify_g06_bundle,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "VN97 R2-G0.6 dynamic-sequence Mamba-2 ONNX export."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export")
    export.add_argument("--capsule-root", required=True, type=Path)
    export.add_argument("--output-dir", required=True, type=Path)
    export.add_argument(
        "--max-sequence-length",
        type=int,
        default=32,
    )

    verify = sub.add_parser("verify")
    verify.add_argument("--bundle-dir", required=True, type=Path)

    args = parser.parse_args(argv)

    if args.command == "export":
        manifest = export_g03_capsule_dynamic_onnx(
            args.capsule_root,
            args.output_dir,
            max_sequence_length=args.max_sequence_length,
        )
        print(
            json.dumps(
                {
                    "status": "VN97_G06_DYNAMIC_ONNX_EXPORTED",
                    "manifest_id": manifest["manifest_id"],
                    "capsule_id": manifest["capsule_id"],
                    "sequence_length_min":
                        manifest["sequence_length_min"],
                    "sequence_length_max":
                        manifest["sequence_length_max"],
                    "state_dtype":
                        manifest["state_contract"]["dtype"],
                    "logits_dtype": manifest["logits_dtype"],
                    "single_weight_graph":
                        manifest["single_weight_graph"],
                    "production_activation_authorized": False,
                },
                sort_keys=True,
            )
        )
        return 0

    manifest = verify_g06_bundle(args.bundle_dir)
    print(
        json.dumps(
            {
                "status": "VN97_G06_DYNAMIC_ONNX_VERIFIED",
                "manifest_id": manifest["manifest_id"],
                "graph_filename": manifest["graph_filename"],
                "sequence_length_max":
                    manifest["sequence_length_max"],
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
        print(f"vn97-r2-mamba2-g06: {exc}", file=sys.stderr)
        raise
