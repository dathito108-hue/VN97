from __future__ import annotations

import argparse
import json
from pathlib import Path

from .mamba2_android_runtime import build_g06_runtime_descriptor


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the VN97 R2-G0.6 Android runtime descriptor from "
            "a verified G0.5 recurrent ONNX bundle."
        )
    )
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    descriptor = build_g06_runtime_descriptor(
        bundle_dir=args.bundle_dir,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "status": "VN97_MAMBA2_G06_RUNTIME_DESCRIPTOR_CREATED",
                "runtime_id": descriptor["runtime_id"],
                "g05_manifest_id": descriptor["g05_manifest_id"],
                "graph_filename": descriptor["graph_filename"],
                "max_chunk_size": descriptor["max_chunk_size"],
                "state_dtype": descriptor["state_dtype"],
                "single_weight_graph": descriptor["single_weight_graph"],
                "parallel_prefill_ready": descriptor[
                    "parallel_prefill_ready"
                ],
                "production_activation_authorized": descriptor[
                    "production_activation_authorized"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
