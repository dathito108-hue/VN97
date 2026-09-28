from __future__ import annotations

import argparse
import json
from pathlib import Path

from .mamba2_cognition_bridge import (
    VN97_MAMBA2_G09_BRIDGE_FILENAME,
    build_g09_bridge_binding,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the VN97 R2-G0.9 identity-bound Mamba-2 "
            "candidate cognition bridge."
        )
    )
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--tuning", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output = args.output or (
        args.runtime.parent / VN97_MAMBA2_G09_BRIDGE_FILENAME
    )
    result = build_g09_bridge_binding(
        runtime_path=args.runtime,
        tokenizer_path=args.tokenizer,
        tuning_path=args.tuning,
        output_path=output,
    )
    print(
        json.dumps(
            {
                "status": "VN97_MAMBA2_G09_BRIDGE_CREATED",
                "bridge_id": result["bridge_id"],
                "runtime_id": result["runtime_id"],
                "tokenizer_id": result["tokenizer_id"],
                "tuning_id": result["tuning_id"],
                "candidate_validation_only":
                    result["candidate_validation_only"],
                "production_activation_authorized":
                    result["production_activation_authorized"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
