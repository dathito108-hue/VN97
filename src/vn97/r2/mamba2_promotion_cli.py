from __future__ import annotations

import argparse
import json
from pathlib import Path

from .mamba2_promotion import build_g10_promotion


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compile the VN97 R2-G0.10 Mamba-2 production promotion "
            "receipt from real model/token/mobile evidence."
        )
    )
    parser.add_argument("--bridge", required=True, type=Path)
    parser.add_argument("--model-parity", required=True, type=Path)
    parser.add_argument("--token-parity", required=True, type=Path)
    parser.add_argument("--mobile-qualification", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    promotion = build_g10_promotion(
        bridge_path=args.bridge,
        model_parity_path=args.model_parity,
        token_parity_path=args.token_parity,
        mobile_qualification_path=args.mobile_qualification,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "status": "VN97_MAMBA2_G10_PROMOTION_CREATED",
                "promotion_id": promotion["promotion_id"],
                "bridge_id": promotion["bridge_id"],
                "runtime_id": promotion["runtime_id"],
                "tokenizer_id": promotion["tokenizer_id"],
                "tuning_id": promotion["tuning_id"],
                "target_family": promotion["target_family"],
                "production_activation_authorized":
                    promotion["production_activation_authorized"],
                "rollback_required": promotion["rollback_required"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
