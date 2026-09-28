from __future__ import annotations

import argparse
import json
from pathlib import Path

from .mamba2_autotune import (
    VN97_MAMBA2_G07_TUNE_FILENAME,
    build_g07_tuning_file,
    verify_g07_tuning,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compile or verify VN97 R2-G0.7 device-measured Mamba-2 "
            "provider autotuning."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    compile_cmd = sub.add_parser("compile")
    compile_cmd.add_argument("--profile", required=True, type=Path)
    compile_cmd.add_argument("--output", type=Path)
    compile_cmd.add_argument("--runtime-id")

    verify_cmd = sub.add_parser("verify")
    verify_cmd.add_argument("--tuning", required=True, type=Path)
    verify_cmd.add_argument("--runtime-id")

    args = parser.parse_args()

    if args.command == "compile":
        output = args.output or (
            args.profile.parent / VN97_MAMBA2_G07_TUNE_FILENAME
        )
        result = build_g07_tuning_file(
            profile_path=args.profile,
            output_path=output,
            expected_runtime_id=args.runtime_id,
        )
        print(
            json.dumps(
                {
                    "status": "VN97_MAMBA2_G07_TUNING_CREATED",
                    "tuning_id": result["tuning_id"],
                    "runtime_id": result["runtime_id"],
                    "graph_filename": result["graph_filename"],
                    "max_chunk_size": result["max_chunk_size"],
                    "profile_is_device_measured":
                        result["profile_is_device_measured"],
                    "production_activation_authorized":
                        result["production_activation_authorized"],
                    "output": str(output),
                },
                sort_keys=True,
            )
        )
        return

    payload = json.loads(args.tuning.read_text(encoding="ascii"))
    if not isinstance(payload, dict):
        raise ValueError("G0.7 tuning must be JSON object")
    result = verify_g07_tuning(
        payload,
        expected_runtime_id=args.runtime_id,
    )
    print(
        json.dumps(
            {
                "status": "VN97_MAMBA2_G07_TUNING_VERIFIED",
                "tuning_id": result["tuning_id"],
                "runtime_id": result["runtime_id"],
                "production_activation_authorized":
                    result["production_activation_authorized"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
