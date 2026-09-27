from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .production_registry import (
    admit_r2d11_pack,
    attach_r2d11_campaign,
    init_r2d11_registry,
    verify_r2d11_registry,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Manage the VN97-R2D11 append-only production source "
            "registry and incremental campaign ledger."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.add_argument("--definition", required=True)
    init.add_argument("--tokenizer", required=True)
    init.add_argument("--registry-dir", required=True)

    admit = sub.add_parser("admit-pack")
    admit.add_argument("--registry-dir", required=True)
    admit.add_argument("--pack-dir", required=True)

    attach = sub.add_parser("attach-campaign")
    attach.add_argument("--registry-dir", required=True)
    attach.add_argument("--pack-dir", required=True)
    attach.add_argument("--campaign-dir", required=True)
    attach.add_argument("--d6-package", required=True)

    verify = sub.add_parser("verify")
    verify.add_argument("--registry-dir", required=True)

    status = sub.add_parser("status")
    status.add_argument("--registry-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "init":
        config = init_r2d11_registry(
            definition_path=Path(args.definition),
            tokenizer_path=Path(args.tokenizer),
            registry_dir=Path(args.registry_dir),
        )
        print(
            "VN97R2D11 "
            f"command=init registry={config['registry_id']} "
            f"parameters={config['parameter_count']}",
            flush=True,
        )
        return 0

    if args.command == "admit-pack":
        snapshot = admit_r2d11_pack(
            registry_dir=Path(args.registry_dir),
            pack_dir=Path(args.pack_dir),
        )
        print(
            "VN97R2D11 "
            f"command=admit-pack generation={snapshot['generation']} "
            f"pack={snapshot['event']['pack_id']}",
            flush=True,
        )
        return 0

    if args.command == "attach-campaign":
        snapshot = attach_r2d11_campaign(
            registry_dir=Path(args.registry_dir),
            pack_dir=Path(args.pack_dir),
            campaign_dir=Path(args.campaign_dir),
            d6_package_dir=Path(args.d6_package),
        )
        state = snapshot["state"]
        print(
            "VN97R2D11 "
            f"command=attach-campaign generation={snapshot['generation']} "
            f"pack={snapshot['event']['pack_id']} "
            f"tokens_per_parameter={state['global_tokens_per_parameter']:.8f} "
            f"production_floor_ready="
            f"{str(state['production_floor_ready']).lower()}",
            flush=True,
        )
        return 0

    verified = verify_r2d11_registry(
        Path(args.registry_dir)
    )
    state = verified["state"]
    latest = verified["latest_snapshot"]

    if args.command == "verify":
        print(
            "VN97R2D11 "
            f"command=verify generation={latest['generation']} "
            f"admitted={len(state['admitted_batches'])} "
            f"attached={len(state['attached_batches'])} "
            "status=PASS",
            flush=True,
        )
        return 0

    print(
        json.dumps(
            {
                "generation": latest["generation"],
                "registry_id": verified["config"]["registry_id"],
                "state": state,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-r2-source-registry: {exc}", file=sys.stderr)
        raise
