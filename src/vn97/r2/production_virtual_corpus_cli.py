from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .production_virtual_corpus import (
    build_r2d12_view,
    verify_r2d12_view,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build or verify a VN97-R2D12 registry freeze and virtual "
            "multi-batch corpus view without copying shard data."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--registry-dir", required=True)
    build.add_argument("--generation", type=int, required=True)
    build.add_argument("--workspace-root", required=True)
    build.add_argument("--mount-definition", required=True)
    build.add_argument("--output-dir", required=True)

    verify = sub.add_parser("verify")
    verify.add_argument("--view-dir", required=True)
    verify.add_argument("--workspace-root", required=True)

    status = sub.add_parser("status")
    status.add_argument("--view-dir", required=True)
    status.add_argument("--workspace-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "build":
        view = build_r2d12_view(
            registry_dir=Path(args.registry_dir),
            generation=args.generation,
            workspace_root=Path(args.workspace_root),
            mount_definition_path=Path(args.mount_definition),
            output_dir=Path(args.output_dir),
        )
        index = view["virtual_index"]
        print(
            "VN97R2D12 "
            f"command=build view={view['view_id']} "
            f"generation={view['registry_generation']} "
            f"batches={len(view['attached_batches'])} "
            f"tokens_per_parameter="
            f"{index['scale']['tokens_per_parameter']:.8f}",
            flush=True,
        )
        return 0

    verified = verify_r2d12_view(
        Path(args.view_dir),
        workspace_root=Path(args.workspace_root),
    )
    view = verified["view"]
    index = verified["index"]

    if args.command == "verify":
        print(
            "VN97R2D12 "
            f"command=verify view={view['view_id']} "
            f"generation={view['registry_generation']} "
            f"batches={len(view['attached_batches'])} "
            "status=PASS",
            flush=True,
        )
        return 0

    print(
        json.dumps(
            {
                "view_id": view["view_id"],
                "registry_generation": view["registry_generation"],
                "attached_batches": view["attached_batches"],
                "virtual_index_id": index["index_id"],
                "scale": index["scale"],
                "split_totals": index["split_totals"],
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
        print(f"vn97-r2-virtual-corpus: {exc}", file=sys.stderr)
        raise
