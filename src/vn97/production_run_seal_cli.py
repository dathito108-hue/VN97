from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .production_run_manifest import (
    load_production_run_manifest,
)
from .production_run_sealer import (
    bootstrap_workspace,
    build_production_run_manifest,
    probe_production_environment,
    write_manifest_atomic,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap a real VN97 production workspace or seal it into one canonical VN97RUN1."
        )
    )
    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    bootstrap = sub.add_parser(
        "bootstrap",
        help=(
            "create an intentionally incomplete production workspace skeleton"
        ),
    )
    bootstrap.add_argument(
        "--workspace-root",
        required=True,
    )

    seal = sub.add_parser(
        "seal",
        help=(
            "scan real production inputs and atomically create production-run.vn97run1"
        ),
    )
    seal.add_argument(
        "--workspace-root",
        required=True,
    )
    seal.add_argument(
        "--repository-root",
        default=".",
    )
    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    args = _parser().parse_args(argv)

    if args.command == "bootstrap":
        root = Path(
            args.workspace_root
        )
        bootstrap_workspace(root)
        print(
            f"VN97 production workspace bootstrap: {root.resolve(strict=True)}"
        )
        return 0

    workspace = Path(
        args.workspace_root
    )
    repository = Path(
        args.repository_root
    )
    environment = (
        probe_production_environment()
    )
    manifest = (
        build_production_run_manifest(
            workspace_root=workspace,
            repository_root=repository,
            environment=environment,
        )
    )
    output = (
        workspace /
        "production-run.vn97run1"
    )
    write_manifest_atomic(
        output,
        manifest,
    )
    loaded = (
        load_production_run_manifest(
            output
        )
    )
    if loaded != manifest:
        raise RuntimeError(
            "sealed VN97RUN1 post-write parse does not equal verified manifest"
        )
    print(
        "VN97RUN1 "
        f"sha256={manifest.manifest_sha256} "
        f"commit={manifest.repository_commit} "
        f"path={output.resolve(strict=True)}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"vn97-production-seal: {exc}",
            file=sys.stderr,
        )
        raise
