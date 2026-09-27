from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ..training_cli import _atomic_write
from .production_corpus_scale import verify_r2d6_corpus_index
from .production_curriculum import (
    compile_r2d8_plan,
    load_r2d8_definition,
    load_r2d8_plan,
)
from .production_virtual_corpus import (
    projection_for_stage,
    verify_r2d12_view,
)


def _add_corpus_source(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--corpus-package")
    group.add_argument("--virtual-view")
    parser.add_argument("--workspace-root")


def _virtual_index(
    *,
    view_dir: Path,
    workspace_root: Path | None,
) -> dict[str, object]:
    if workspace_root is None:
        raise ValueError(
            "R2-D8 virtual view requires --workspace-root"
        )
    verified = verify_r2d12_view(
        view_dir,
        workspace_root=workspace_root,
    )
    return verified["index"]


def _plan_preview(path: Path) -> dict[str, object]:
    if path.is_symlink():
        raise ValueError("R2-D8 plan must not be a symlink")
    try:
        value = json.loads(
            path.resolve(strict=True).read_text(
                encoding="utf-8",
                errors="strict",
            ),
            parse_constant=lambda raw: (
                _ for _ in ()
            ).throw(ValueError(raw)),
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise ValueError("R2-D8 plan must be strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("R2-D8 plan must be an object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile or verify a deterministic VN97-R2D8 task-family "
            "curriculum against either one R2-D6 package or an R2-D12 "
            "virtual multi-batch corpus view."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    _add_corpus_source(build)
    build.add_argument("--definition", required=True)
    build.add_argument("--output", required=True)

    verify = sub.add_parser("verify")
    _add_corpus_source(verify)
    verify.add_argument("--plan", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "build":
        definition = load_r2d8_definition(
            Path(args.definition)
        )
        if args.virtual_view:
            full_index = _virtual_index(
                view_dir=Path(args.virtual_view),
                workspace_root=(
                    None
                    if args.workspace_root is None
                    else Path(args.workspace_root)
                ),
            )
            index = projection_for_stage(
                full_index,
                stage=definition.stage,
                family_weights=definition.weights,
            )
        else:
            if args.workspace_root is not None:
                raise ValueError(
                    "--workspace-root is only valid with --virtual-view"
                )
            index = verify_r2d6_corpus_index(
                Path(args.corpus_package).resolve(strict=True)
            )

        plan = compile_r2d8_plan(index, definition)
        output = Path(args.output)
        if output.exists() or output.is_symlink():
            raise ValueError(
                "R2-D8 plan output must not already exist"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(
            output,
            json.dumps(
                plan,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ).encode("utf-8") + b"\n",
        )
        print(
            "VN97R2D8 "
            f"command=build plan={plan['plan_id']} "
            f"stage={plan['stage']} "
            f"epochs={plan['epochs']} "
            f"corpus_index={index['index_id']}",
            flush=True,
        )
        return 0

    plan_path = Path(args.plan)
    if args.virtual_view:
        preview = _plan_preview(plan_path)
        weights = preview.get("family_weights")
        if not isinstance(weights, dict) or not weights:
            raise ValueError(
                "R2-D8 plan family_weights are missing"
            )
        full_index = _virtual_index(
            view_dir=Path(args.virtual_view),
            workspace_root=(
                None
                if args.workspace_root is None
                else Path(args.workspace_root)
            ),
        )
        index = projection_for_stage(
            full_index,
            stage=str(preview.get("stage")),
            family_weights=weights,
        )
    else:
        if args.workspace_root is not None:
            raise ValueError(
                "--workspace-root is only valid with --virtual-view"
            )
        index = verify_r2d6_corpus_index(
            Path(args.corpus_package).resolve(strict=True)
        )

    plan = load_r2d8_plan(
        plan_path,
        index,
    )
    print(
        "VN97R2D8 "
        f"command=verify plan={plan['plan_id']} "
        f"stage={plan['stage']} "
        f"epochs={plan['epochs']} "
        f"corpus_index={index['index_id']} "
        "status=PASS",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-r2-curriculum: {exc}", file=sys.stderr)
        raise
