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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile or verify a deterministic VN97-R2D8 task-family "
            "curriculum against an immutable R2-D6 corpus package."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--corpus-package", required=True)
    build.add_argument("--definition", required=True)
    build.add_argument("--output", required=True)

    verify = sub.add_parser("verify")
    verify.add_argument("--corpus-package", required=True)
    verify.add_argument("--plan", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    package = Path(args.corpus_package).resolve(strict=True)
    index = verify_r2d6_corpus_index(package)

    if args.command == "build":
        definition = load_r2d8_definition(
            Path(args.definition)
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
            f"epochs={plan['epochs']}",
            flush=True,
        )
        return 0

    plan = load_r2d8_plan(
        Path(args.plan),
        index,
    )
    print(
        "VN97R2D8 "
        f"command=verify plan={plan['plan_id']} "
        f"stage={plan['stage']} "
        f"epochs={plan['epochs']} "
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
