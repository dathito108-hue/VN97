from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .production_campaign import R2D5_MAX_SPLIT_BYTES
from .production_corpus_scale import (
    R2D6_DEFAULT_MIN_TOKENS_PER_PARAMETER,
    R2D6_DEFAULT_TARGET_TOKENS_PER_PARAMETER,
    build_r2d6_corpus_index,
    load_r2d6_definition,
    verify_r2d6_corpus_index,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build or verify a sharded VN97-R2D6 production corpus "
            "index from one or more sealed VN97CORPUS1 inputs."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--definition", required=True)
    build.add_argument("--tokenizer", required=True)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--sequence-length", type=int, default=128)
    build.add_argument(
        "--minimum-tokens-per-parameter",
        type=float,
        default=R2D6_DEFAULT_MIN_TOKENS_PER_PARAMETER,
    )
    build.add_argument(
        "--target-tokens-per-parameter",
        type=float,
        default=R2D6_DEFAULT_TARGET_TOKENS_PER_PARAMETER,
    )
    build.add_argument(
        "--max-split-bytes",
        type=int,
        default=R2D5_MAX_SPLIT_BYTES,
    )

    verify = sub.add_parser("verify")
    verify.add_argument("--package-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "build":
        corpus_inputs = load_r2d6_definition(
            Path(args.definition)
        )
        payload = build_r2d6_corpus_index(
            corpus_inputs=corpus_inputs,
            tokenizer_path=Path(args.tokenizer),
            output_dir=Path(args.output_dir),
            sequence_length=args.sequence_length,
            minimum_tokens_per_parameter=(
                args.minimum_tokens_per_parameter
            ),
            target_tokens_per_parameter=(
                args.target_tokens_per_parameter
            ),
            max_split_bytes=args.max_split_bytes,
        )
        scale = payload["scale"]
        print(
            "VN97R2D6 "
            f"command=build index={payload['index_id']} "
            f"target_tokens={scale['training_target_tokens']} "
            f"tokens_per_parameter={scale['tokens_per_parameter']:.8f} "
            f"scale_floor_passed={str(scale['scale_floor_passed']).lower()} "
            f"scale_target_passed={str(scale['scale_target_passed']).lower()}",
            flush=True,
        )
        return 0

    payload = verify_r2d6_corpus_index(
        Path(args.package_dir)
    )
    scale = payload["scale"]
    print(
        "VN97R2D6 "
        f"command=verify index={payload['index_id']} "
        f"shards={len(payload['shards'])} "
        f"tokens_per_parameter={scale['tokens_per_parameter']:.8f} "
        "status=PASS",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-r2-corpus-scale: {exc}", file=sys.stderr)
        raise
