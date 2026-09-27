from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .assistant_bridge import (
    R2F2_BINDING_FILENAME,
    build_r2f2_binding,
    verify_r2f2_binding,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build/verify the VN97-R2 F2 tokenizer/runtime identity binding."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--bundle-dir", required=True)
    build.add_argument("--tokenizer-model-sha256", required=True)
    build.add_argument("--output")
    verify = sub.add_parser("verify")
    verify.add_argument("--binding", required=True)
    verify.add_argument("--runtime-id")
    verify.add_argument("--tokenizer-model-sha256")
    status = sub.add_parser("status")
    status.add_argument("--binding", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        bundle = Path(args.bundle_dir)
        output = (
            Path(args.output)
            if args.output is not None
            else bundle / R2F2_BINDING_FILENAME
        )
        payload = build_r2f2_binding(
            bundle_dir=bundle,
            tokenizer_model_sha256=args.tokenizer_model_sha256,
            output_path=output,
        )
        print(
            "VN97R2F2 "
            f"binding={payload['binding_id']} "
            f"runtime={payload['runtime_id']} status=PASS",
            flush=True,
        )
        return 0
    payload = verify_r2f2_binding(
        Path(args.binding),
        expected_runtime_id=getattr(args, "runtime_id", None),
        expected_tokenizer_model_sha256=getattr(
            args, "tokenizer_model_sha256", None
        ),
    )
    if args.command == "verify":
        print(
            f"VN97R2F2 binding={payload['binding_id']} status=PASS",
            flush=True,
        )
    else:
        print(
            json.dumps(
                payload,
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
        print(f"vn97-r2-assistant-bind: {exc}", file=sys.stderr)
        raise
