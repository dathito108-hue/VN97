from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .p5d_distillation import (
    P5D0_PROFILE_ID,
    TEACHER,
    profile_object,
    profile_sha256,
)
from .tokenizer import VN97TokenizerPackage


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the frozen P5D teacher/student distillation contract "
            "without downloading or loading the teacher model."
        )
    )
    parser.add_argument(
        "--vn97-tokenizer",
        required=True,
    )
    parser.add_argument(
        "--output",
        default=None,
    )
    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    args = _parser().parse_args(
        argv
    )

    package = VN97TokenizerPackage.from_bytes(
        Path(
            args.vn97_tokenizer
        ).read_bytes()
    )
    report = profile_object(
        vocab_size=package.vocab_size
    )
    digest = profile_sha256(
        vocab_size=package.vocab_size
    )

    target = report[
        "target"
    ]
    print(
        "VN97 P5D0 TEACHER "
        f"repo={TEACHER.repo} "
        f"architecture={TEACHER.architecture!r} "
        f"layers={TEACHER.layers} "
        f"d_model={TEACHER.d_model} "
        f"d_state={TEACHER.d_state}",
        flush=True,
    )
    print(
        "VN97 P5D0 TARGET "
        f"parameters={target['estimated_parameters']} "
        f"d_model={target['d_model']} "
        f"layers={target['layers']} "
        f"d_state={target['d_state']} "
        f"embedding_rank={target['embedding_rank']} "
        f"packed_lower_bound_bytes={target['approx_packed_ternary_bytes']} "
        f"state_fp16_bytes={target['recurrent_state_bytes_fp16_batch1']}",
        flush=True,
    )
    print(
        "VN97 P5D0 DISTILL "
        "initial_precision=float-shadow "
        "initial_ternary=false "
        f"profile={P5D0_PROFILE_ID} "
        f"profile_sha256={digest}",
        flush=True,
    )

    if args.output is not None:
        output = Path(
            args.output
        )
        if (
            output.exists()
            and output.is_symlink()
        ):
            raise ValueError(
                "output must not be a symlink"
            )
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        data = dict(report)
        data["profile_sha256"] = digest
        output.write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"VN97 P5D0 REPORT {output}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except Exception as exc:
        print(
            "vn97-p5d-plan: "
            f"{exc}",
            file=sys.stderr,
        )
        raise
