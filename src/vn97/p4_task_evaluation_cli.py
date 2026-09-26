from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

from .p4_task_evaluation import (
    VN97P4EvaluationError,
    evaluate_p4_task_suite,
    verify_p3_final_artifact,
)


def _create_only(
    path: Path,
    data: bytes,
) -> None:
    if path.exists() or path.is_symlink():
        raise VN97P4EvaluationError(
            "P4 output must not already exist"
        )
    parent = path.parent
    parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    if parent.is_symlink():
        raise VN97P4EvaluationError(
            "P4 output parent must not be a symlink"
        )
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=parent,
    )
    temp = Path(temp_name)
    try:
        with os.fdopen(
            fd,
            "wb",
            closefd=True,
        ) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(
                handle.fileno()
            )
        try:
            os.link(
                temp,
                path,
            )
        except FileExistsError as exc:
            raise VN97P4EvaluationError(
                "P4 output raced with another writer"
            ) from exc
        dir_fd = os.open(
            parent,
            os.O_RDONLY
            | getattr(
                os,
                "O_DIRECTORY",
                0,
            ),
        )
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        temp.unlink(
            missing_ok=True
        )

    if path.read_bytes() != data:
        raise VN97P4EvaluationError(
            "P4 output post-write verification failed"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a canonical P3 final artifact and run "
            "held-out P4 task-level intelligence measurement."
        )
    )
    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    verify = sub.add_parser(
        "verify-p3",
        help=(
            "Verify the exact P3 final bundle before P4."
        ),
    )
    verify.add_argument(
        "--p3-dir",
        required=True,
    )

    evaluate = sub.add_parser(
        "evaluate",
        help=(
            "Run the held-out P4 task suite over the verified P3 winner."
        ),
    )
    evaluate.add_argument(
        "--p3-dir",
        required=True,
    )
    evaluate.add_argument(
        "--suite",
        required=True,
    )
    evaluate.add_argument(
        "--device",
        default="auto",
    )
    evaluate.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=4096,
    )
    evaluate.add_argument(
        "--output",
        required=True,
    )
    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    args = _parser().parse_args(
        argv
    )

    if args.command == "verify-p3":
        artifact = (
            verify_p3_final_artifact(
                Path(args.p3_dir)
            )
        )
        print(
            "VN97P4A P3 VERIFIED "
            f"candidate={artifact.selected_candidate_id} "
            f"checkpoint={artifact.checkpoint_sha256} "
            f"model_image={artifact.model_image_sha256} "
            f"model_image_bytes={artifact.model_image_bytes} "
            f"p3_run={artifact.p3_run_sha256}"
        )
        return 0

    report = evaluate_p4_task_suite(
        p3_dir=Path(
            args.p3_dir
        ),
        suite_path=Path(
            args.suite
        ),
        device=args.device,
        max_prompt_tokens=
            args.max_prompt_tokens,
    )
    data = report.to_bytes()
    _create_only(
        Path(args.output),
        data,
    )
    body = report.canonical_object()
    print(
        "VN97P4EVAL1 "
        f"id={body['evaluation_id']} "
        f"tasks={body['task_count']} "
        f"passed={body['passed_tasks']} "
        f"pass_rate={body['pass_rate']:.6f} "
        f"status={body['status']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except Exception as exc:
        print(
            "vn97-p4-task-eval: "
            f"{exc}",
            file=sys.stderr,
        )
        raise
