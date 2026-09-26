from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from .cognition_adapter import (
    TorchVN97InferenceEngine,
    VN97InferenceLimits,
    recover_chat_response_text,
)
from .p4_artifact import verify_p4d_artifact
from .p4_generation_diagnostic_cli import (
    _classify_failure,
    _score_strict,
    _select_diagnostic_records,
)
from .p4_task_evaluation import (
    load_p4_task_suite,
    render_p4_chat_prompt,
    score_p4_output,
)
from .tokenizer import VN97Tokenizer
from .training_cli import _atomic_write


P4E_B_REPORT_SCHEMA = "VN97P4EBBOUNDARY1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _dev_expected_text(task) -> str:
    if task.scoring_kind == "exact_text":
        return str(task.expected)
    if task.scoring_kind == "json_exact":
        return json.dumps(
            task.expected,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return json.dumps(
        list(task.expected),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure legacy textual chat-boundary recovery on the exact "
            "P4D checkpoint without changing model weights."
        )
    )
    parser.add_argument("--p4d-dir", required=True)
    parser.add_argument("--dev-suite", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--samples-per-category",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=96,
    )
    parser.add_argument("--output-json", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (
        args.samples_per_category <= 0
        or args.max_new_tokens <= 0
    ):
        raise ValueError("P4E-B arguments are invalid")
    if (
        str(args.device).startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "requested CUDA device is unavailable"
        )

    artifact = verify_p4d_artifact(
        Path(args.p4d_dir)
    )
    tokenizer = VN97Tokenizer(
        artifact.tokenizer_package
    )
    model = artifact.checkpoint.model
    model.to(args.device)
    model.eval()

    engine = TorchVN97InferenceEngine(
        model,
        tokenizer,
        limits=VN97InferenceLimits(
            max_prompt_tokens=4096,
            repetition_penalty=1.12,
            no_repeat_ngram_size=4,
        ),
    )

    rows: list[dict[str, object]] = []

    for index, record in enumerate(
        _select_diagnostic_records(
            per_category=args.samples_per_category
        ),
        start=1,
    ):
        raw = engine.generate_text(
            render_p4_chat_prompt(record.prompt),
            max_new_tokens=args.max_new_tokens,
        )
        recovered = recover_chat_response_text(raw)
        raw_pass = _score_strict(
            record.answer,
            raw,
        )
        recovered_pass = _score_strict(
            record.answer,
            recovered,
        )
        raw_failure = (
            "pass"
            if raw_pass
            else _classify_failure(
                record.answer,
                raw,
            )
        )
        recovered_failure = (
            "pass"
            if recovered_pass
            else _classify_failure(
                record.answer,
                recovered,
            )
        )
        identifier = f"validation-{index:03d}"
        print(
            "\n"
            + "=" * 78
            + "\n"
            + (
                f"P4E-B source=validation "
                f"category={record.category} "
                f"id={identifier}"
            )
            + "\n"
            + (
                "RAW_JSON="
                + json.dumps(
                    raw,
                    ensure_ascii=False,
                )
            )
            + "\n"
            + (
                "RECOVERED_JSON="
                + json.dumps(
                    recovered,
                    ensure_ascii=False,
                )
            )
            + "\n"
            + (
                f"RAW_PASS={str(raw_pass).lower()} "
                f"RECOVERED_PASS={str(recovered_pass).lower()} "
                f"RAW_FAILURE={raw_failure} "
                f"RECOVERED_FAILURE={recovered_failure}"
            ),
            flush=True,
        )
        rows.append(
            {
                "category": record.category,
                "expected": record.answer,
                "id": identifier,
                "prompt": record.prompt,
                "raw": raw,
                "raw_failure": raw_failure,
                "raw_pass": raw_pass,
                "recovered": recovered,
                "recovered_failure":
                    recovered_failure,
                "recovered_pass":
                    recovered_pass,
                "source": "validation",
            }
        )

    if args.dev_suite is not None:
        suite = load_p4_task_suite(
            Path(args.dev_suite)
        )
        for task in suite.tasks:
            raw = engine.generate_text(
                render_p4_chat_prompt(
                    task.prompt
                ),
                max_new_tokens=
                    task.max_new_tokens,
            )
            recovered = (
                recover_chat_response_text(
                    raw
                )
            )
            raw_pass = score_p4_output(
                task,
                raw,
            )
            recovered_pass = score_p4_output(
                task,
                recovered,
            )
            expected = _dev_expected_text(
                task
            )
            raw_failure = (
                "pass"
                if raw_pass
                else _classify_failure(
                    expected,
                    raw,
                )
            )
            recovered_failure = (
                "pass"
                if recovered_pass
                else _classify_failure(
                    expected,
                    recovered,
                )
            )
            print(
                "\n"
                + "=" * 78
                + "\n"
                + (
                    f"P4E-B source=dev "
                    f"category={task.category} "
                    f"id={task.task_id}"
                )
                + "\n"
                + (
                    "RAW_JSON="
                    + json.dumps(
                        raw,
                        ensure_ascii=False,
                    )
                )
                + "\n"
                + (
                    "RECOVERED_JSON="
                    + json.dumps(
                        recovered,
                        ensure_ascii=False,
                    )
                )
                + "\n"
                + (
                    f"RAW_PASS={str(raw_pass).lower()} "
                    f"RECOVERED_PASS={str(recovered_pass).lower()} "
                    f"RAW_FAILURE={raw_failure} "
                    f"RECOVERED_FAILURE={recovered_failure}"
                ),
                flush=True,
            )
            rows.append(
                {
                    "category":
                        task.category,
                    "expected":
                        expected,
                    "id":
                        task.task_id,
                    "prompt":
                        task.prompt,
                    "raw":
                        raw,
                    "raw_failure":
                        raw_failure,
                    "raw_pass":
                        raw_pass,
                    "recovered":
                        recovered,
                    "recovered_failure":
                        recovered_failure,
                    "recovered_pass":
                        recovered_pass,
                    "source":
                        "dev",
                }
            )

    raw_passed = sum(
        1 for row in rows
        if row["raw_pass"]
    )
    recovered_passed = sum(
        1 for row in rows
        if row["recovered_pass"]
    )
    recovered_gains = sum(
        1 for row in rows
        if (
            not row["raw_pass"]
            and row["recovered_pass"]
        )
    )
    regressions = sum(
        1 for row in rows
        if (
            row["raw_pass"]
            and not row["recovered_pass"]
        )
    )
    report = {
        "checkpoint_sha256":
            artifact.checkpoint_sha256,
        "p4d_status":
            artifact.status,
        "raw_passed":
            raw_passed,
        "recovered_passed":
            recovered_passed,
        "recovered_gains":
            recovered_gains,
        "regressions":
            regressions,
        "rows":
            rows,
        "schema":
            P4E_B_REPORT_SCHEMA,
        "tasks":
            len(rows),
    }

    print(
        "\n"
        + "=" * 78,
        flush=True,
    )
    print(
        "VN97 P4E-B SUMMARY "
        f"raw={raw_passed}/{len(rows)} "
        f"recovered={recovered_passed}/{len(rows)} "
        f"gains={recovered_gains} "
        f"regressions={regressions}",
        flush=True,
    )

    if args.output_json is not None:
        output = Path(
            args.output_json
        )
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        _atomic_write(
            output,
            _canonical_json(report)
            + b"\n",
        )
        print(
            f"VN97 P4E-B REPORT {output}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            "vn97-p4e-boundary-eval: "
            f"{exc}",
            file=sys.stderr,
        )
        raise
