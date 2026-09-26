from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import torch

from .cognition_adapter import (
    TorchVN97InferenceEngine,
    VN97InferenceLimits,
)
from .p4_artifact import (
    verify_p4d_artifact,
)
from .p4_generalization_curriculum import (
    P4D_CATEGORIES,
    default_validation,
)
from .p4_task_evaluation import (
    load_p4_task_suite,
    render_p4_chat_prompt,
    score_p4_output,
)
from .tokenizer import VN97Tokenizer
from .training_cli import _atomic_write


P4E_A_REPORT_SCHEMA = "VN97P4EADIAG1"


class VN97P4EADiagnosticError(
    RuntimeError
):
    pass


def _canonical_json(
    value: object,
) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _select_diagnostic_records(
    *,
    per_category: int,
):
    if per_category <= 0:
        raise VN97P4EADiagnosticError(
            "samples-per-category must be positive"
        )
    records = default_validation()
    selected = []
    for category in P4D_CATEGORIES:
        subset = [
            item
            for item in records
            if item.category == category
        ]
        subset.sort(
            key=lambda item: (
                hashlib.sha256(
                    item.prompt.encode(
                        "utf-8"
                    )
                ).digest()
            )
        )
        if len(subset) < per_category:
            raise VN97P4EADiagnosticError(
                "validation category is too small"
            )
        selected.extend(
            subset[:per_category]
        )
    return tuple(selected)


def _repeated_word_ngram(
    text: str,
    *,
    n: int = 4,
) -> bool:
    words = text.split()
    if len(words) < n * 2:
        return False
    seen: set[tuple[str, ...]] = set()
    for index in range(
        len(words) - n + 1
    ):
        gram = tuple(
            words[index:index + n]
        )
        if gram in seen:
            return True
        seen.add(gram)
    return False


def _classify_failure(
    expected: str,
    generated: str,
) -> str:
    expected_text = expected.strip()
    output_text = generated.strip()

    if output_text == expected_text:
        return "pass"
    if not output_text:
        return "empty_output"
    if (
        "<|assistant|>" in output_text
        or "<|user|>" in output_text
        or "<|system|>" in output_text
    ):
        return "role_marker_leakage"

    if expected_text.startswith("{"):
        try:
            expected_json = json.loads(
                expected_text
            )
        except json.JSONDecodeError:
            expected_json = None
        try:
            output_json = json.loads(
                output_text
            )
        except json.JSONDecodeError:
            if expected_text in output_text:
                return "structured_extra_text"
            if _repeated_word_ngram(
                output_text
            ):
                return "repetition"
            return "structured_json_parse_error"
        if (
            expected_json is not None
            and output_json
            == expected_json
        ):
            return "json_format_only_mismatch"
        return "structured_json_value_mismatch"

    if expected_text in output_text:
        return "extra_text"
    if _repeated_word_ngram(
        output_text
    ):
        return "repetition"
    return "text_mismatch"


def _score_strict(
    expected: str,
    generated: str,
) -> bool:
    expected_text = expected.strip()
    output_text = generated.strip()
    if expected_text.startswith("{"):
        try:
            return (
                json.loads(output_text)
                == json.loads(
                    expected_text
                )
            )
        except (
            json.JSONDecodeError,
            ValueError,
        ):
            return False
    return output_text == expected_text


def _render_block(
    *,
    source: str,
    category: str,
    identifier: str,
    prompt: str,
    expected: str,
    generated: str,
    passed: bool,
    failure: str,
) -> None:
    print(
        "\n"
        + "=" * 78,
        flush=True,
    )
    print(
        f"P4E-A source={source} "
        f"category={category} "
        f"id={identifier}",
        flush=True,
    )
    print(
        f"PROMPT_JSON={json.dumps(prompt, ensure_ascii=False)}",
        flush=True,
    )
    print(
        f"EXPECTED_JSON={json.dumps(expected, ensure_ascii=False)}",
        flush=True,
    )
    print(
        f"GENERATED_JSON={json.dumps(generated, ensure_ascii=False)}",
        flush=True,
    )
    print(
        f"PASS={str(passed).lower()} "
        f"FAILURE={failure}",
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect real generated answers from a measured P4D artifact "
            "without performing any training."
        )
    )
    parser.add_argument(
        "--p4d-dir",
        required=True,
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
    )
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
    parser.add_argument(
        "--dev-suite",
        default=None,
    )
    parser.add_argument(
        "--output-json",
        default=None,
    )
    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    args = _parser().parse_args(
        argv
    )
    if (
        args.samples_per_category <= 0
        or args.max_new_tokens <= 0
    ):
        raise VN97P4EADiagnosticError(
            "diagnostic arguments are invalid"
        )
    if (
        str(args.device).startswith(
            "cuda"
        )
        and not torch.cuda.is_available()
    ):
        raise VN97P4EADiagnosticError(
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
    failures: Counter[str] = Counter()

    selected = _select_diagnostic_records(
        per_category=
            args.samples_per_category,
    )
    for index, record in enumerate(
        selected,
        start=1,
    ):
        error_type = ""
        generated = ""
        try:
            generated = engine.generate_text(
                render_p4_chat_prompt(
                    record.prompt
                ),
                max_new_tokens=
                    args.max_new_tokens,
            )
            passed = _score_strict(
                record.answer,
                generated,
            )
            failure = (
                "pass"
                if passed
                else _classify_failure(
                    record.answer,
                    generated,
                )
            )
        except Exception as exc:
            passed = False
            failure = "generation_exception"
            error_type = type(exc).__name__

        failures[failure] += 1
        identifier = (
            f"validation-{index:03d}"
        )
        _render_block(
            source="validation",
            category=record.category,
            identifier=identifier,
            prompt=record.prompt,
            expected=record.answer,
            generated=generated,
            passed=passed,
            failure=failure,
        )
        rows.append(
            {
                "category":
                    record.category,
                "error_type":
                    error_type,
                "expected":
                    record.answer,
                "failure":
                    failure,
                "generated":
                    generated,
                "id":
                    identifier,
                "passed":
                    passed,
                "prompt":
                    record.prompt,
                "source":
                    "validation",
            }
        )

    if args.dev_suite is not None:
        suite = load_p4_task_suite(
            Path(args.dev_suite)
        )
        for task in suite.tasks:
            error_type = ""
            generated = ""
            try:
                generated = engine.generate_text(
                    render_p4_chat_prompt(
                        task.prompt
                    ),
                    max_new_tokens=
                        task.max_new_tokens,
                )
                passed = score_p4_output(
                    task,
                    generated,
                )
                expected = json.dumps(
                    task.expected,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ) if not isinstance(
                    task.expected,
                    str,
                ) else task.expected
                failure = (
                    "pass"
                    if passed
                    else _classify_failure(
                        expected,
                        generated,
                    )
                )
            except Exception as exc:
                passed = False
                expected = ""
                failure = (
                    "generation_exception"
                )
                error_type = (
                    type(exc).__name__
                )

            failures[failure] += 1
            _render_block(
                source="dev",
                category=task.category,
                identifier=task.task_id,
                prompt=task.prompt,
                expected=expected,
                generated=generated,
                passed=passed,
                failure=failure,
            )
            rows.append(
                {
                    "category":
                        task.category,
                    "error_type":
                        error_type,
                    "expected":
                        expected,
                    "failure":
                        failure,
                    "generated":
                        generated,
                    "id":
                        task.task_id,
                    "passed":
                        passed,
                    "prompt":
                        task.prompt,
                    "source":
                        "dev",
                }
            )

    passed_count = sum(
        1
        for row in rows
        if row["passed"]
    )
    report = {
        "checkpoint_sha256":
            artifact.checkpoint_sha256,
        "failure_counts":
            dict(sorted(
                failures.items()
            )),
        "p4d_report_sha256":
            artifact.report_sha256,
        "p4d_status":
            artifact.status,
        "passed":
            passed_count,
        "rows":
            rows,
        "schema":
            P4E_A_REPORT_SCHEMA,
        "tasks":
            len(rows),
    }

    print(
        "\n"
        + "=" * 78,
        flush=True,
    )
    print(
        "VN97 P4E-A SUMMARY "
        f"passed={passed_count}/{len(rows)} "
        f"failures={json.dumps(dict(sorted(failures.items())), sort_keys=True)}",
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
            f"VN97 P4E-A REPORT {output}",
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
            "vn97-p4e-diagnose: "
            f"{exc}",
            file=sys.stderr,
        )
        raise
