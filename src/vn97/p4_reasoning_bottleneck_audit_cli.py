from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

import torch

from .cognition_adapter import (
    TorchVN97InferenceEngine,
    VN97InferenceLimits,
)
from .deployment_checkpoint import (
    load_deployment_checkpoint_file,
)
from .p4_generalization_cli import (
    _probe_records,
    _record_score,
)
from .p4_generalization_curriculum import (
    default_validation,
)
from .tokenizer import (
    VN97Tokenizer,
    VN97TokenizerPackage,
)
from .training import (
    VN97ChatMessage,
    encode_chat_completion_messages,
    encode_chat_completion_prompt,
)


P4E_H_REPORT_SCHEMA = "VN97P4EHREASON1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _verify_artifact(root: Path):
    resolved = root.resolve(strict=True)
    required = {
        "SHA256SUMS",
        "model.vn97ck1",
        "model.vn97mi1",
        "p4g-report.json",
        "tokenizer.vn97tk1",
    }
    names = {
        item.name
        for item in resolved.iterdir()
    }
    if names != required:
        raise RuntimeError(
            "P4E-G artifact file set mismatch"
        )

    lines = (
        resolved
        / "SHA256SUMS"
    ).read_text(
        encoding="ascii"
    ).splitlines()
    sums: dict[str, str] = {}
    for line in lines:
        if (
            len(line) < 67
            or line[64:66] != "  "
        ):
            raise RuntimeError(
                "malformed SHA256SUMS"
            )
        sums[line[66:]] = line[:64]

    if set(sums) != (
        required - {"SHA256SUMS"}
    ):
        raise RuntimeError(
            "P4E-G SHA256SUMS file set mismatch"
        )

    for name, expected in sums.items():
        actual = hashlib.sha256(
            (resolved / name).read_bytes()
        ).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"SHA256 mismatch: {name}"
            )

    report = json.loads(
        (
            resolved
            / "p4g-report.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    if (
        not isinstance(report, dict)
        or report.get("schema")
        != "VN97P4EG1"
    ):
        raise RuntimeError(
            "artifact report is not VN97P4EG1"
        )

    checkpoint = (
        load_deployment_checkpoint_file(
            resolved / "model.vn97ck1"
        )
    )
    package = (
        VN97TokenizerPackage.from_bytes(
            (
                resolved
                / "tokenizer.vn97tk1"
            ).read_bytes()
        )
    )
    return (
        resolved,
        checkpoint,
        package,
        report,
    )


def _operation(prompt: str) -> str:
    if "*" in prompt and "(" in prompt:
        return "mul_parenthesized"
    if "*" in prompt:
        return "multiply"
    if "+" in prompt:
        return "add"
    if "-" in prompt:
        return "subtract"
    return "unknown"


def _copy_prompt(answer: str) -> str:
    return (
        "Copy exactly this decimal integer: "
        f"{answer}. Return only that integer."
    )


def _target_tokens(
    tokenizer: VN97Tokenizer,
    prompt: str,
    answer: str,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
]:
    messages = (
        VN97ChatMessage(
            role="user",
            content=prompt,
        ),
        VN97ChatMessage(
            role="assistant",
            content=answer,
        ),
    )
    example = (
        encode_chat_completion_messages(
            tokenizer,
            messages,
        )
    )
    first_target = next(
        index
        for index, enabled
        in enumerate(
            example.target_mask
        )
        if enabled
    )
    prefix = (
        example.token_ids[
            :first_target
        ]
    )
    target = tuple(
        token
        for token, enabled
        in zip(
            example.token_ids,
            example.target_mask,
        )
        if enabled
    )
    return prefix, target


def _teacher_forced_accuracy(
    *,
    model,
    tokens: tuple[int, ...],
    mask: tuple[bool, ...],
    device: str,
) -> tuple[int, int]:
    inputs = torch.tensor(
        [tokens[:-1]],
        dtype=torch.long,
        device=device,
    )
    logits, _ = model(inputs)
    predictions = (
        logits[0]
        .argmax(dim=-1)
        .tolist()
    )
    correct = 0
    total = 0
    for pred, target, enabled in zip(
        predictions,
        tokens[1:],
        mask[1:],
    ):
        if not enabled:
            continue
        total += 1
        correct += int(
            pred == target
        )
    return correct, total


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether P4E-G reasoning failures come from arithmetic "
            "computation, numeric copying, or answer-token prediction."
        )
    )
    parser.add_argument(
        "--artifact-dir",
        required=True,
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=8,
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
        args.examples < 0
        or not torch.cuda.is_available()
        or not str(
            args.device
        ).startswith("cuda")
    ):
        raise RuntimeError(
            "P4E-H requires CUDA"
        )

    (
        _root,
        checkpoint,
        package,
        report,
    ) = _verify_artifact(
        Path(args.artifact_dir)
    )

    tokenizer = VN97Tokenizer(
        package
    )
    model = checkpoint.model.to(
        args.device
    )
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

    records = tuple(
        record
        for record in _probe_records(
            default_validation(),
            per_category=30,
        )
        if record.category
        == "reasoning_planning"
    )

    summary = Counter()
    by_operation: dict[
        str,
        Counter[str],
    ] = defaultdict(Counter)
    rows: list[
        dict[str, object]
    ] = []
    examples: list[
        dict[str, str]
    ] = []

    with torch.inference_mode():
        for record in records:
            operation = _operation(
                record.prompt
            )
            messages = (
                VN97ChatMessage(
                    role="user",
                    content=record.prompt,
                ),
            )
            output = (
                engine.generate_chat_completion(
                    messages,
                    max_new_tokens=32,
                )
            )
            solve_ok = _record_score(
                record.answer,
                output,
            )

            copy_output = (
                engine.generate_chat_completion(
                    (
                        VN97ChatMessage(
                            role="user",
                            content=_copy_prompt(
                                record.answer
                            ),
                        ),
                    ),
                    max_new_tokens=16,
                )
            )
            copy_ok = _record_score(
                record.answer,
                copy_output,
            )

            (
                training_prefix,
                target_tokens,
            ) = _target_tokens(
                tokenizer,
                record.prompt,
                record.answer,
            )
            canonical_prefix = (
                encode_chat_completion_prompt(
                    tokenizer,
                    messages,
                )
            )
            prefix_match = (
                training_prefix
                == canonical_prefix
            )

            prefix_tensor = torch.tensor(
                [
                    list(
                        canonical_prefix
                    )
                ],
                dtype=torch.long,
                device=args.device,
            )
            logits, _ = model(
                prefix_tensor
            )
            scores = logits[
                0,
                -1,
            ]
            first_target = (
                target_tokens[0]
            )
            top1 = int(
                scores.argmax().item()
            )
            top5 = torch.topk(
                scores,
                k=min(
                    5,
                    scores.shape[-1],
                ),
            ).indices.tolist()
            first_top1 = (
                top1 == first_target
            )
            first_top5 = (
                first_target
                in top5
            )

            full_example = (
                encode_chat_completion_messages(
                    tokenizer,
                    (
                        *messages,
                        VN97ChatMessage(
                            role="assistant",
                            content=record.answer,
                        ),
                    ),
                )
            )
            (
                teacher_correct,
                teacher_total,
            ) = _teacher_forced_accuracy(
                model=model,
                tokens=
                    full_example.token_ids,
                mask=
                    full_example.target_mask,
                device=args.device,
            )

            summary["tasks"] += 1
            summary["solve_pass"] += int(
                solve_ok
            )
            summary["copy_pass"] += int(
                copy_ok
            )
            summary[
                "copy_pass_solve_fail"
            ] += int(
                copy_ok
                and not solve_ok
            )
            summary[
                "prefix_match"
            ] += int(
                prefix_match
            )
            summary[
                "first_top1"
            ] += int(
                first_top1
            )
            summary[
                "first_top5"
            ] += int(
                first_top5
            )
            summary[
                "teacher_correct"
            ] += teacher_correct
            summary[
                "teacher_total"
            ] += teacher_total

            op = by_operation[
                operation
            ]
            op["tasks"] += 1
            op["solve_pass"] += int(
                solve_ok
            )
            op["copy_pass"] += int(
                copy_ok
            )
            op["first_top1"] += int(
                first_top1
            )
            op[
                "teacher_correct"
            ] += teacher_correct
            op[
                "teacher_total"
            ] += teacher_total

            row = {
                "answer":
                    record.answer,
                "answer_token_count":
                    len(
                        target_tokens
                    ),
                "copy_output":
                    copy_output,
                "copy_pass":
                    copy_ok,
                "first_target_top1":
                    first_top1,
                "first_target_top5":
                    first_top5,
                "operation":
                    operation,
                "output":
                    output,
                "prefix_match":
                    prefix_match,
                "prompt":
                    record.prompt,
                "solve_pass":
                    solve_ok,
                "teacher_correct":
                    teacher_correct,
                "teacher_total":
                    teacher_total,
            }
            rows.append(row)

            if (
                copy_ok
                and not solve_ok
                and len(examples)
                < args.examples
            ):
                examples.append(
                    {
                        "answer":
                            record.answer,
                        "copy_output":
                            copy_output,
                        "operation":
                            operation,
                        "output":
                            output,
                        "prompt":
                            record.prompt,
                    }
                )

    print(
        "VN97 P4E-H SUMMARY "
        f"solve={summary['solve_pass']}/{summary['tasks']} "
        f"copy={summary['copy_pass']}/{summary['tasks']} "
        "copy_pass_solve_fail="
        f"{summary['copy_pass_solve_fail']}/{summary['tasks']} "
        f"prefix={summary['prefix_match']}/{summary['tasks']} "
        f"first_top1={summary['first_top1']}/{summary['tasks']} "
        f"first_top5={summary['first_top5']}/{summary['tasks']} "
        "teacher="
        f"{summary['teacher_correct']}/{summary['teacher_total']}",
        flush=True,
    )

    for operation in sorted(
        by_operation
    ):
        op = by_operation[
            operation
        ]
        print(
            "VN97 P4E-H OP "
            f"{operation} "
            f"solve={op['solve_pass']}/{op['tasks']} "
            f"copy={op['copy_pass']}/{op['tasks']} "
            f"first_top1={op['first_top1']}/{op['tasks']} "
            "teacher="
            f"{op['teacher_correct']}/{op['teacher_total']}",
            flush=True,
        )

    for item in examples:
        print(
            "\n"
            + "=" * 78,
            flush=True,
        )
        print(
            "P4E-H COPY-PASS/SOLVE-FAIL",
            flush=True,
        )
        print(
            "PROMPT_JSON="
            + json.dumps(
                item["prompt"],
                ensure_ascii=False,
            ),
            flush=True,
        )
        print(
            "ANSWER_JSON="
            + json.dumps(
                item["answer"],
                ensure_ascii=False,
            ),
            flush=True,
        )
        print(
            "SOLVE_OUTPUT_JSON="
            + json.dumps(
                item["output"],
                ensure_ascii=False,
            ),
            flush=True,
        )
        print(
            "COPY_OUTPUT_JSON="
            + json.dumps(
                item["copy_output"],
                ensure_ascii=False,
            ),
            flush=True,
        )

    result = {
        "by_operation": {
            key: dict(value)
            for key, value
            in sorted(
                by_operation.items()
            )
        },
        "checkpoint_sha256":
            checkpoint.checkpoint_sha256,
        "parent_status":
            report.get("status"),
        "rows":
            rows,
        "schema":
            P4E_H_REPORT_SCHEMA,
        "summary":
            dict(summary),
    }

    if args.output_json is not None:
        path = Path(
            args.output_json
        )
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        path.write_bytes(
            _canonical_json(
                result
            )
            + b"\n"
        )
        print(
            f"VN97 P4E-H REPORT {path}",
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
            "vn97-p4e-reasoning-audit: "
            f"{exc}",
            file=sys.stderr,
        )
        raise
