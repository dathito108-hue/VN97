from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import torch
import torch.nn.functional as F

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


P4E_L_REPORT_SCHEMA = "VN97P4EL1"

_SIMPLE_RE = re.compile(
    r"(?P<a>\d+)\s*(?P<op>[+\-*])\s*(?P<b>\d+)"
)
_PAREN_RE = re.compile(
    r"(?P<a>\d+)\s*\*\s*\(\s*(?P<b>\d+)\s*\+\s*(?P<c>\d+)\s*\)"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _verify_artifact(
    root: Path,
):
    resolved = root.resolve(strict=True)
    if (
        not resolved.is_dir()
        or resolved.is_symlink()
    ):
        raise RuntimeError(
            "P4E-K root must be a real directory"
        )

    required = {
        "SHA256SUMS",
        "model.vn97ck1",
        "model.vn97mi1",
        "p4k-report.json",
        "tokenizer.vn97tk1",
    }
    names = {
        item.name
        for item in resolved.iterdir()
    }
    if names != required:
        raise RuntimeError(
            "P4E-K artifact file set mismatch"
        )

    sums_text = (
        resolved
        / "SHA256SUMS"
    ).read_text(
        encoding="ascii"
    )
    if not sums_text.endswith("\n"):
        raise RuntimeError(
            "P4E-K SHA256SUMS must end with newline"
        )

    sums: dict[str, str] = {}
    for line in sums_text.splitlines():
        if (
            len(line) < 67
            or line[64:66] != "  "
        ):
            raise RuntimeError(
                "malformed P4E-K SHA256SUMS"
            )
        digest = line[:64]
        name = line[66:]
        sums[name] = digest

    if set(sums) != (
        required - {"SHA256SUMS"}
    ):
        raise RuntimeError(
            "P4E-K SHA256SUMS file set mismatch"
        )

    for name, expected in sums.items():
        actual = hashlib.sha256(
            (
                resolved
                / name
            ).read_bytes()
        ).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"P4E-K SHA256 mismatch: {name}"
            )

    report = json.loads(
        (
            resolved
            / "p4k-report.json"
        ).read_text(
            encoding="utf-8"
        )
    )
    if (
        not isinstance(report, dict)
        or report.get("schema")
        != "VN97P4EK1"
    ):
        raise RuntimeError(
            "artifact report is not VN97P4EK1"
        )

    checkpoint = (
        load_deployment_checkpoint_file(
            resolved
            / "model.vn97ck1"
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

    if (
        checkpoint.config.vocab_size
        != package.vocab_size
    ):
        raise RuntimeError(
            "P4E-K checkpoint/tokenizer vocabulary mismatch"
        )

    return (
        resolved,
        checkpoint,
        package,
        report,
    )


def _parse_expression(
    prompt: str,
) -> tuple[
    str,
    tuple[int, ...],
]:
    paren = _PAREN_RE.search(
        prompt
    )
    if paren is not None:
        return (
            "mul_parenthesized",
            (
                int(paren.group("a")),
                int(paren.group("b")),
                int(paren.group("c")),
            ),
        )

    simple = _SIMPLE_RE.search(
        prompt
    )
    if simple is None:
        raise RuntimeError(
            f"could not parse arithmetic expression from prompt: {prompt!r}"
        )

    op = simple.group("op")
    kind = {
        "+": "add",
        "-": "subtract",
        "*": "multiply",
    }[op]
    return (
        kind,
        (
            int(simple.group("a")),
            int(simple.group("b")),
        ),
    )


def _correct_answer(
    kind: str,
    operands: tuple[int, ...],
) -> int:
    if kind == "add":
        return (
            operands[0]
            + operands[1]
        )
    if kind == "subtract":
        return (
            operands[0]
            - operands[1]
        )
    if kind == "multiply":
        return (
            operands[0]
            * operands[1]
        )
    if kind == "mul_parenthesized":
        return (
            operands[0]
            * (
                operands[1]
                + operands[2]
            )
        )
    raise RuntimeError(
        f"unsupported arithmetic kind: {kind}"
    )


def _candidate_answers(
    kind: str,
    operands: tuple[int, ...],
) -> tuple[str, ...]:
    correct = _correct_answer(
        kind,
        operands,
    )
    values: set[int] = {
        correct,
        correct - 10,
        correct - 2,
        correct - 1,
        correct + 1,
        correct + 2,
        correct + 10,
    }

    if len(operands) == 2:
        a, b = operands
        values.update(
            {
                a,
                b,
                a + b,
                a - b,
                b - a,
                a * b,
                abs(
                    a - b
                ),
            }
        )
    else:
        a, b, c = operands
        values.update(
            {
                a,
                b,
                c,
                a * b + c,
                a + b * c,
                (a + b) * c,
                a * b,
                b + c,
            }
        )

    values = {
        value
        for value in values
        if value >= 0
    }
    ordered = sorted(
        values
    )
    return tuple(
        str(value)
        for value in ordered
    )


def _normalized_prompt(
    kind: str,
    operands: tuple[int, ...],
) -> str:
    if kind == "add":
        expr = (
            f"{operands[0]} + {operands[1]}"
        )
    elif kind == "subtract":
        expr = (
            f"{operands[0]} - {operands[1]}"
        )
    elif kind == "multiply":
        expr = (
            f"{operands[0]} * {operands[1]}"
        )
    elif kind == "mul_parenthesized":
        expr = (
            f"{operands[0]} * "
            f"({operands[1]} + {operands[2]})"
        )
    else:
        raise RuntimeError(
            f"unsupported arithmetic kind: {kind}"
        )

    return (
        "Compute exactly this arithmetic expression: "
        f"{expr}. Return only the decimal integer result."
    )


def _sequence_logprob(
    *,
    model,
    tokenizer: VN97Tokenizer,
    prompt: str,
    answer: str,
    device: str,
) -> dict[str, object]:
    example = (
        encode_chat_completion_messages(
            tokenizer,
            (
                VN97ChatMessage(
                    role="user",
                    content=prompt,
                ),
                VN97ChatMessage(
                    role="assistant",
                    content=answer,
                ),
            ),
        )
    )

    inputs = torch.tensor(
        [
            list(
                example.token_ids[
                    :-1
                ]
            )
        ],
        dtype=torch.long,
        device=device,
    )

    logits, _ = model(
        inputs
    )
    log_probs = F.log_softmax(
        logits[0],
        dim=-1,
    )

    total = 0.0
    count = 0
    first_rank = None
    first_top5 = False

    for index, (
        target,
        enabled,
    ) in enumerate(
        zip(
            example.token_ids[
                1:
            ],
            example.target_mask[
                1:
            ],
        )
    ):
        if not enabled:
            continue

        row = log_probs[
            index
        ]
        target_score = float(
            row[target].item()
        )

        if first_rank is None:
            raw_logits = logits[
                0,
                index,
            ]
            target_logit = (
                raw_logits[
                    target
                ]
            )
            first_rank = int(
                (
                    raw_logits
                    > target_logit
                ).sum().item()
            ) + 1
            top5 = torch.topk(
                raw_logits,
                k=min(
                    5,
                    raw_logits.shape[-1],
                ),
            ).indices.tolist()
            first_top5 = (
                target
                in top5
            )

        total += target_score
        count += 1

    if count <= 0:
        raise RuntimeError(
            "candidate answer produced no supervised tokens"
        )

    return {
        "mean_logprob":
            total / count,
        "sum_logprob":
            total,
        "target_tokens":
            count,
        "first_token_rank":
            first_rank,
        "first_token_top5":
            first_top5,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether VN97 arithmetic failure is due to decoding/exposure "
            "or absence of a correct arithmetic preference in model likelihoods."
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
        ).startswith(
            "cuda"
        )
    ):
        raise RuntimeError(
            "P4E-L requires CUDA"
        )

    (
        _root,
        checkpoint,
        package,
        parent_report,
    ) = _verify_artifact(
        Path(
            args.artifact_dir
        )
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
    ] = {}
    rows: list[
        dict[str, object]
    ] = []
    decode_mismatch_examples: list[
        dict[str, object]
    ] = []
    representation_failure_examples: list[
        dict[str, object]
    ] = []

    with torch.inference_mode():
        for record in records:
            kind, operands = (
                _parse_expression(
                    record.prompt
                )
            )
            correct_int = (
                _correct_answer(
                    kind,
                    operands,
                )
            )
            if str(
                correct_int
            ) != record.answer:
                raise RuntimeError(
                    "parsed arithmetic result does not match frozen expected answer"
                )

            normalized_prompt = (
                _normalized_prompt(
                    kind,
                    operands,
                )
            )

            direct = (
                engine.generate_chat_completion(
                    (
                        VN97ChatMessage(
                            role="user",
                            content=
                                record.prompt,
                        ),
                    ),
                    max_new_tokens=32,
                )
            )
            normalized = (
                engine.generate_chat_completion(
                    (
                        VN97ChatMessage(
                            role="user",
                            content=
                                normalized_prompt,
                        ),
                    ),
                    max_new_tokens=32,
                )
            )

            direct_ok = (
                _record_score(
                    record.answer,
                    direct,
                )
            )
            normalized_ok = (
                _record_score(
                    record.answer,
                    normalized,
                )
            )

            candidates = (
                _candidate_answers(
                    kind,
                    operands,
                )
            )
            scored: list[
                tuple[
                    str,
                    dict[str, object],
                ]
            ] = []
            for candidate in candidates:
                score = (
                    _sequence_logprob(
                        model=model,
                        tokenizer=
                            tokenizer,
                        prompt=
                            record.prompt,
                        answer=
                            candidate,
                        device=
                            args.device,
                    )
                )
                scored.append(
                    (
                        candidate,
                        score,
                    )
                )

            scored.sort(
                key=lambda item:
                    float(
                        item[1][
                            "mean_logprob"
                        ]
                    ),
                reverse=True,
            )
            correct_rank = next(
                index
                for index, (
                    candidate,
                    _score,
                ) in enumerate(
                    scored,
                    start=1,
                )
                if candidate
                == record.answer
            )
            correct_score = next(
                score
                for candidate, score
                in scored
                if candidate
                == record.answer
            )
            best_candidate = (
                scored[0][0]
            )
            best_score = (
                scored[0][1]
            )

            prefix = (
                encode_chat_completion_prompt(
                    tokenizer,
                    (
                        VN97ChatMessage(
                            role="user",
                            content=
                                record.prompt,
                        ),
                    ),
                )
            )
            training = (
                encode_chat_completion_messages(
                    tokenizer,
                    (
                        VN97ChatMessage(
                            role="user",
                            content=
                                record.prompt,
                        ),
                        VN97ChatMessage(
                            role="assistant",
                            content=
                                record.answer,
                        ),
                    ),
                )
            )
            first_target = next(
                index
                for index, enabled
                in enumerate(
                    training.target_mask
                )
                if enabled
            )
            prefix_match = (
                prefix
                == training.token_ids[
                    :first_target
                ]
            )

            summary[
                "tasks"
            ] += 1
            summary[
                "direct_pass"
            ] += int(
                direct_ok
            )
            summary[
                "normalized_pass"
            ] += int(
                normalized_ok
            )
            summary[
                "rank1"
            ] += int(
                correct_rank == 1
            )
            summary[
                "rank3"
            ] += int(
                correct_rank <= 3
            )
            summary[
                "rank5"
            ] += int(
                correct_rank <= 5
            )
            summary[
                "first_top1"
            ] += int(
                correct_score[
                    "first_token_rank"
                ]
                == 1
            )
            summary[
                "first_top5"
            ] += int(
                bool(
                    correct_score[
                        "first_token_top5"
                    ]
                )
            )
            summary[
                "prefix_match"
            ] += int(
                prefix_match
            )
            summary[
                "rank1_but_decode_fail"
            ] += int(
                correct_rank == 1
                and not direct_ok
            )

            op = by_operation.setdefault(
                kind,
                Counter(),
            )
            op["tasks"] += 1
            op[
                "direct_pass"
            ] += int(
                direct_ok
            )
            op[
                "normalized_pass"
            ] += int(
                normalized_ok
            )
            op[
                "rank1"
            ] += int(
                correct_rank == 1
            )
            op[
                "rank3"
            ] += int(
                correct_rank <= 3
            )

            row = {
                "best_candidate":
                    best_candidate,
                "best_mean_logprob":
                    best_score[
                        "mean_logprob"
                    ],
                "correct_answer":
                    record.answer,
                "correct_first_token_rank":
                    correct_score[
                        "first_token_rank"
                    ],
                "correct_mean_logprob":
                    correct_score[
                        "mean_logprob"
                    ],
                "correct_rank":
                    correct_rank,
                "direct_output":
                    direct,
                "direct_pass":
                    direct_ok,
                "normalized_output":
                    normalized,
                "normalized_pass":
                    normalized_ok,
                "operation":
                    kind,
                "operands":
                    list(
                        operands
                    ),
                "prefix_match":
                    prefix_match,
                "prompt":
                    record.prompt,
                "top_candidates": [
                    {
                        "answer":
                            candidate,
                        "mean_logprob":
                            score[
                                "mean_logprob"
                            ],
                    }
                    for candidate, score
                    in scored[:5]
                ],
            }
            rows.append(
                row
            )

            if (
                correct_rank == 1
                and not direct_ok
                and len(
                    decode_mismatch_examples
                )
                < args.examples
            ):
                decode_mismatch_examples.append(
                    row
                )

            if (
                correct_rank > 5
                and len(
                    representation_failure_examples
                )
                < args.examples
            ):
                representation_failure_examples.append(
                    row
                )

    print(
        "VN97 P4E-L SUMMARY "
        f"direct={summary['direct_pass']}/{summary['tasks']} "
        f"normalized={summary['normalized_pass']}/{summary['tasks']} "
        f"likelihood_rank1={summary['rank1']}/{summary['tasks']} "
        f"rank3={summary['rank3']}/{summary['tasks']} "
        f"rank5={summary['rank5']}/{summary['tasks']} "
        f"first_top1={summary['first_top1']}/{summary['tasks']} "
        f"first_top5={summary['first_top5']}/{summary['tasks']} "
        f"prefix={summary['prefix_match']}/{summary['tasks']} "
        f"rank1_decode_fail={summary['rank1_but_decode_fail']}/{summary['tasks']}",
        flush=True,
    )

    for operation in sorted(
        by_operation
    ):
        op = by_operation[
            operation
        ]
        print(
            "VN97 P4E-L OP "
            f"{operation} "
            f"direct={op['direct_pass']}/{op['tasks']} "
            f"normalized={op['normalized_pass']}/{op['tasks']} "
            f"rank1={op['rank1']}/{op['tasks']} "
            f"rank3={op['rank3']}/{op['tasks']}",
            flush=True,
        )

    for label, examples in (
        (
            "RANK1-BUT-DECODE-FAIL",
            decode_mismatch_examples,
        ),
        (
            "CORRECT-RANK-GT5",
            representation_failure_examples,
        ),
    ):
        for row in examples:
            print(
                "\n"
                + "=" * 78,
                flush=True,
            )
            print(
                f"P4E-L {label}",
                flush=True,
            )
            print(
                "PROMPT_JSON="
                + json.dumps(
                    row["prompt"],
                    ensure_ascii=False,
                ),
                flush=True,
            )
            print(
                "EXPECTED_JSON="
                + json.dumps(
                    row[
                        "correct_answer"
                    ],
                    ensure_ascii=False,
                ),
                flush=True,
            )
            print(
                "DIRECT_JSON="
                + json.dumps(
                    row[
                        "direct_output"
                    ],
                    ensure_ascii=False,
                ),
                flush=True,
            )
            print(
                "CORRECT_RANK="
                + str(
                    row[
                        "correct_rank"
                    ]
                ),
                flush=True,
            )
            print(
                "TOP_CANDIDATES_JSON="
                + json.dumps(
                    row[
                        "top_candidates"
                    ],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )

    report = {
        "checkpoint_sha256":
            checkpoint.checkpoint_sha256,
        "decode_mismatch_examples":
            decode_mismatch_examples,
        "parent_selected_candidate":
            parent_report.get(
                "selected_candidate"
            ),
        "parent_status":
            parent_report.get(
                "status"
            ),
        "representation_failure_examples":
            representation_failure_examples,
        "rows":
            rows,
        "schema":
            P4E_L_REPORT_SCHEMA,
        "summary":
            dict(
                summary
            ),
        "by_operation": {
            key:
                dict(
                    value
                )
            for key, value
            in sorted(
                by_operation.items()
            )
        },
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
                report
            )
            + b"\n"
        )
        print(
            f"VN97 P4E-L REPORT {path}",
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
            "vn97-p4e-arithmetic-mechanism-audit: "
            f"{exc}",
            file=sys.stderr,
        )
        raise
