from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

from .deployment_checkpoint import (
    load_deployment_checkpoint_file,
    save_deployment_checkpoint,
)
from .model_image import build_model_image
from .p3_language_campaign import (
    BATCH_SIZE as P3_BATCH_SIZE,
    MAX_VALIDATION_WINDOWS as P3_MAX_VALIDATION_WINDOWS,
    SEQUENCE_LENGTH as P3_SEQUENCE_LENGTH,
)
from .p3_language_campaign_cli import (
    _validate_corpus,
)
from .p3_tensor_cache import (
    evaluate_vn97_from_tensors,
    train_vn97_from_tensors,
)
from .p4_arithmetic_coverage_repair_cli import (
    _verify_parent,
)
from .p4_compositional_repair_cli import (
    _canonical_measure_dev,
    _canonical_measure_records,
)
from .p4_generalization_cli import (
    _canonical_json,
    _probe_records,
    _select_replay,
    _sha256,
    _windows_to_tensors,
)
from .p4_generalization_curriculum import (
    P4D_CATEGORIES,
    curriculum_bytes,
    default_validation,
)
from .p4_interference_safe_reasoning_curriculum import (
    P4E_K_CANDIDATES,
    P4E_K_PROFILE_ID,
    candidate_training_records,
    held_out_expressions,
)
from .p4_numeric_arithmetic_repair_cli import (
    _measure_copy_suite,
)
from .p4_task_evaluation import (
    load_p4_task_suite,
)
from .tokenizer import VN97Tokenizer
from .training import (
    VN97TrainingConfig,
    build_training_windows,
    encode_chat_completion_messages,
    encode_chat_messages,
)
from .training_cli import (
    _atomic_write,
    _load_records,
)


P4E_K_REPORT_SCHEMA = "VN97P4EK1"


class VN97P4EKError(RuntimeError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Interference-safe reasoning candidate search starting from "
            "the P4E-I checkpoint."
        )
    )
    parser.add_argument(
        "--p4e-i-dir",
        required=True,
    )
    parser.add_argument(
        "--p3-corpus-dir",
        required=True,
    )
    parser.add_argument(
        "--dev-suite",
        default=None,
    )
    parser.add_argument(
        "--work-dir",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--progress-interval-steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--checkpoint-interval-steps",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--cpu-prefetch-workers",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--min-reasoning-passes",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--min-numeric-copy-passes",
        type=int,
        default=48,
    )
    parser.add_argument(
        "--max-overall-drop",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max-copy-drop",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--max-tool-drop",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-authority-drop",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-other-category-drop",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max-dev-drop",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-token-validation-loss-increase",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--max-token-top1-drop",
        type=float,
        default=0.003,
    )
    parser.add_argument(
        "--max-p3-validation-loss-increase",
        type=float,
        default=0.07,
    )
    parser.add_argument(
        "--max-p3-top1-drop",
        type=float,
        default=0.007,
    )
    return parser


def _metric_summary(
    *,
    generation: dict[str, object],
    copy: dict[str, object],
    token,
    p3,
    dev: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "generation":
            generation,
        "numeric_copy":
            copy,
        "token_validation": {
            "mean_loss":
                token.mean_loss,
            "top1_accuracy":
                token.top1_accuracy,
        },
        "p3_retention": {
            "mean_loss":
                p3.mean_loss,
            "top1_accuracy":
                p3.top1_accuracy,
        },
        "dev":
            dev,
    }


def _safe_candidate(
    *,
    metrics: dict[str, object],
    baseline: dict[str, object],
    args: argparse.Namespace,
) -> tuple[
    bool,
    list[str],
]:
    reasons: list[str] = []

    gen = metrics[
        "generation"
    ]
    base_gen = baseline[
        "generation"
    ]
    copy = metrics[
        "numeric_copy"
    ]
    base_copy = baseline[
        "numeric_copy"
    ]
    token = metrics[
        "token_validation"
    ]
    base_token = baseline[
        "token_validation"
    ]
    p3 = metrics[
        "p3_retention"
    ]
    base_p3 = baseline[
        "p3_retention"
    ]

    overall_floor = (
        int(
            base_gen["passed"]
        )
        - args.max_overall_drop
    )
    if int(
        gen["passed"]
    ) < overall_floor:
        reasons.append(
            "overall"
        )

    copy_floor = max(
        args.min_numeric_copy_passes,
        int(
            base_copy["passed"]
        )
        - args.max_copy_drop,
    )
    if int(
        copy["passed"]
    ) < copy_floor:
        reasons.append(
            "numeric_copy"
        )

    categories = gen[
        "categories"
    ]
    base_categories = (
        base_gen[
            "categories"
        ]
    )

    category_drops = {
        "tool_intent":
            args.max_tool_drop,
        "authority_behavior":
            args.max_authority_drop,
        "instruction_following":
            args.max_other_category_drop,
        "memory_use":
            args.max_other_category_drop,
        "structured_cognition":
            args.max_other_category_drop,
    }
    for (
        category,
        allowed_drop,
    ) in category_drops.items():
        floor = (
            int(
                base_categories[
                    category
                ]["passed"]
            )
            - allowed_drop
        )
        if int(
            categories[
                category
            ]["passed"]
        ) < floor:
            reasons.append(
                category
            )

    if (
        float(
            token["mean_loss"]
        )
        > float(
            base_token[
                "mean_loss"
            ]
        )
        + args.max_token_validation_loss_increase
        or float(
            token[
                "top1_accuracy"
            ]
        )
        < float(
            base_token[
                "top1_accuracy"
            ]
        )
        - args.max_token_top1_drop
    ):
        reasons.append(
            "token_validation"
        )

    if (
        float(
            p3["mean_loss"]
        )
        > float(
            base_p3[
                "mean_loss"
            ]
        )
        + args.max_p3_validation_loss_increase
        or float(
            p3[
                "top1_accuracy"
            ]
        )
        < float(
            base_p3[
                "top1_accuracy"
            ]
        )
        - args.max_p3_top1_drop
    ):
        reasons.append(
            "p3_retention"
        )

    dev = metrics.get(
        "dev"
    )
    base_dev = baseline.get(
        "dev"
    )
    if (
        dev is not None
        and base_dev is not None
        and int(
            dev["passed"]
        )
        < int(
            base_dev["passed"]
        )
        - args.max_dev_drop
    ):
        reasons.append(
            "dev"
        )

    return (
        not reasons,
        reasons,
    )


def _selection_score(
    metrics: dict[str, object],
) -> tuple[
    int,
    int,
    int,
    int,
    float,
]:
    generation = metrics[
        "generation"
    ]
    categories = generation[
        "categories"
    ]
    dev = metrics.get(
        "dev"
    )

    return (
        int(
            categories[
                "reasoning_planning"
            ]["passed"]
        ),
        int(
            generation[
                "passed"
            ]
        ),
        int(
            metrics[
                "numeric_copy"
            ]["passed"]
        ),
        int(
            dev["passed"]
        )
        if dev is not None
        else 0,
        -float(
            metrics[
                "token_validation"
            ]["mean_loss"]
        ),
    )


def _completed_candidate(
    *,
    report_path: Path,
    checkpoint_path: Path,
) -> dict[str, object] | None:
    if (
        not report_path.exists()
        and not checkpoint_path.exists()
    ):
        return None
    if (
        not report_path.is_file()
        or not checkpoint_path.is_file()
    ):
        raise VN97P4EKError(
            "partial completed P4E-K candidate artifact"
        )

    report = json.loads(
        report_path.read_text(
            encoding="utf-8"
        )
    )
    if (
        not isinstance(
            report,
            dict,
        )
        or report.get(
            "schema"
        )
        != "VN97P4EKCANDIDATE1"
    ):
        raise VN97P4EKError(
            "invalid P4E-K candidate report"
        )

    checkpoint = (
        load_deployment_checkpoint_file(
            checkpoint_path
        )
    )
    if (
        checkpoint.checkpoint_sha256
        != report.get(
            "checkpoint_sha256"
        )
    ):
        raise VN97P4EKError(
            "P4E-K candidate checkpoint identity mismatch"
        )

    return report


def main(
    argv: list[str] | None = None,
) -> int:
    args = _parser().parse_args(
        argv
    )

    if (
        not torch.cuda.is_available()
        or not str(
            args.device
        ).startswith(
            "cuda"
        )
    ):
        raise VN97P4EKError(
            "P4E-K requires CUDA"
        )
    if (
        args.sequence_length < 2
        or args.batch_size <= 0
        or args.micro_batch_size <= 0
        or (
            args.batch_size
            % args.micro_batch_size
        )
        != 0
        or not math.isfinite(
            args.weight_decay
        )
        or args.weight_decay < 0.0
        or args.min_reasoning_passes <= 0
        or args.min_numeric_copy_passes <= 0
    ):
        raise VN97P4EKError(
            "P4E-K arguments are invalid"
        )

    output = Path(
        args.output_dir
    )
    if (
        output.exists()
        and (
            output.is_symlink()
            or not output.is_dir()
            or any(
                output.iterdir()
            )
        )
    ):
        raise VN97P4EKError(
            "P4E-K output-dir must be new or empty"
        )

    (
        parent_root,
        parent_checkpoint,
        parent_package,
        parent_report,
        parent_sums,
    ) = _verify_parent(
        Path(
            args.p4e_i_dir
        )
    )

    p3_root = Path(
        args.p3_corpus_dir
    ).resolve(
        strict=True
    )
    _validate_corpus(
        p3_root
    )

    (
        p3_training_records,
        p3_training_sha,
    ) = _load_records(
        [
            p3_root
            / "training.jsonl"
        ],
        mode="chat",
        max_input_bytes=
            64 * 1024 * 1024,
        max_examples=100_000,
    )
    (
        p3_validation_records,
        p3_validation_sha,
    ) = _load_records(
        [
            p3_root
            / "validation.jsonl"
        ],
        mode="chat",
        max_input_bytes=
            64 * 1024 * 1024,
        max_examples=100_000,
    )

    retention = (
        parent_report.get(
            "p3_retention"
        )
    )
    if (
        not isinstance(
            retention,
            dict,
        )
        or retention.get(
            "training_dataset_sha256"
        )
        != p3_training_sha
        or retention.get(
            "validation_dataset_sha256"
        )
        != p3_validation_sha
    ):
        raise VN97P4EKError(
            "P3 corpus identity does not match P4E-I parent"
        )

    tokenizer = VN97Tokenizer(
        parent_package
    )
    validation_records = (
        default_validation()
    )
    probe_records = (
        _probe_records(
            validation_records,
            per_category=30,
        )
    )

    suite_path = (
        Path(
            args.dev_suite
        )
        if args.dev_suite
        is not None
        else None
    )
    dev_prompts: tuple[
        str,
        ...
    ] = ()
    if suite_path is not None:
        suite = (
            load_p4_task_suite(
                suite_path
            )
        )
        dev_prompts = tuple(
            task.prompt
            for task in suite.tasks
            if task.category
            == "reasoning_planning"
        )

    forbidden_expressions = (
        held_out_expressions(
            dev_prompts
        )
    )

    validation_config = (
        VN97TrainingConfig(
            sequence_length=
                args.sequence_length,
            batch_size=
                args.batch_size,
            epochs=1,
            seed=0,
            shuffle=False,
            max_windows=100_000,
        )
    )
    validation_examples = [
        encode_chat_completion_messages(
            tokenizer,
            item.messages,
        )
        for item
        in validation_records
    ]
    validation_windows = (
        build_training_windows(
            validation_examples,
            validation_config,
            pad_token_id=
                tokenizer.pad_id,
        )
    )
    (
        validation_inputs,
        validation_labels,
    ) = _windows_to_tensors(
        validation_windows
    )

    p3_validation_config = (
        VN97TrainingConfig(
            sequence_length=
                P3_SEQUENCE_LENGTH,
            batch_size=
                P3_BATCH_SIZE,
            epochs=1,
            seed=0,
            shuffle=False,
            max_windows=
                P3_MAX_VALIDATION_WINDOWS,
        )
    )
    p3_validation_examples = [
        encode_chat_messages(
            tokenizer,
            record,
        )
        for record
        in p3_validation_records
    ]
    p3_validation_windows = (
        build_training_windows(
            p3_validation_examples,
            p3_validation_config,
            pad_token_id=
                tokenizer.pad_id,
        )
    )
    (
        p3_inputs,
        p3_labels,
    ) = _windows_to_tensors(
        p3_validation_windows
    )

    baseline_model = (
        parent_checkpoint.model
    )
    baseline_generation = (
        _canonical_measure_records(
            model=
                baseline_model,
            tokenizer=
                tokenizer,
            records=
                probe_records,
            device=
                args.device,
        )
    )
    baseline_copy = (
        _measure_copy_suite(
            model=
                baseline_model,
            tokenizer=
                tokenizer,
            device=
                args.device,
        )
    )
    baseline_dev = None
    if suite_path is not None:
        baseline_dev = (
            _canonical_measure_dev(
                model=
                    baseline_model,
                tokenizer=
                    tokenizer,
                suite_path=
                    suite_path,
                device=
                    args.device,
            )
        )
    baseline_token = (
        evaluate_vn97_from_tensors(
            baseline_model,
            validation_inputs,
            validation_labels,
            batch_size=
                args.batch_size,
            device=
                args.device,
            cpu_prefetch_workers=
                args.cpu_prefetch_workers,
            micro_batch_size=
                args.micro_batch_size,
        )
    )
    baseline_p3 = (
        evaluate_vn97_from_tensors(
            baseline_model,
            p3_inputs,
            p3_labels,
            batch_size=
                P3_BATCH_SIZE,
            device=
                args.device,
            cpu_prefetch_workers=
                args.cpu_prefetch_workers,
            micro_batch_size=2,
        )
    )

    baseline = _metric_summary(
        generation=
            baseline_generation,
        copy=
            baseline_copy,
        token=
            baseline_token,
        p3=
            baseline_p3,
        dev=
            baseline_dev,
    )

    baseline_reasoning = int(
        baseline_generation[
            "categories"
        ][
            "reasoning_planning"
        ]["passed"]
    )

    print(
        "VN97 P4E-K BASELINE "
        f"canonical={baseline_generation['passed']}/"
        f"{baseline_generation['task_count']} "
        f"copy={baseline_copy['passed']}/"
        f"{baseline_copy['task_count']} "
        f"reasoning={baseline_reasoning}/30",
        flush=True,
    )

    work = Path(
        args.work_dir
    )
    if work.is_symlink():
        raise VN97P4EKError(
            "P4E-K work-dir must not be a symlink"
        )
    work.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_id = "parent"
    best_metrics = baseline
    best_checkpoint_path = (
        parent_root
        / "model.vn97ck1"
    )
    best_score = (
        _selection_score(
            baseline
        )
    )
    candidate_reports: list[
        dict[str, object]
    ] = []

    for profile in P4E_K_CANDIDATES:
        candidate_id = (
            profile.candidate_id
        )
        report_path = (
            work
            / (
                "candidate-"
                + candidate_id
                + ".json"
            )
        )
        checkpoint_path = (
            work
            / (
                "candidate-"
                + candidate_id
                + ".vn97ck1"
            )
        )
        resume_path = (
            work
            / (
                "candidate-"
                + candidate_id
                + ".resume.pt"
            )
        )

        completed = (
            _completed_candidate(
                report_path=
                    report_path,
                checkpoint_path=
                    checkpoint_path,
            )
        )

        if completed is None:
            model = (
                load_deployment_checkpoint_file(
                    parent_root
                    / "model.vn97ck1"
                ).model
            )

            train_records = (
                candidate_training_records(
                    profile,
                    forbidden_expressions=
                        forbidden_expressions,
                )
            )
            validation_prompts = {
                item.prompt
                for item
                in validation_records
            }
            if {
                item.prompt
                for item
                in train_records
            }.intersection(
                validation_prompts
            ):
                raise VN97P4EKError(
                    "P4E-K candidate overlaps validation prompts"
                )

            replay_records = (
                _select_replay(
                    p3_training_records,
                    count=
                        profile.p3_replay_records,
                )
            )

            train_examples = [
                encode_chat_completion_messages(
                    tokenizer,
                    item.messages,
                )
                for item
                in train_records
            ]
            train_examples.extend(
                encode_chat_completion_messages(
                    tokenizer,
                    record,
                )
                for record
                in replay_records
            )

            config = (
                VN97TrainingConfig(
                    sequence_length=
                        args.sequence_length,
                    batch_size=
                        args.batch_size,
                    epochs=1,
                    learning_rate=
                        profile.learning_rate,
                    weight_decay=
                        args.weight_decay,
                    max_grad_norm=
                        args.max_grad_norm,
                    seed=
                        131197
                        + profile.seed_offset,
                    shuffle=True,
                    max_windows=100_000,
                )
            )
            train_windows = (
                build_training_windows(
                    train_examples,
                    config,
                    pad_token_id=
                        tokenizer.pad_id,
                )
            )
            (
                train_inputs,
                train_labels,
            ) = _windows_to_tensors(
                train_windows
            )

            train_bytes = (
                curriculum_bytes(
                    train_records
                )
            )
            replay_bytes = b"".join(
                (
                    _canonical_json(
                        {
                            "messages": [
                                {
                                    "content":
                                        message.content,
                                    "role":
                                        message.role,
                                }
                                for message
                                in record
                            ]
                        }
                    )
                    + b"\n"
                )
                for record
                in replay_records
            )
            resume_identity = (
                _sha256(
                    b"VN97P4EKCANDIDATE1\0"
                    + _canonical_json(
                        {
                            "candidate_id":
                                candidate_id,
                            "learning_rate":
                                profile.learning_rate,
                            "p3_replay_sha256":
                                _sha256(
                                    replay_bytes
                                ),
                            "parent_checkpoint_sha256":
                                parent_checkpoint.checkpoint_sha256,
                            "profile_id":
                                P4E_K_PROFILE_ID,
                            "sequence_length":
                                args.sequence_length,
                            "training_sha256":
                                _sha256(
                                    train_bytes
                                ),
                        }
                    )
                )
            )

            print(
                "VN97 P4E-K CANDIDATE START "
                f"id={candidate_id} "
                f"records={len(train_records)} "
                f"p3={len(replay_records)} "
                f"lr={profile.learning_rate}",
                flush=True,
            )

            training = (
                train_vn97_from_tensors(
                    model,
                    train_inputs,
                    train_labels,
                    config,
                    device=
                        args.device,
                    cpu_prefetch_workers=
                        args.cpu_prefetch_workers,
                    micro_batch_size=
                        args.micro_batch_size,
                    progress_label=
                        candidate_id,
                    progress_interval_steps=
                        args.progress_interval_steps,
                    resume_checkpoint_path=
                        resume_path,
                    resume_identity=
                        resume_identity,
                    checkpoint_interval_steps=
                        args.checkpoint_interval_steps,
                    progress_protocol=
                        "VN97 P4E-K",
                )
            )

            generation = (
                _canonical_measure_records(
                    model=model,
                    tokenizer=
                        tokenizer,
                    records=
                        probe_records,
                    device=
                        args.device,
                )
            )
            copy = (
                _measure_copy_suite(
                    model=model,
                    tokenizer=
                        tokenizer,
                    device=
                        args.device,
                )
            )
            dev = None
            if suite_path is not None:
                dev = (
                    _canonical_measure_dev(
                        model=model,
                        tokenizer=
                            tokenizer,
                        suite_path=
                            suite_path,
                        device=
                            args.device,
                    )
                )
            token = (
                evaluate_vn97_from_tensors(
                    model,
                    validation_inputs,
                    validation_labels,
                    batch_size=
                        args.batch_size,
                    device=
                        args.device,
                    cpu_prefetch_workers=
                        args.cpu_prefetch_workers,
                    micro_batch_size=
                        args.micro_batch_size,
                )
            )
            p3 = (
                evaluate_vn97_from_tensors(
                    model,
                    p3_inputs,
                    p3_labels,
                    batch_size=
                        P3_BATCH_SIZE,
                    device=
                        args.device,
                    cpu_prefetch_workers=
                        args.cpu_prefetch_workers,
                    micro_batch_size=2,
                )
            )

            metrics = _metric_summary(
                generation=
                    generation,
                copy=
                    copy,
                token=
                    token,
                p3=
                    p3,
                dev=
                    dev,
            )

            checkpoint_sha = (
                save_deployment_checkpoint(
                    model,
                    checkpoint_path,
                )
            )
            completed = {
                "candidate_id":
                    candidate_id,
                "checkpoint_sha256":
                    checkpoint_sha,
                "learning_rate":
                    profile.learning_rate,
                "metrics":
                    metrics,
                "schema":
                    "VN97P4EKCANDIDATE1",
                "training": {
                    "mean_loss":
                        training.mean_loss,
                    "steps":
                        training.steps,
                    "target_tokens":
                        training.target_tokens,
                },
                "training_records":
                    len(
                        train_records
                    ),
                "training_sha256":
                    _sha256(
                        train_bytes
                    ),
            }
            _atomic_write(
                report_path,
                _canonical_json(
                    completed
                )
                + b"\n",
            )
            resume_path.unlink(
                missing_ok=True
            )

        metrics = completed[
            "metrics"
        ]
        safe, unsafe_reasons = (
            _safe_candidate(
                metrics=metrics,
                baseline=baseline,
                args=args,
            )
        )
        score = (
            _selection_score(
                metrics
            )
        )
        reasoning = int(
            metrics[
                "generation"
            ][
                "categories"
            ][
                "reasoning_planning"
            ]["passed"]
        )

        annotated = {
            **completed,
            "safe":
                safe,
            "unsafe_reasons":
                unsafe_reasons,
        }
        candidate_reports.append(
            annotated
        )

        print(
            "VN97 P4E-K CANDIDATE RESULT "
            f"id={candidate_id} "
            f"safe={str(safe).lower()} "
            f"canonical={metrics['generation']['passed']}/"
            f"{metrics['generation']['task_count']} "
            f"copy={metrics['numeric_copy']['passed']}/"
            f"{metrics['numeric_copy']['task_count']} "
            f"reasoning={reasoning}/30 "
            f"unsafe={','.join(unsafe_reasons) if unsafe_reasons else 'none'}",
            flush=True,
        )

        if (
            safe
            and score
            > best_score
        ):
            best_id = (
                candidate_id
            )
            best_metrics = (
                metrics
            )
            best_checkpoint_path = (
                checkpoint_path
            )
            best_score = score

        if (
            safe
            and reasoning
            >= args.min_reasoning_passes
            and int(
                metrics[
                    "generation"
                ]["passed"]
            )
            >= int(
                baseline_generation[
                    "passed"
                ]
            )
        ):
            print(
                "VN97 P4E-K EARLY STOP "
                f"id={candidate_id} "
                "reason=eligible_safe_gain",
                flush=True,
            )
            break

    best_reasoning = int(
        best_metrics[
            "generation"
        ][
            "categories"
        ][
            "reasoning_planning"
        ]["passed"]
    )
    best_overall = int(
        best_metrics[
            "generation"
        ]["passed"]
    )
    best_copy = int(
        best_metrics[
            "numeric_copy"
        ]["passed"]
    )

    if best_id == "parent":
        status = (
            "REJECTED_NO_SAFE_GAIN"
        )
    elif best_reasoning < (
        args.min_reasoning_passes
    ):
        status = (
            "REJECTED_REASONING"
        )
    elif best_overall < int(
        baseline_generation[
            "passed"
        ]
    ):
        status = (
            "REJECTED_GENERALIZATION"
        )
    elif best_copy < max(
        args.min_numeric_copy_passes,
        int(
            baseline_copy[
                "passed"
            ]
        )
        - args.max_copy_drop,
    ):
        status = (
            "REJECTED_NUMERIC_COPY"
        )
    else:
        status = "ELIGIBLE"

    selected = (
        load_deployment_checkpoint_file(
            best_checkpoint_path
        )
    )
    selected_model = (
        selected.model
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )
    checkpoint_path = (
        output
        / "model.vn97ck1"
    )
    checkpoint_sha = (
        save_deployment_checkpoint(
            selected_model,
            checkpoint_path,
        )
    )
    tokenizer_bytes = (
        parent_package.to_bytes()
    )
    _atomic_write(
        output
        / "tokenizer.vn97tk1",
        tokenizer_bytes,
    )
    image_bytes = (
        build_model_image(
            selected_model,
            tokenizer=
                parent_package,
            tile_rows=16,
            tile_cols=16,
        ).data
    )
    _atomic_write(
        output
        / "model.vn97mi1",
        image_bytes,
    )

    report = {
        "baseline":
            baseline,
        "candidates":
            candidate_reports,
        "forbidden_expressions":
            sorted(
                forbidden_expressions
            ),
        "model_image_sha256":
            _sha256(
                image_bytes
            ),
        "output_checkpoint_sha256":
            checkpoint_sha,
        "parent_p4e_i": {
            "checkpoint_sha256":
                parent_checkpoint.checkpoint_sha256,
            "report_sha256":
                parent_sums[
                    "p4i-report.json"
                ],
            "status":
                parent_report.get(
                    "status"
                ),
        },
        "profile_id":
            P4E_K_PROFILE_ID,
        "schema":
            P4E_K_REPORT_SCHEMA,
        "selected_candidate":
            best_id,
        "selected_metrics":
            best_metrics,
        "status":
            status,
        "tokenizer_sha256":
            _sha256(
                tokenizer_bytes
            ),
    }
    report_bytes = (
        _canonical_json(
            report
        )
        + b"\n"
    )
    _atomic_write(
        output
        / "p4k-report.json",
        report_bytes,
    )

    files = {
        "model.vn97ck1":
            checkpoint_path.read_bytes(),
        "model.vn97mi1":
            image_bytes,
        "p4k-report.json":
            report_bytes,
        "tokenizer.vn97tk1":
            tokenizer_bytes,
    }
    sums = b"".join(
        (
            _sha256(
                files[name]
            )
            + "  "
            + name
            + "\n"
        ).encode(
            "ascii"
        )
        for name
        in sorted(
            files
        )
    )
    _atomic_write(
        output
        / "SHA256SUMS",
        sums,
    )

    print(
        "VN97P4EK1 "
        f"status={status} "
        f"selected={best_id} "
        f"checkpoint={checkpoint_sha} "
        f"canonical_before={baseline_generation['passed']}/"
        f"{baseline_generation['task_count']} "
        f"canonical_after={best_overall}/"
        f"{best_metrics['generation']['task_count']} "
        f"copy_before={baseline_copy['passed']}/"
        f"{baseline_copy['task_count']} "
        f"copy_after={best_copy}/"
        f"{best_metrics['numeric_copy']['task_count']} "
        f"reasoning_before={baseline_reasoning}/30 "
        f"reasoning_after={best_reasoning}/30",
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
            "vn97-p4e-interference-safe-reasoning: "
            f"{exc}",
            file=sys.stderr,
        )
        raise
