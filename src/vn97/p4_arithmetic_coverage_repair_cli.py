from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

from .cognition_adapter import (
    TorchVN97InferenceEngine,
    VN97InferenceLimits,
)
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
from .p4_arithmetic_coverage_curriculum import (
    P4E_J_PROFILE_ID,
    arithmetic_coverage_training,
    extract_expression,
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
from .p4_numeric_arithmetic_repair_cli import (
    _measure_copy_suite,
)
from .p4_task_evaluation import (
    load_p4_task_suite,
)
from .tokenizer import (
    VN97Tokenizer,
    VN97TokenizerPackage,
)
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


P4E_J_REPORT_SCHEMA = "VN97P4EJ1"
DEFAULT_P3_REPLAY_RECORDS = 1000


class VN97P4EJError(RuntimeError):
    pass


def _verify_parent(
    root: Path,
):
    resolved = root.resolve(
        strict=True
    )
    required = {
        "SHA256SUMS",
        "model.vn97ck1",
        "model.vn97mi1",
        "p4i-report.json",
        "tokenizer.vn97tk1",
    }
    names = {
        item.name
        for item in resolved.iterdir()
    }
    if (
        not resolved.is_dir()
        or resolved.is_symlink()
        or names != required
    ):
        raise VN97P4EJError(
            "P4E-I parent artifact file set mismatch"
        )

    sums: dict[
        str,
        str,
    ] = {}
    sums_text = (
        resolved
        / "SHA256SUMS"
    ).read_text(
        encoding="ascii"
    )
    if not sums_text.endswith(
        "\n"
    ):
        raise VN97P4EJError(
            "P4E-I SHA256SUMS must end with newline"
        )

    for line in sums_text.splitlines():
        if (
            len(line) < 67
            or line[64:66] != "  "
        ):
            raise VN97P4EJError(
                "malformed P4E-I SHA256SUMS"
            )
        digest = line[:64]
        name = line[66:]
        sums[name] = digest

    if set(sums) != (
        required
        - {"SHA256SUMS"}
    ):
        raise VN97P4EJError(
            "P4E-I SHA256SUMS file set mismatch"
        )

    for name, expected in sums.items():
        actual = hashlib.sha256(
            (
                resolved
                / name
            ).read_bytes()
        ).hexdigest()
        if actual != expected:
            raise VN97P4EJError(
                f"P4E-I SHA256 mismatch: {name}"
            )

    report = json.loads(
        (
            resolved
            / "p4i-report.json"
        ).read_text(
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
        != "VN97P4EI1"
    ):
        raise VN97P4EJError(
            "parent report is not VN97P4EI1"
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
        raise VN97P4EJError(
            "parent checkpoint/tokenizer vocabulary mismatch"
        )

    return (
        resolved,
        checkpoint,
        package,
        report,
        sums,
    )


def _dev_forbidden_expressions(
    suite_path: Path | None,
) -> set[str]:
    if suite_path is None:
        return set()

    suite = load_p4_task_suite(
        suite_path
    )
    result: set[str] = set()
    for task in suite.tasks:
        if (
            task.category
            != "reasoning_planning"
        ):
            continue
        expression = extract_expression(
            task.prompt
        )
        if expression is not None:
            result.add(
                expression
            )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "P4E-J held-out-safe arithmetic coverage repair."
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
        "--epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
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
        "--seed",
        type=int,
        default=115197,
    )
    parser.add_argument(
        "--p3-replay-records",
        type=int,
        default=DEFAULT_P3_REPLAY_RECORDS,
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
        "--max-token-validation-loss-increase",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--max-token-top1-drop",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--max-p3-validation-loss-increase",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--max-p3-top1-drop",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--min-copy-passes",
        type=int,
        default=48,
    )
    parser.add_argument(
        "--min-reasoning-passes",
        type=int,
        default=8,
    )
    return parser


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
        raise VN97P4EJError(
            "P4E-J requires CUDA"
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
        or args.epochs <= 0
        or not math.isfinite(
            args.learning_rate
        )
        or args.learning_rate <= 0.0
    ):
        raise VN97P4EJError(
            "P4E-J arguments are invalid"
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
        raise VN97P4EJError(
            "P4E-J output-dir must be new or empty"
        )

    (
        _parent_root,
        parent_checkpoint,
        parent_package,
        parent_report,
        parent_sums,
    ) = _verify_parent(
        Path(
            args.p4e_i_dir
        )
    )

    tokenizer = VN97Tokenizer(
        parent_package
    )
    model = (
        parent_checkpoint.model
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
        raise VN97P4EJError(
            "P3 corpus identity does not match P4E-I parent"
        )

    suite_path = (
        Path(
            args.dev_suite
        )
        if args.dev_suite
        is not None
        else None
    )
    forbidden = (
        _dev_forbidden_expressions(
            suite_path
        )
    )
    train_records = (
        arithmetic_coverage_training(
            forbidden_expressions=
                forbidden,
        )
    )
    validation_records = (
        default_validation()
    )
    replay_records = (
        _select_replay(
            p3_training_records,
            count=
                args.p3_replay_records,
        )
    )

    train_prompts = {
        item.prompt
        for item
        in train_records
    }
    validation_prompts = {
        item.prompt
        for item
        in validation_records
    }
    if train_prompts.intersection(
        validation_prompts
    ):
        raise VN97P4EJError(
            "P4E-J training overlaps validation prompts"
        )

    probe_records = (
        _probe_records(
            validation_records,
            per_category=30,
        )
    )

    generalization_before = (
        _canonical_measure_records(
            model=model,
            tokenizer=tokenizer,
            records=probe_records,
            device=args.device,
        )
    )
    copy_before = (
        _measure_copy_suite(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
        )
    )
    dev_before = None
    if suite_path is not None:
        dev_before = (
            _canonical_measure_dev(
                model=model,
                tokenizer=tokenizer,
                suite_path=
                    suite_path,
                device=args.device,
            )
        )

    print(
        "VN97 P4E-J GENERALIZATION BEFORE "
        f"canonical={generalization_before['passed']}/"
        f"{generalization_before['task_count']}",
        flush=True,
    )
    print(
        "VN97 P4E-J NUMERIC COPY BEFORE "
        f"passed={copy_before['passed']}/"
        f"{copy_before['task_count']}",
        flush=True,
    )
    print(
        "VN97 P4E-J COVERAGE "
        f"training_records={len(train_records)} "
        f"forbidden_expressions={len(forbidden)}",
        flush=True,
    )

    config = (
        VN97TrainingConfig(
            sequence_length=
                args.sequence_length,
            batch_size=
                args.batch_size,
            epochs=
                args.epochs,
            learning_rate=
                args.learning_rate,
            weight_decay=
                args.weight_decay,
            max_grad_norm=
                args.max_grad_norm,
            seed=
                args.seed,
            shuffle=True,
            max_windows=100_000,
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
    validation_examples = [
        encode_chat_completion_messages(
            tokenizer,
            item.messages,
        )
        for item
        in validation_records
    ]

    train_windows = (
        build_training_windows(
            train_examples,
            config,
            pad_token_id=
                tokenizer.pad_id,
        )
    )
    validation_windows = (
        build_training_windows(
            validation_examples,
            validation_config,
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

    token_before = (
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
    p3_before = (
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

    print(
        "VN97 P4E-J TOKEN BEFORE "
        f"loss={token_before.mean_loss:.6f} "
        f"top1={token_before.top1_accuracy:.6f}",
        flush=True,
    )
    print(
        "VN97 P4E-J P3 RETENTION BEFORE "
        f"loss={p3_before.mean_loss:.6f} "
        f"top1={p3_before.top1_accuracy:.6f}",
        flush=True,
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
            b"VN97P4EJRESUME1\0"
            + _canonical_json(
                {
                    "batch_size":
                        args.batch_size,
                    "epochs":
                        args.epochs,
                    "learning_rate":
                        args.learning_rate,
                    "micro_batch_size":
                        args.micro_batch_size,
                    "p3_replay_sha256":
                        _sha256(
                            replay_bytes
                        ),
                    "parent_checkpoint_sha256":
                        parent_checkpoint.checkpoint_sha256,
                    "profile_id":
                        P4E_J_PROFILE_ID,
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

    work = Path(
        args.work_dir
    )
    work.mkdir(
        parents=True,
        exist_ok=True,
    )
    resume_path = (
        work
        / "state.vn97p3resume1.pt"
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
                "arithmetic_coverage_repair",
            progress_interval_steps=
                args.progress_interval_steps,
            resume_checkpoint_path=
                resume_path,
            resume_identity=
                resume_identity,
            checkpoint_interval_steps=
                args.checkpoint_interval_steps,
            progress_protocol=
                "VN97 P4E-J",
        )
    )

    token_after = (
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
    p3_after = (
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
    generalization_after = (
        _canonical_measure_records(
            model=model,
            tokenizer=tokenizer,
            records=probe_records,
            device=args.device,
        )
    )
    copy_after = (
        _measure_copy_suite(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
        )
    )
    dev_after = None
    if suite_path is not None:
        dev_after = (
            _canonical_measure_dev(
                model=model,
                tokenizer=tokenizer,
                suite_path=
                    suite_path,
                device=args.device,
            )
        )

    print(
        "VN97 P4E-J TOKEN AFTER "
        f"loss={token_after.mean_loss:.6f} "
        f"top1={token_after.top1_accuracy:.6f}",
        flush=True,
    )
    print(
        "VN97 P4E-J P3 RETENTION AFTER "
        f"loss={p3_after.mean_loss:.6f} "
        f"top1={p3_after.top1_accuracy:.6f}",
        flush=True,
    )
    print(
        "VN97 P4E-J NUMERIC COPY AFTER "
        f"passed={copy_after['passed']}/"
        f"{copy_after['task_count']}",
        flush=True,
    )
    print(
        "VN97 P4E-J GENERALIZATION AFTER "
        f"canonical={generalization_after['passed']}/"
        f"{generalization_after['task_count']}",
        flush=True,
    )
    for category in P4D_CATEGORIES:
        row = (
            generalization_after[
                "categories"
            ][category]
        )
        print(
            "VN97 P4E-J CATEGORY AFTER "
            f"{category} "
            f"{row['passed']}/{row['tasks']}",
            flush=True,
        )
    if dev_after is not None:
        print(
            "VN97 P4E-J DEV AFTER "
            f"canonical={dev_after['passed']}/"
            f"{dev_after['task_count']}",
            flush=True,
        )

    token_ok = (
        token_after.mean_loss
        <= token_before.mean_loss
        + args.max_token_validation_loss_increase
        and token_after.top1_accuracy
        >= token_before.top1_accuracy
        - args.max_token_top1_drop
    )
    p3_ok = (
        p3_after.mean_loss
        <= p3_before.mean_loss
        + args.max_p3_validation_loss_increase
        and p3_after.top1_accuracy
        >= p3_before.top1_accuracy
        - args.max_p3_top1_drop
    )

    before_cats = (
        generalization_before[
            "categories"
        ]
    )
    after_cats = (
        generalization_after[
            "categories"
        ]
    )
    reasoning_after = (
        after_cats[
            "reasoning_planning"
        ]["passed"]
    )
    strong_ok = (
        after_cats[
            "tool_intent"
        ]["passed"]
        >= before_cats[
            "tool_intent"
        ]["passed"] - 2
        and after_cats[
            "authority_behavior"
        ]["passed"]
        >= before_cats[
            "authority_behavior"
        ]["passed"] - 2
    )
    overall_ok = (
        generalization_after[
            "passed"
        ]
        >= generalization_before[
            "passed"
        ] - 2
    )

    if not token_ok:
        status = "REJECTED_VALIDATION"
    elif not p3_ok:
        status = "REJECTED_RETENTION"
    elif not strong_ok:
        status = (
            "REJECTED_STRONG_CATEGORY_REGRESSION"
        )
    elif not overall_ok:
        status = (
            "REJECTED_GENERALIZATION_REGRESSION"
        )
    elif copy_after[
        "passed"
    ] < args.min_copy_passes:
        status = "REJECTED_NUMERIC_COPY"
    elif reasoning_after < (
        args.min_reasoning_passes
    ):
        status = "REJECTED_REASONING"
    else:
        status = "ELIGIBLE"

    output.mkdir(
        parents=True,
        exist_ok=True,
    )
    tokenizer_bytes = (
        parent_package.to_bytes()
    )
    _atomic_write(
        output
        / "tokenizer.vn97tk1",
        tokenizer_bytes,
    )
    checkpoint_path = (
        output
        / "model.vn97ck1"
    )
    checkpoint_sha = (
        save_deployment_checkpoint(
            model,
            checkpoint_path,
        )
    )
    image_bytes = (
        build_model_image(
            model,
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
        "canonical_after":
            generalization_after,
        "canonical_before":
            generalization_before,
        "coverage": {
            "dev_forbidden_expressions":
                sorted(
                    forbidden
                ),
            "training_records":
                len(
                    train_records
                ),
            "training_sha256":
                _sha256(
                    train_bytes
                ),
        },
        "dev_after":
            dev_after,
        "dev_before":
            dev_before,
        "model_image_sha256":
            _sha256(
                image_bytes
            ),
        "numeric_copy_after":
            copy_after,
        "numeric_copy_before":
            copy_before,
        "output_checkpoint_sha256":
            checkpoint_sha,
        "p3_retention": {
            "after": {
                "mean_loss":
                    p3_after.mean_loss,
                "top1_accuracy":
                    p3_after.top1_accuracy,
            },
            "before": {
                "mean_loss":
                    p3_before.mean_loss,
                "top1_accuracy":
                    p3_before.top1_accuracy,
            },
            "training_dataset_sha256":
                p3_training_sha,
            "validation_dataset_sha256":
                p3_validation_sha,
        },
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
            P4E_J_PROFILE_ID,
        "schema":
            P4E_J_REPORT_SCHEMA,
        "status":
            status,
        "token_validation_after": {
            "mean_loss":
                token_after.mean_loss,
            "top1_accuracy":
                token_after.top1_accuracy,
        },
        "token_validation_before": {
            "mean_loss":
                token_before.mean_loss,
            "top1_accuracy":
                token_before.top1_accuracy,
        },
        "tokenizer_sha256":
            _sha256(
                tokenizer_bytes
            ),
        "training": {
            "epochs":
                args.epochs,
            "learning_rate":
                args.learning_rate,
            "mean_loss":
                training.mean_loss,
            "steps":
                training.steps,
            "target_tokens":
                training.target_tokens,
        },
    }
    report_bytes = (
        _canonical_json(
            report
        )
        + b"\n"
    )
    _atomic_write(
        output
        / "p4j-report.json",
        report_bytes,
    )

    files = {
        "model.vn97ck1":
            checkpoint_path.read_bytes(),
        "model.vn97mi1":
            image_bytes,
        "p4j-report.json":
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

    if status == "ELIGIBLE":
        resume_path.unlink(
            missing_ok=True
        )

    print(
        "VN97P4EJ1 "
        f"status={status} "
        f"checkpoint={checkpoint_sha} "
        f"copy_before={copy_before['passed']}/"
        f"{copy_before['task_count']} "
        f"copy_after={copy_after['passed']}/"
        f"{copy_after['task_count']} "
        f"reasoning_after={reasoning_after} "
        f"canonical_before={generalization_before['passed']}/"
        f"{generalization_before['task_count']} "
        f"canonical_after={generalization_after['passed']}/"
        f"{generalization_after['task_count']}",
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
            "vn97-p4e-arithmetic-coverage-repair: "
            f"{exc}",
            file=sys.stderr,
        )
        raise
