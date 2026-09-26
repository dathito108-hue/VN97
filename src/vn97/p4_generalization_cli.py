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
from .p4_artifact import (
    verify_p4c_artifact,
)
from .p4_generalization_curriculum import (
    P4D_CATEGORIES,
    P4D_PROFILE_ID,
    TRAIN_PER_CATEGORY,
    TRAIN_SEED,
    VALIDATION_PER_CATEGORY,
    VALIDATION_SEED,
    curriculum_bytes,
    default_training,
    default_validation,
)
from .p4_task_evaluation import (
    load_p4_task_suite,
    render_p4_chat_prompt,
    score_p4_output,
)
from .tokenizer import VN97Tokenizer
from .training import (
    IGNORE_INDEX,
    VN97TrainingConfig,
    build_training_windows,
    encode_chat_completion_messages,
    encode_chat_messages,
    render_chat_text,
)
from .training_cli import (
    _atomic_write,
    _load_records,
)


P4D_REPORT_SCHEMA = "VN97P4D1"
DEFAULT_P3_REPLAY_RECORDS = 2000
DEFAULT_PROBE_PER_CATEGORY = 30


class VN97P4DGeneralizationError(
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


def _sha256(data: bytes) -> str:
    return hashlib.sha256(
        data
    ).hexdigest()


def _windows_to_tensors(
    windows,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    input_ids = torch.tensor(
        [
            item.input_ids
            for item in windows
        ],
        dtype=torch.int32,
    )
    labels = torch.tensor(
        [
            item.labels
            for item in windows
        ],
        dtype=torch.int32,
    )
    if (
        input_ids.ndim != 2
        or labels.shape
        != input_ids.shape
        or int(
            (
                labels
                != IGNORE_INDEX
            ).sum().item()
        )
        <= 0
    ):
        raise VN97P4DGeneralizationError(
            "P4D tensors are invalid"
        )
    return input_ids, labels


def _select_replay(
    records,
    *,
    count: int,
) -> tuple:
    if count < 0 or count > len(
        records
    ):
        raise VN97P4DGeneralizationError(
            "P3 replay count is invalid"
        )
    ranked = sorted(
        records,
        key=lambda record: (
            hashlib.sha256(
                render_chat_text(
                    record
                ).encode("utf-8")
            ).digest()
        ),
    )
    return tuple(
        ranked[:count]
    )


def _record_score(
    expected: str,
    output: str,
) -> bool:
    expected_text = expected.strip()
    output_text = output.strip()
    if expected_text.startswith(
        "{"
    ):
        try:
            expected_json = json.loads(
                expected_text
            )
            output_json = json.loads(
                output_text
            )
        except (
            json.JSONDecodeError,
            ValueError,
        ):
            return False
        return output_json == expected_json
    return output_text == expected_text


def _probe_records(
    records,
    *,
    per_category: int,
) -> tuple:
    if per_category <= 0:
        raise VN97P4DGeneralizationError(
            "probe_per_category must be positive"
        )
    selected = []
    for category in P4D_CATEGORIES:
        subset = [
            item
            for item in records
            if item.category
            == category
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
            raise VN97P4DGeneralizationError(
                "P4D validation category is too small for probe"
            )
        selected.extend(
            subset[:per_category]
        )
    return tuple(selected)


def _measure_generalization(
    *,
    model,
    tokenizer,
    records,
    device: str,
) -> dict[str, object]:
    engine = TorchVN97InferenceEngine(
        model,
        tokenizer,
        limits=VN97InferenceLimits(
            max_prompt_tokens=4096,
            repetition_penalty=1.12,
            no_repeat_ngram_size=4,
        ),
    )
    model.to(device)
    model.eval()

    categories = {
        category: {
            "passed": 0,
            "tasks": 0,
        }
        for category
        in P4D_CATEGORIES
    }
    results: list[
        dict[str, object]
    ] = []

    for record in records:
        output = ""
        error_type = ""
        try:
            output = engine.generate_text(
                render_p4_chat_prompt(
                    record.prompt
                ),
                max_new_tokens=96,
            )
            passed = _record_score(
                record.answer,
                output,
            )
        except Exception as exc:
            passed = False
            error_type = (
                type(exc).__name__
            )

        row = categories[
            record.category
        ]
        row["tasks"] += 1
        if passed:
            row["passed"] += 1

        output_bytes = output.encode(
            "utf-8"
        )
        results.append(
            {
                "category":
                    record.category,
                "error_type":
                    error_type,
                "output_sha256":
                    _sha256(
                        output_bytes
                    ),
                "output_utf8_bytes":
                    len(output_bytes),
                "passed":
                    passed,
                "prompt_sha256":
                    _sha256(
                        record.prompt.encode(
                            "utf-8"
                        )
                    ),
            }
        )

    normalized = {}
    for category, row in (
        categories.items()
    ):
        normalized[category] = {
            "pass_rate":
                row["passed"]
                / row["tasks"],
            "passed":
                row["passed"],
            "tasks":
                row["tasks"],
        }

    passed = sum(
        1
        for row in results
        if row["passed"]
    )
    return {
        "categories":
            normalized,
        "pass_rate":
            passed
            / len(results),
        "passed_tasks":
            passed,
        "results":
            results,
        "task_count":
            len(results),
    }


def _measure_dev_suite(
    *,
    model,
    tokenizer,
    suite_path: Path,
    device: str,
) -> dict[str, object]:
    suite = load_p4_task_suite(
        suite_path
    )
    engine = TorchVN97InferenceEngine(
        model,
        tokenizer,
        limits=VN97InferenceLimits(
            max_prompt_tokens=4096,
            repetition_penalty=1.12,
            no_repeat_ngram_size=4,
        ),
    )
    model.to(device)
    model.eval()
    results = []
    passed = 0
    for task in suite.tasks:
        output = ""
        error_type = ""
        try:
            output = engine.generate_text(
                render_p4_chat_prompt(
                    task.prompt
                ),
                max_new_tokens=
                    task.max_new_tokens,
            )
            ok = score_p4_output(
                task,
                output,
            )
        except Exception as exc:
            ok = False
            error_type = (
                type(exc).__name__
            )
        if ok:
            passed += 1
        data = output.encode(
            "utf-8"
        )
        results.append(
            {
                "category":
                    task.category,
                "error_type":
                    error_type,
                "output_sha256":
                    _sha256(data),
                "output_utf8_bytes":
                    len(data),
                "passed": ok,
                "task_id":
                    task.task_id,
            }
        )
    return {
        "pass_rate":
            passed
            / len(results),
        "passed_tasks":
            passed,
        "results":
            results,
        "suite_sha256":
            suite.suite_sha256,
        "task_count":
            len(results),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Repair P4C generalization while preserving "
            "the single VN97 architecture."
        )
    )
    parser.add_argument(
        "--p4c-dir",
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
        default=2,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
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
        default=14097,
    )
    parser.add_argument(
        "--p3-replay-records",
        type=int,
        default=
            DEFAULT_P3_REPLAY_RECORDS,
    )
    parser.add_argument(
        "--probe-per-category",
        type=int,
        default=
            DEFAULT_PROBE_PER_CATEGORY,
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
        "--max-p3-validation-loss-increase",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--max-p3-top1-drop",
        type=float,
        default=0.01,
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
        ).startswith("cuda")
    ):
        raise VN97P4DGeneralizationError(
            "P4D requires CUDA"
        )
    if (
        args.sequence_length < 2
        or args.batch_size <= 0
        or args.micro_batch_size <= 0
        or args.batch_size
        % args.micro_batch_size
        != 0
        or args.epochs <= 0
        or not math.isfinite(
            args.learning_rate
        )
        or args.learning_rate <= 0.0
        or args.p3_replay_records < 0
        or args.probe_per_category <= 0
    ):
        raise VN97P4DGeneralizationError(
            "P4D arguments are invalid"
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
        raise VN97P4DGeneralizationError(
            "P4D output-dir must be new or empty"
        )

    parent = verify_p4c_artifact(
        Path(args.p4c_dir)
    )
    tokenizer = VN97Tokenizer(
        parent.tokenizer_package
    )
    model = parent.checkpoint.model

    p3_root = Path(
        args.p3_corpus_dir
    ).resolve(strict=True)
    (
        manifest_id,
        manifest_sha,
    ) = _validate_corpus(
        p3_root
    )
    if (
        manifest_id
        != parent.parent_p3_corpus_manifest_id
        or manifest_sha
        != parent.parent_p3_corpus_manifest_sha256
    ):
        raise VN97P4DGeneralizationError(
            "P3 corpus identity does not match P4C parent"
        )

    p3_training_records, (
        p3_training_sha
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
    p3_validation_records, (
        p3_validation_sha
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

    train_records = (
        default_training()
    )
    validation_records = (
        default_validation()
    )
    train_prompts = {
        item.prompt
        for item in train_records
    }
    validation_prompts = {
        item.prompt
        for item
        in validation_records
    }
    if train_prompts.intersection(
        validation_prompts
    ):
        raise VN97P4DGeneralizationError(
            "P4D training and validation prompts overlap"
        )

    replay_records = _select_replay(
        p3_training_records,
        count=
            args.p3_replay_records,
    )
    replay_prompts = {
        message.content
        for record
        in replay_records
        for message in record
        if message.role == "user"
    }
    if (
        train_prompts.intersection(
            replay_prompts
        )
        or validation_prompts.intersection(
            replay_prompts
        )
    ):
        raise VN97P4DGeneralizationError(
            "P4D prompts overlap P3 replay subset"
        )

    dev_before = None
    if args.dev_suite is not None:
        suite = load_p4_task_suite(
            Path(args.dev_suite)
        )
        dev_prompts = {
            task.prompt
            for task
            in suite.tasks
        }
        if dev_prompts.intersection(
            train_prompts
            | validation_prompts
            | replay_prompts
        ):
            raise VN97P4DGeneralizationError(
                "P4D dev suite overlaps training/replay/validation"
            )
        dev_before = _measure_dev_suite(
            model=model,
            tokenizer=tokenizer,
            suite_path=Path(
                args.dev_suite
            ),
            device=args.device,
        )
        print(
            "VN97 P4D DEV BEFORE "
            f"passed={dev_before['passed_tasks']}/"
            f"{dev_before['task_count']} "
            f"pass_rate={dev_before['pass_rate']:.6f}",
            flush=True,
        )

    probe_records = _probe_records(
        validation_records,
        per_category=
            args.probe_per_category,
    )
    generalization_before = (
        _measure_generalization(
            model=model,
            tokenizer=tokenizer,
            records=probe_records,
            device=args.device,
        )
    )
    print(
        "VN97 P4D GENERALIZATION BEFORE "
        f"passed={generalization_before['passed_tasks']}/"
        f"{generalization_before['task_count']} "
        f"pass_rate={generalization_before['pass_rate']:.6f}",
        flush=True,
    )

    config = VN97TrainingConfig(
        sequence_length=
            args.sequence_length,
        batch_size=
            args.batch_size,
        epochs=args.epochs,
        learning_rate=
            args.learning_rate,
        weight_decay=
            args.weight_decay,
        max_grad_norm=
            args.max_grad_norm,
        seed=args.seed,
        shuffle=True,
        max_windows=100_000,
    )

    train_examples = [
        encode_chat_completion_messages(
            tokenizer,
            item.messages,
        )
        for item in train_records
    ]
    train_examples.extend(
        encode_chat_messages(
            tokenizer,
            record,
        )
        for record
        in replay_records
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
    train_inputs, train_labels = (
        _windows_to_tensors(
            train_windows
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

    validation_before = (
        evaluate_vn97_from_tensors(
            model,
            validation_inputs,
            validation_labels,
            batch_size=
                args.batch_size,
            device=args.device,
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
            batch_size=P3_BATCH_SIZE,
            device=args.device,
            cpu_prefetch_workers=
                args.cpu_prefetch_workers,
            micro_batch_size=2,
        )
    )
    print(
        "VN97 P4D VALIDATION BEFORE "
        f"loss={validation_before.mean_loss:.6f} "
        f"top1={validation_before.top1_accuracy:.6f}",
        flush=True,
    )
    print(
        "VN97 P4D P3 RETENTION BEFORE "
        f"loss={p3_before.mean_loss:.6f} "
        f"top1={p3_before.top1_accuracy:.6f}",
        flush=True,
    )

    train_bytes = curriculum_bytes(
        train_records
    )
    validation_bytes = (
        curriculum_bytes(
            validation_records
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
    resume_identity = _sha256(
        b"VN97P4DRESUME1\0"
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
                "p4c_checkpoint_sha256":
                    parent.checkpoint_sha256,
                "p3_replay_sha256":
                    _sha256(
                        replay_bytes
                    ),
                "profile_id":
                    P4D_PROFILE_ID,
                "seed": args.seed,
                "sequence_length":
                    args.sequence_length,
                "training_sha256":
                    _sha256(
                        train_bytes
                    ),
                "validation_sha256":
                    _sha256(
                        validation_bytes
                    ),
                "weight_decay":
                    args.weight_decay,
            }
        )
    )

    work = Path(
        args.work_dir
    )
    if work.is_symlink():
        raise VN97P4DGeneralizationError(
            "P4D work-dir must not be a symlink"
        )
    work.mkdir(
        parents=True,
        exist_ok=True,
    )
    resume_path = (
        work
        / "state.vn97p3resume1.pt"
    )

    training = train_vn97_from_tensors(
        model,
        train_inputs,
        train_labels,
        config,
        device=args.device,
        cpu_prefetch_workers=
            args.cpu_prefetch_workers,
        micro_batch_size=
            args.micro_batch_size,
        progress_label=
            "generalization",
        progress_interval_steps=
            args.progress_interval_steps,
        resume_checkpoint_path=
            resume_path,
        resume_identity=
            resume_identity,
        checkpoint_interval_steps=
            args.checkpoint_interval_steps,
        progress_protocol=
            "VN97 P4D",
    )

    validation_after = (
        evaluate_vn97_from_tensors(
            model,
            validation_inputs,
            validation_labels,
            batch_size=
                args.batch_size,
            device=args.device,
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
            batch_size=P3_BATCH_SIZE,
            device=args.device,
            cpu_prefetch_workers=
                args.cpu_prefetch_workers,
            micro_batch_size=2,
        )
    )
    generalization_after = (
        _measure_generalization(
            model=model,
            tokenizer=tokenizer,
            records=probe_records,
            device=args.device,
        )
    )
    dev_after = None
    if args.dev_suite is not None:
        dev_after = _measure_dev_suite(
            model=model,
            tokenizer=tokenizer,
            suite_path=Path(
                args.dev_suite
            ),
            device=args.device,
        )

    print(
        "VN97 P4D VALIDATION AFTER "
        f"loss={validation_after.mean_loss:.6f} "
        f"top1={validation_after.top1_accuracy:.6f}",
        flush=True,
    )
    print(
        "VN97 P4D P3 RETENTION AFTER "
        f"loss={p3_after.mean_loss:.6f} "
        f"top1={p3_after.top1_accuracy:.6f}",
        flush=True,
    )
    print(
        "VN97 P4D GENERALIZATION AFTER "
        f"passed={generalization_after['passed_tasks']}/"
        f"{generalization_after['task_count']} "
        f"pass_rate={generalization_after['pass_rate']:.6f}",
        flush=True,
    )
    if dev_after is not None:
        print(
            "VN97 P4D DEV AFTER "
            f"passed={dev_after['passed_tasks']}/"
            f"{dev_after['task_count']} "
            f"pass_rate={dev_after['pass_rate']:.6f}",
            flush=True,
        )

    validation_ok = (
        validation_after.mean_loss
        <= validation_before.mean_loss
        and validation_after.top1_accuracy
        >= validation_before.top1_accuracy
    )
    retention_ok = (
        p3_after.mean_loss
        <= p3_before.mean_loss
        + args.max_p3_validation_loss_increase
        and p3_after.top1_accuracy
        >= p3_before.top1_accuracy
        - args.max_p3_top1_drop
    )
    generalization_ok = (
        generalization_after[
            "passed_tasks"
        ]
        > generalization_before[
            "passed_tasks"
        ]
    )

    if not validation_ok:
        status = (
            "REJECTED_VALIDATION"
        )
    elif not retention_ok:
        status = (
            "REJECTED_RETENTION"
        )
    elif not generalization_ok:
        status = (
            "REJECTED_GENERALIZATION"
        )
    else:
        status = "ELIGIBLE"

    output.mkdir(
        parents=True,
        exist_ok=True,
    )
    tokenizer_bytes = (
        parent.tokenizer_package.to_bytes()
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
    image_bytes = build_model_image(
        model,
        tokenizer=
            parent.tokenizer_package,
        tile_rows=16,
        tile_cols=16,
    ).data
    _atomic_write(
        output
        / "model.vn97mi1",
        image_bytes,
    )

    report = {
        "curriculum": {
            "categories":
                list(P4D_CATEGORIES),
            "training_records":
                len(train_records),
            "training_seed":
                TRAIN_SEED,
            "training_sha256":
                _sha256(train_bytes),
            "validation_records":
                len(
                    validation_records
                ),
            "validation_seed":
                VALIDATION_SEED,
            "validation_sha256":
                _sha256(
                    validation_bytes
                ),
        },
        "dev_after": dev_after,
        "dev_before": dev_before,
        "generalization_after":
            generalization_after,
        "generalization_before":
            generalization_before,
        "model_image_bytes":
            len(image_bytes),
        "model_image_sha256":
            _sha256(image_bytes),
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
        "parent_p4c": {
            "checkpoint_sha256":
                parent.checkpoint_sha256,
            "model_image_sha256":
                parent.model_image_sha256,
            "report_sha256":
                parent.report_sha256,
            "tokenizer_sha256":
                parent.tokenizer_sha256,
        },
        "profile_id":
            P4D_PROFILE_ID,
        "schema":
            P4D_REPORT_SCHEMA,
        "status": status,
        "token_validation_after": {
            "mean_loss":
                validation_after.mean_loss,
            "top1_accuracy":
                validation_after.top1_accuracy,
        },
        "token_validation_before": {
            "mean_loss":
                validation_before.mean_loss,
            "top1_accuracy":
                validation_before.top1_accuracy,
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
        _canonical_json(report)
        + b"\n"
    )
    _atomic_write(
        output
        / "p4d-report.json",
        report_bytes,
    )

    files = {
        "model.vn97ck1":
            checkpoint_path.read_bytes(),
        "model.vn97mi1":
            image_bytes,
        "p4d-report.json":
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
        ).encode("ascii")
        for name in sorted(files)
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
        (
            resume_path.parent
            / "progress.vn97p3resume1.json"
        ).unlink(
            missing_ok=True
        )

    print(
        "VN97P4D1 "
        f"status={status} "
        f"checkpoint={checkpoint_sha} "
        f"generalization_before="
        f"{generalization_before['passed_tasks']}/"
        f"{generalization_before['task_count']} "
        f"generalization_after="
        f"{generalization_after['passed_tasks']}/"
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
            "vn97-p4d-generalization: "
            f"{exc}",
            file=sys.stderr,
        )
        raise
