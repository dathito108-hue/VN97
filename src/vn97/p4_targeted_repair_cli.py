from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

from .chat_boundary import recover_chat_response_text
from .cognition_adapter import (
    TorchVN97InferenceEngine,
    VN97InferenceLimits,
)
from .deployment_checkpoint import save_deployment_checkpoint
from .model_image import build_model_image
from .p3_language_campaign import (
    BATCH_SIZE as P3_BATCH_SIZE,
    MAX_VALIDATION_WINDOWS as P3_MAX_VALIDATION_WINDOWS,
    SEQUENCE_LENGTH as P3_SEQUENCE_LENGTH,
)
from .p3_language_campaign_cli import _validate_corpus
from .p3_tensor_cache import (
    evaluate_vn97_from_tensors,
    train_vn97_from_tensors,
)
from .p4_artifact import verify_p4d_artifact
from .p4_generalization_cli import (
    _canonical_json,
    _probe_records,
    _record_score,
    _select_replay,
    _sha256,
    _windows_to_tensors,
)
from .p4_generalization_curriculum import (
    P4D_CATEGORIES,
    curriculum_bytes,
    default_validation,
)
from .p4_targeted_repair_curriculum import (
    P4E_C_PROFILE_ID,
    P4E_C_TRAIN_PER_CATEGORY,
    P4E_C_TRAIN_SEED,
    targeted_training,
)
from .p4_task_evaluation import (
    load_p4_task_suite,
    render_p4_chat_prompt,
    score_p4_output,
)
from .tokenizer import VN97Tokenizer
from .training import (
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


P4E_C_REPORT_SCHEMA = "VN97P4EC1"
DEFAULT_P3_REPLAY_RECORDS = 1000
DEFAULT_PROBE_PER_CATEGORY = 30


class VN97P4ECTargetedRepairError(RuntimeError):
    pass


def _measure_records(
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

    raw_passed = 0
    recovered_passed = 0
    raw_failures: Counter[str] = Counter()
    recovered_failures: Counter[str] = Counter()
    results: list[dict[str, object]] = []

    for record in records:
        raw = ""
        recovered = ""
        error_type = ""
        try:
            raw = engine.generate_text(
                render_p4_chat_prompt(record.prompt),
                max_new_tokens=96,
            )
            recovered = recover_chat_response_text(raw)
            raw_ok = _record_score(
                record.answer,
                raw,
            )
            recovered_ok = _record_score(
                record.answer,
                recovered,
            )
        except Exception as exc:
            raw_ok = False
            recovered_ok = False
            error_type = type(exc).__name__

        if raw_ok:
            raw_passed += 1
            raw_failures["pass"] += 1
        else:
            raw_failures["fail"] += 1
        if recovered_ok:
            recovered_passed += 1
            recovered_failures["pass"] += 1
        else:
            recovered_failures["fail"] += 1

        results.append(
            {
                "category": record.category,
                "error_type": error_type,
                "prompt_sha256": _sha256(
                    record.prompt.encode("utf-8")
                ),
                "raw_output_sha256": _sha256(
                    raw.encode("utf-8")
                ),
                "raw_passed": raw_ok,
                "recovered_output_sha256": _sha256(
                    recovered.encode("utf-8")
                ),
                "recovered_passed": recovered_ok,
            }
        )

    tasks = len(results)
    return {
        "raw_passed": raw_passed,
        "raw_pass_rate": raw_passed / tasks,
        "recovered_passed": recovered_passed,
        "recovered_pass_rate": recovered_passed / tasks,
        "task_count": tasks,
        "results": results,
        "raw_counts": dict(sorted(raw_failures.items())),
        "recovered_counts": dict(
            sorted(recovered_failures.items())
        ),
    }


def _measure_dev(
    *,
    model,
    tokenizer,
    suite_path: Path,
    device: str,
) -> dict[str, object]:
    suite = load_p4_task_suite(suite_path)
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

    raw_passed = 0
    recovered_passed = 0
    results: list[dict[str, object]] = []

    for task in suite.tasks:
        raw = ""
        recovered = ""
        error_type = ""
        try:
            raw = engine.generate_text(
                render_p4_chat_prompt(task.prompt),
                max_new_tokens=task.max_new_tokens,
            )
            recovered = recover_chat_response_text(raw)
            raw_ok = score_p4_output(task, raw)
            recovered_ok = score_p4_output(
                task,
                recovered,
            )
        except Exception as exc:
            raw_ok = False
            recovered_ok = False
            error_type = type(exc).__name__

        raw_passed += int(raw_ok)
        recovered_passed += int(recovered_ok)
        results.append(
            {
                "category": task.category,
                "error_type": error_type,
                "raw_output_sha256": _sha256(
                    raw.encode("utf-8")
                ),
                "raw_passed": raw_ok,
                "recovered_output_sha256": _sha256(
                    recovered.encode("utf-8")
                ),
                "recovered_passed": recovered_ok,
                "task_id": task.task_id,
            }
        )

    tasks = len(results)
    return {
        "raw_passed": raw_passed,
        "raw_pass_rate": raw_passed / tasks,
        "recovered_passed": recovered_passed,
        "recovered_pass_rate": recovered_passed / tasks,
        "task_count": tasks,
        "results": results,
        "suite_sha256": suite.suite_sha256,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Targeted post-P4D generation repair using completion-aligned "
            "training for both new repair data and P3 replay."
        )
    )
    parser.add_argument("--p4d-dir", required=True)
    parser.add_argument("--p3-corpus-dir", required=True)
    parser.add_argument("--dev-suite", default=None)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
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
        default=2e-5,
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
        default=51997,
    )
    parser.add_argument(
        "--p3-replay-records",
        type=int,
        default=DEFAULT_P3_REPLAY_RECORDS,
    )
    parser.add_argument(
        "--probe-per-category",
        type=int,
        default=DEFAULT_PROBE_PER_CATEGORY,
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if (
        not torch.cuda.is_available()
        or not str(args.device).startswith("cuda")
    ):
        raise VN97P4ECTargetedRepairError(
            "P4E-C requires CUDA"
        )
    if (
        args.sequence_length < 2
        or args.batch_size <= 0
        or args.micro_batch_size <= 0
        or args.batch_size % args.micro_batch_size != 0
        or args.epochs <= 0
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0.0
        or args.p3_replay_records < 0
        or args.probe_per_category <= 0
    ):
        raise VN97P4ECTargetedRepairError(
            "P4E-C arguments are invalid"
        )

    output = Path(args.output_dir)
    if (
        output.exists()
        and (
            output.is_symlink()
            or not output.is_dir()
            or any(output.iterdir())
        )
    ):
        raise VN97P4ECTargetedRepairError(
            "P4E-C output-dir must be new or empty"
        )

    parent = verify_p4d_artifact(
        Path(args.p4d_dir)
    )
    tokenizer = VN97Tokenizer(
        parent.tokenizer_package
    )
    model = parent.checkpoint.model

    p3_root = Path(
        args.p3_corpus_dir
    ).resolve(strict=True)
    _validate_corpus(p3_root)

    p3_training_records, p3_training_sha = (
        _load_records(
            [p3_root / "training.jsonl"],
            mode="chat",
            max_input_bytes=64 * 1024 * 1024,
            max_examples=100_000,
        )
    )
    p3_validation_records, p3_validation_sha = (
        _load_records(
            [p3_root / "validation.jsonl"],
            mode="chat",
            max_input_bytes=64 * 1024 * 1024,
            max_examples=100_000,
        )
    )

    retention = parent.report.get(
        "p3_retention"
    )
    if not isinstance(retention, dict):
        raise VN97P4ECTargetedRepairError(
            "P4D report lacks P3 retention identity"
        )
    if (
        retention.get("training_dataset_sha256")
        != p3_training_sha
        or retention.get("validation_dataset_sha256")
        != p3_validation_sha
    ):
        raise VN97P4ECTargetedRepairError(
            "P3 corpus does not match the measured P4D parent"
        )

    train_records = targeted_training()
    validation_records = default_validation()
    train_prompts = {
        item.prompt
        for item in train_records
    }
    validation_prompts = {
        item.prompt
        for item in validation_records
    }
    if train_prompts.intersection(
        validation_prompts
    ):
        raise VN97P4ECTargetedRepairError(
            "P4E-C train/validation prompts overlap"
        )

    replay_records = _select_replay(
        p3_training_records,
        count=args.p3_replay_records,
    )
    replay_prompts = {
        message.content
        for record in replay_records
        for message in record
        if message.role == "user"
    }
    if (
        train_prompts.intersection(replay_prompts)
        or validation_prompts.intersection(
            replay_prompts
        )
    ):
        raise VN97P4ECTargetedRepairError(
            "P4E-C prompts overlap P3 replay"
        )

    probe_records = _probe_records(
        validation_records,
        per_category=args.probe_per_category,
    )

    dev_before = None
    if args.dev_suite is not None:
        suite = load_p4_task_suite(
            Path(args.dev_suite)
        )
        dev_prompts = {
            task.prompt
            for task in suite.tasks
        }
        if dev_prompts.intersection(
            train_prompts
            | validation_prompts
            | replay_prompts
        ):
            raise VN97P4ECTargetedRepairError(
                "P4E-C dev suite overlaps train/validation/replay"
            )
        dev_before = _measure_dev(
            model=model,
            tokenizer=tokenizer,
            suite_path=Path(args.dev_suite),
            device=args.device,
        )
        print(
            "VN97 P4E-C DEV BEFORE "
            f"raw={dev_before['raw_passed']}/"
            f"{dev_before['task_count']} "
            f"recovered={dev_before['recovered_passed']}/"
            f"{dev_before['task_count']}",
            flush=True,
        )

    generation_before = _measure_records(
        model=model,
        tokenizer=tokenizer,
        records=probe_records,
        device=args.device,
    )
    print(
        "VN97 P4E-C GENERALIZATION BEFORE "
        f"raw={generation_before['raw_passed']}/"
        f"{generation_before['task_count']} "
        f"recovered={generation_before['recovered_passed']}/"
        f"{generation_before['task_count']}",
        flush=True,
    )

    config = VN97TrainingConfig(
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        shuffle=True,
        max_windows=100_000,
    )
    validation_config = VN97TrainingConfig(
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        epochs=1,
        seed=0,
        shuffle=False,
        max_windows=100_000,
    )

    # Critical P4E-C change: replay is completion-aligned too.
    # This stops reinforcing the historical textual assistant marker target.
    train_examples = [
        encode_chat_completion_messages(
            tokenizer,
            item.messages,
        )
        for item in train_records
    ]
    train_examples.extend(
        encode_chat_completion_messages(
            tokenizer,
            record,
        )
        for record in replay_records
    )
    validation_examples = [
        encode_chat_completion_messages(
            tokenizer,
            item.messages,
        )
        for item in validation_records
    ]

    train_windows = build_training_windows(
        train_examples,
        config,
        pad_token_id=tokenizer.pad_id,
    )
    validation_windows = build_training_windows(
        validation_examples,
        validation_config,
        pad_token_id=tokenizer.pad_id,
    )
    train_inputs, train_labels = (
        _windows_to_tensors(train_windows)
    )
    validation_inputs, validation_labels = (
        _windows_to_tensors(
            validation_windows
        )
    )

    p3_validation_config = VN97TrainingConfig(
        sequence_length=P3_SEQUENCE_LENGTH,
        batch_size=P3_BATCH_SIZE,
        epochs=1,
        seed=0,
        shuffle=False,
        max_windows=P3_MAX_VALIDATION_WINDOWS,
    )
    p3_validation_examples = [
        encode_chat_messages(
            tokenizer,
            record,
        )
        for record in p3_validation_records
    ]
    p3_validation_windows = build_training_windows(
        p3_validation_examples,
        p3_validation_config,
        pad_token_id=tokenizer.pad_id,
    )
    p3_inputs, p3_labels = _windows_to_tensors(
        p3_validation_windows
    )

    validation_before = evaluate_vn97_from_tensors(
        model,
        validation_inputs,
        validation_labels,
        batch_size=args.batch_size,
        device=args.device,
        cpu_prefetch_workers=
            args.cpu_prefetch_workers,
        micro_batch_size=args.micro_batch_size,
    )
    p3_before = evaluate_vn97_from_tensors(
        model,
        p3_inputs,
        p3_labels,
        batch_size=P3_BATCH_SIZE,
        device=args.device,
        cpu_prefetch_workers=
            args.cpu_prefetch_workers,
        micro_batch_size=2,
    )

    print(
        "VN97 P4E-C TOKEN BEFORE "
        f"loss={validation_before.mean_loss:.6f} "
        f"top1={validation_before.top1_accuracy:.6f}",
        flush=True,
    )
    print(
        "VN97 P4E-C P3 RETENTION BEFORE "
        f"loss={p3_before.mean_loss:.6f} "
        f"top1={p3_before.top1_accuracy:.6f}",
        flush=True,
    )

    train_bytes = curriculum_bytes(
        train_records
    )
    replay_bytes = b"".join(
        (
            _canonical_json(
                {
                    "messages": [
                        {
                            "content": message.content,
                            "role": message.role,
                        }
                        for message in record
                    ]
                }
            )
            + b"\n"
        )
        for record in replay_records
    )
    resume_identity = _sha256(
        b"VN97P4ECRESUME1\0"
        + _canonical_json(
            {
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate":
                    args.learning_rate,
                "micro_batch_size":
                    args.micro_batch_size,
                "p3_replay_sha256":
                    _sha256(replay_bytes),
                "parent_p4d_checkpoint_sha256":
                    parent.checkpoint_sha256,
                "profile_id":
                    P4E_C_PROFILE_ID,
                "seed": args.seed,
                "sequence_length":
                    args.sequence_length,
                "training_sha256":
                    _sha256(train_bytes),
                "weight_decay":
                    args.weight_decay,
            }
        )
    )

    work = Path(args.work_dir)
    if work.is_symlink():
        raise VN97P4ECTargetedRepairError(
            "P4E-C work-dir must not be a symlink"
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
        micro_batch_size=args.micro_batch_size,
        progress_label=
            "targeted_generation_repair",
        progress_interval_steps=
            args.progress_interval_steps,
        resume_checkpoint_path=resume_path,
        resume_identity=resume_identity,
        checkpoint_interval_steps=
            args.checkpoint_interval_steps,
        progress_protocol="VN97 P4E-C",
    )

    validation_after = evaluate_vn97_from_tensors(
        model,
        validation_inputs,
        validation_labels,
        batch_size=args.batch_size,
        device=args.device,
        cpu_prefetch_workers=
            args.cpu_prefetch_workers,
        micro_batch_size=args.micro_batch_size,
    )
    p3_after = evaluate_vn97_from_tensors(
        model,
        p3_inputs,
        p3_labels,
        batch_size=P3_BATCH_SIZE,
        device=args.device,
        cpu_prefetch_workers=
            args.cpu_prefetch_workers,
        micro_batch_size=2,
    )
    generation_after = _measure_records(
        model=model,
        tokenizer=tokenizer,
        records=probe_records,
        device=args.device,
    )
    dev_after = None
    if args.dev_suite is not None:
        dev_after = _measure_dev(
            model=model,
            tokenizer=tokenizer,
            suite_path=Path(args.dev_suite),
            device=args.device,
        )

    print(
        "VN97 P4E-C TOKEN AFTER "
        f"loss={validation_after.mean_loss:.6f} "
        f"top1={validation_after.top1_accuracy:.6f}",
        flush=True,
    )
    print(
        "VN97 P4E-C P3 RETENTION AFTER "
        f"loss={p3_after.mean_loss:.6f} "
        f"top1={p3_after.top1_accuracy:.6f}",
        flush=True,
    )
    print(
        "VN97 P4E-C GENERALIZATION AFTER "
        f"raw={generation_after['raw_passed']}/"
        f"{generation_after['task_count']} "
        f"recovered={generation_after['recovered_passed']}/"
        f"{generation_after['task_count']}",
        flush=True,
    )
    if dev_after is not None:
        print(
            "VN97 P4E-C DEV AFTER "
            f"raw={dev_after['raw_passed']}/"
            f"{dev_after['task_count']} "
            f"recovered={dev_after['recovered_passed']}/"
            f"{dev_after['task_count']}",
            flush=True,
        )

    validation_ok = (
        validation_after.mean_loss
        <= validation_before.mean_loss
        + args.max_token_validation_loss_increase
        and validation_after.top1_accuracy
        >= validation_before.top1_accuracy
        - args.max_token_top1_drop
    )
    retention_ok = (
        p3_after.mean_loss
        <= p3_before.mean_loss
        + args.max_p3_validation_loss_increase
        and p3_after.top1_accuracy
        >= p3_before.top1_accuracy
        - args.max_p3_top1_drop
    )
    raw_gain = (
        generation_after["raw_passed"]
        > generation_before["raw_passed"]
    )
    recovered_non_regression = (
        generation_after["recovered_passed"]
        >= generation_before[
            "recovered_passed"
        ]
    )

    if not validation_ok:
        status = "REJECTED_VALIDATION"
    elif not retention_ok:
        status = "REJECTED_RETENTION"
    elif not raw_gain:
        status = "REJECTED_RAW_GENERALIZATION"
    elif not recovered_non_regression:
        status = "REJECTED_BOUNDARY_REGRESSION"
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
        output / "tokenizer.vn97tk1",
        tokenizer_bytes,
    )
    checkpoint_path = (
        output / "model.vn97ck1"
    )
    checkpoint_sha = save_deployment_checkpoint(
        model,
        checkpoint_path,
    )
    image_bytes = build_model_image(
        model,
        tokenizer=parent.tokenizer_package,
        tile_rows=16,
        tile_cols=16,
    ).data
    _atomic_write(
        output / "model.vn97mi1",
        image_bytes,
    )

    report = {
        "dev_after": dev_after,
        "dev_before": dev_before,
        "generation_after":
            generation_after,
        "generation_before":
            generation_before,
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
        "parent_p4d": {
            "checkpoint_sha256":
                parent.checkpoint_sha256,
            "report_sha256":
                parent.report_sha256,
            "status":
                parent.status,
        },
        "profile_id":
            P4E_C_PROFILE_ID,
        "repair_curriculum": {
            "categories":
                list(P4D_CATEGORIES),
            "completion_aligned_p3_replay":
                True,
            "p3_replay_records":
                len(replay_records),
            "training_records":
                len(train_records),
            "training_seed":
                P4E_C_TRAIN_SEED,
            "training_sha256":
                _sha256(train_bytes),
            "training_per_category":
                P4E_C_TRAIN_PER_CATEGORY,
        },
        "schema":
            P4E_C_REPORT_SCHEMA,
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
            _sha256(tokenizer_bytes),
        "training": {
            "epochs": args.epochs,
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
        output / "p4e-report.json",
        report_bytes,
    )

    files = {
        "model.vn97ck1":
            checkpoint_path.read_bytes(),
        "model.vn97mi1":
            image_bytes,
        "p4e-report.json":
            report_bytes,
        "tokenizer.vn97tk1":
            tokenizer_bytes,
    }
    sums = b"".join(
        (
            _sha256(files[name])
            + "  "
            + name
            + "\n"
        ).encode("ascii")
        for name in sorted(files)
    )
    _atomic_write(
        output / "SHA256SUMS",
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
        "VN97P4EC1 "
        f"status={status} "
        f"checkpoint={checkpoint_sha} "
        f"raw_before={generation_before['raw_passed']}/"
        f"{generation_before['task_count']} "
        f"raw_after={generation_after['raw_passed']}/"
        f"{generation_after['task_count']} "
        f"recovered_before="
        f"{generation_before['recovered_passed']}/"
        f"{generation_before['task_count']} "
        f"recovered_after="
        f"{generation_after['recovered_passed']}/"
        f"{generation_after['task_count']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            "vn97-p4e-targeted-repair: "
            f"{exc}",
            file=sys.stderr,
        )
        raise
