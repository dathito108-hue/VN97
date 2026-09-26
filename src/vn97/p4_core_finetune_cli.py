from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import torch

from .deployment_checkpoint import (
    save_deployment_checkpoint,
)
from .model_image import build_model_image
from .p3_tensor_cache import (
    evaluate_vn97_from_tensors,
    train_vn97_from_tensors,
)
from .p4_core_curriculum import (
    P4C_CATEGORIES,
    P4C_PROFILE_ID,
    TRAIN_PER_CATEGORY,
    TRAIN_SEED,
    VALIDATION_PER_CATEGORY,
    VALIDATION_SEED,
    curriculum_jsonl_bytes,
    default_p4c_training,
    default_p4c_validation,
)
from .p4_task_evaluation import (
    load_p4_task_suite,
    render_p4_chat_prompt,
    score_p4_output,
    verify_p3_final_artifact,
)
from .cognition_adapter import (
    TorchVN97InferenceEngine,
    VN97InferenceLimits,
)
from .tokenizer import VN97Tokenizer
from .training import (
    IGNORE_INDEX,
    VN97TrainingConfig,
    build_training_windows,
    encode_chat_messages,
)
from .training_cli import _atomic_write


P4C_REPORT_SCHEMA = "VN97P4C1"


class VN97P4CoreFineTuneError(RuntimeError):
    pass


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _windows_to_tensors(
    windows,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor(
        [
            window.input_ids
            for window in windows
        ],
        dtype=torch.int32,
    )
    labels = torch.tensor(
        [
            window.labels
            for window in windows
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
        raise VN97P4CoreFineTuneError(
            "P4C training tensors are invalid"
        )
    return input_ids, labels


def _curriculum_prompt_set(
    records,
) -> set[str]:
    prompts = {
        record.prompt
        for record in records
    }
    if len(prompts) != len(records):
        raise VN97P4CoreFineTuneError(
            "P4C curriculum prompt identities are not unique"
        )
    return prompts


def _measure_diagnostic(
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
        ),
    )
    results: list[dict[str, object]] = []
    by_category = {
        category: {
            "passed": 0,
            "tasks": 0,
        }
        for category in P4C_CATEGORIES
    }

    model.to(device)
    model.eval()

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
            passed = score_p4_output(
                task,
                output,
            )
        except Exception as exc:
            passed = False
            error_type = type(exc).__name__

        row = by_category[
            task.category
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
                    task.category,
                "error_type":
                    error_type,
                "output_sha256":
                    _sha256(
                        output_bytes
                    ),
                "output_utf8_bytes":
                    len(output_bytes),
                "passed": passed,
                "task_id":
                    task.task_id,
            }
        )

    categories: dict[str, object] = {}
    for category in P4C_CATEGORIES:
        row = by_category[category]
        categories[category] = {
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
        "categories": categories,
        "pass_rate":
            passed / len(results),
        "passed_tasks": passed,
        "results": results,
        "suite_sha256":
            suite.suite_sha256,
        "task_count": len(results),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune the exact P3 VN97 winner on the deterministic "
            "P4 core instruction/cognition curriculum."
        )
    )
    parser.add_argument(
        "--p3-dir",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    parser.add_argument(
        "--work-dir",
        required=True,
    )
    parser.add_argument(
        "--diagnostic-suite",
        default=None,
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
        default=1e-4,
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
        default=4097,
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
    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    args = _parser().parse_args(
        argv
    )
    if not str(args.device).startswith(
        "cuda"
    ):
        raise VN97P4CoreFineTuneError(
            "P4C requires an explicit CUDA device"
        )
    if not torch.cuda.is_available():
        raise VN97P4CoreFineTuneError(
            "CUDA is not available"
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
        or not math.isfinite(
            args.weight_decay
        )
        or args.weight_decay < 0.0
        or not math.isfinite(
            args.max_grad_norm
        )
        or args.max_grad_norm <= 0.0
        or args.seed < 0
        or args.progress_interval_steps
        <= 0
        or args.checkpoint_interval_steps
        <= 0
        or not 0
        <= args.cpu_prefetch_workers
        <= 8
    ):
        raise VN97P4CoreFineTuneError(
            "P4C training arguments are invalid"
        )

    artifact = verify_p3_final_artifact(
        Path(args.p3_dir)
    )
    tokenizer = VN97Tokenizer(
        artifact.tokenizer_package
    )
    model = artifact.checkpoint.model

    training_records = (
        default_p4c_training()
    )
    validation_records = (
        default_p4c_validation()
    )

    training_prompts = (
        _curriculum_prompt_set(
            training_records
        )
    )
    validation_prompts = (
        _curriculum_prompt_set(
            validation_records
        )
    )
    if training_prompts.intersection(
        validation_prompts
    ):
        raise VN97P4CoreFineTuneError(
            "P4C training/validation prompts overlap"
        )

    diagnostic_suite = None
    diagnostic_before = None
    if args.diagnostic_suite is not None:
        diagnostic_suite = (
            load_p4_task_suite(
                Path(
                    args.diagnostic_suite
                )
            )
        )
        diagnostic_prompts = {
            task.prompt
            for task
            in diagnostic_suite.tasks
        }
        if diagnostic_prompts.intersection(
            training_prompts
            | validation_prompts
        ):
            raise VN97P4CoreFineTuneError(
                "P4 diagnostic suite overlaps the P4C curriculum"
            )
        diagnostic_before = (
            _measure_diagnostic(
                model=model,
                tokenizer=tokenizer,
                suite_path=Path(
                    args.diagnostic_suite
                ),
                device=args.device,
            )
        )
        print(
            "VN97 P4C DIAGNOSTIC BEFORE "
            f"passed={diagnostic_before['passed_tasks']}/"
            f"{diagnostic_before['task_count']} "
            f"pass_rate={diagnostic_before['pass_rate']:.6f}",
            flush=True,
        )

    config = VN97TrainingConfig(
        sequence_length=
            args.sequence_length,
        batch_size=args.batch_size,
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
        encode_chat_messages(
            tokenizer,
            record.messages,
        )
        for record in training_records
    ]
    validation_examples = [
        encode_chat_messages(
            tokenizer,
            record.messages,
        )
        for record
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
    validation_windows = (
        build_training_windows(
            validation_examples,
            validation_config,
            pad_token_id=
                tokenizer.pad_id,
        )
    )

    train_input_ids, train_labels = (
        _windows_to_tensors(
            train_windows
        )
    )
    (
        validation_input_ids,
        validation_labels,
    ) = _windows_to_tensors(
        validation_windows
    )

    baseline = (
        evaluate_vn97_from_tensors(
            model,
            validation_input_ids,
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
    print(
        "VN97 P4C VALIDATION BEFORE "
        f"loss={baseline.mean_loss:.6f} "
        f"top1={baseline.top1_accuracy:.6f} "
        f"targets={baseline.target_tokens}",
        flush=True,
    )

    training_bytes = (
        curriculum_jsonl_bytes(
            training_records
        )
    )
    validation_bytes = (
        curriculum_jsonl_bytes(
            validation_records
        )
    )
    training_sha = _sha256(
        training_bytes
    )
    validation_sha = _sha256(
        validation_bytes
    )

    resume_identity = _sha256(
        b"VN97P4CRESUME1\0"
        + _canonical_json(
            {
                "batch_size":
                    args.batch_size,
                "checkpoint_sha256":
                    artifact.checkpoint_sha256,
                "epochs": args.epochs,
                "learning_rate":
                    args.learning_rate,
                "max_grad_norm":
                    args.max_grad_norm,
                "micro_batch_size":
                    args.micro_batch_size,
                "profile_id":
                    P4C_PROFILE_ID,
                "seed": args.seed,
                "sequence_length":
                    args.sequence_length,
                "training_sha256":
                    training_sha,
                "validation_sha256":
                    validation_sha,
                "weight_decay":
                    args.weight_decay,
            }
        )
    )

    work_dir = Path(
        args.work_dir
    )
    if work_dir.is_symlink():
        raise VN97P4CoreFineTuneError(
            "P4C work-dir must not be a symlink"
        )
    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    resume_path = (
        work_dir
        / "state.vn97p3resume1.pt"
    )

    training = train_vn97_from_tensors(
        model,
        train_input_ids,
        train_labels,
        config,
        device=args.device,
        cpu_prefetch_workers=
            args.cpu_prefetch_workers,
        micro_batch_size=
            args.micro_batch_size,
        progress_label="core",
        progress_interval_steps=
            args.progress_interval_steps,
        resume_checkpoint_path=
            resume_path,
        resume_identity=
            resume_identity,
        checkpoint_interval_steps=
            args.checkpoint_interval_steps,
        progress_protocol=
            "VN97 P4C",
    )

    final = evaluate_vn97_from_tensors(
        model,
        validation_input_ids,
        validation_labels,
        batch_size=args.batch_size,
        device=args.device,
        cpu_prefetch_workers=
            args.cpu_prefetch_workers,
        micro_batch_size=
            args.micro_batch_size,
    )
    print(
        "VN97 P4C VALIDATION AFTER "
        f"loss={final.mean_loss:.6f} "
        f"top1={final.top1_accuracy:.6f} "
        f"targets={final.target_tokens}",
        flush=True,
    )

    if (
        final.mean_loss
        > baseline.mean_loss
        or final.top1_accuracy
        < baseline.top1_accuracy
    ):
        raise VN97P4CoreFineTuneError(
            "P4C curriculum validation regressed; candidate is not publishable"
        )

    diagnostic_after = None
    if args.diagnostic_suite is not None:
        diagnostic_after = (
            _measure_diagnostic(
                model=model,
                tokenizer=tokenizer,
                suite_path=Path(
                    args.diagnostic_suite
                ),
                device=args.device,
            )
        )
        print(
            "VN97 P4C DIAGNOSTIC AFTER "
            f"passed={diagnostic_after['passed_tasks']}/"
            f"{diagnostic_after['task_count']} "
            f"pass_rate={diagnostic_after['pass_rate']:.6f}",
            flush=True,
        )

    output = Path(
        args.output_dir
    )
    if (
        output.exists()
        and (
            output.is_symlink()
            or not output.is_dir()
            or any(output.iterdir())
        )
    ):
        raise VN97P4CoreFineTuneError(
            "P4C output-dir must be new or empty"
        )
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer_bytes = (
        artifact.tokenizer_package.to_bytes()
    )
    _atomic_write(
        output / "tokenizer.vn97tk1",
        tokenizer_bytes,
    )
    checkpoint_path = (
        output / "model.vn97ck1"
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
            artifact.tokenizer_package,
        tile_rows=16,
        tile_cols=16,
    ).data
    _atomic_write(
        output / "model.vn97mi1",
        image_bytes,
    )

    report = {
        "baseline_validation": {
            "mean_loss":
                baseline.mean_loss,
            "target_tokens":
                baseline.target_tokens,
            "top1_accuracy":
                baseline.top1_accuracy,
            "windows":
                baseline.windows,
        },
        "curriculum": {
            "categories":
                list(P4C_CATEGORIES),
            "training_records":
                len(training_records),
            "training_seed":
                TRAIN_SEED,
            "training_sha256":
                training_sha,
            "validation_records":
                len(
                    validation_records
                ),
            "validation_seed":
                VALIDATION_SEED,
            "validation_sha256":
                validation_sha,
        },
        "diagnostic_after":
            diagnostic_after,
        "diagnostic_before":
            diagnostic_before,
        "final_validation": {
            "mean_loss":
                final.mean_loss,
            "target_tokens":
                final.target_tokens,
            "top1_accuracy":
                final.top1_accuracy,
            "windows":
                final.windows,
        },
        "model_image_bytes":
            len(image_bytes),
        "model_image_sha256":
            _sha256(image_bytes),
        "output_checkpoint_sha256":
            checkpoint_sha,
        "parent_p3": {
            "checkpoint_sha256":
                artifact.checkpoint_sha256,
            "model_image_sha256":
                artifact.model_image_sha256,
            "p3_run_sha256":
                artifact.p3_run_sha256,
            "selected_candidate_id":
                artifact.selected_candidate_id,
            "tokenizer_sha256":
                artifact.tokenizer_sha256,
        },
        "profile_id":
            P4C_PROFILE_ID,
        "schema":
            P4C_REPORT_SCHEMA,
        "status":
            "ELIGIBLE",
        "tokenizer_sha256":
            _sha256(tokenizer_bytes),
        "training": {
            "batch_size":
                args.batch_size,
            "epochs":
                args.epochs,
            "final_loss":
                training.final_loss,
            "learning_rate":
                args.learning_rate,
            "mean_loss":
                training.mean_loss,
            "micro_batch_size":
                args.micro_batch_size,
            "seed": args.seed,
            "sequence_length":
                args.sequence_length,
            "steps":
                training.steps,
            "target_tokens":
                training.target_tokens,
            "weight_decay":
                args.weight_decay,
            "windows":
                len(train_windows),
        },
    }
    report_bytes = (
        _canonical_json(report)
        + b"\n"
    )
    _atomic_write(
        output / "p4c-report.json",
        report_bytes,
    )

    files = {
        "model.vn97ck1":
            checkpoint_path.read_bytes(),
        "model.vn97mi1":
            image_bytes,
        "p4c-report.json":
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
        "VN97P4C1 "
        f"status=ELIGIBLE "
        f"checkpoint={checkpoint_sha} "
        f"validation_loss={final.mean_loss:.6f} "
        f"validation_top1={final.top1_accuracy:.6f}",
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
            "vn97-p4-core-finetune: "
            f"{exc}",
            file=sys.stderr,
        )
        raise
