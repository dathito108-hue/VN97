from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

from .continual_training import guarded_fine_tune_vn97
from .deployment_checkpoint import (
    load_deployment_checkpoint_file,
    save_deployment_checkpoint,
)
from .evaluation import VN97ReleaseCriteria
from .tokenizer import VN97Tokenizer, VN97TokenizerPackage
from .training import (
    VN97TrainingConfig,
    build_training_windows,
    encode_causal_text,
    encode_chat_messages,
)
from .training_cli import (
    _atomic_write,
    _canonical_json,
    _load_records,
    _read_bounded_regular_file,
)


_MAX_TOKENIZER_BYTES = 64 * 1024 * 1024


def _file_identities(
    paths: list[Path],
    *,
    label: str,
) -> set[tuple[int, int]]:
    identities: set[tuple[int, int]] = set()
    for path in paths:
        try:
            info = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise ValueError(f"{label} path is unavailable: {path}") from exc
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{label} path must be a regular non-symlink file: {path}")
        identity = (int(info.st_dev), int(info.st_ino))
        if identity in identities:
            raise ValueError(f"{label} contains the same file more than once")
        identities.add(identity)
    return identities


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Guarded continual fine-tuning of an existing canonical VN97CK1 checkpoint."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--validation-input", action="append", required=True)
    parser.add_argument("--format", choices=("text", "chat"), default="chat")
    parser.add_argument(
        "--validation-format",
        choices=("text", "chat"),
        default=None,
    )
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--max-input-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-examples", type=int, default=100_000)
    parser.add_argument("--max-windows", type=int, default=10_000)
    parser.add_argument(
        "--validation-max-input-bytes",
        type=int,
        default=64 * 1024 * 1024,
    )
    parser.add_argument(
        "--validation-max-examples",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--validation-max-windows",
        type=int,
        default=10_000,
    )

    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--validation-batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=97)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--max-validation-loss", type=float, required=True)
    parser.add_argument(
        "--max-validation-loss-increase",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--min-validation-accuracy",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--min-validation-target-tokens",
        type=int,
        default=64,
    )
    return parser


def _examples(
    records: list[object],
    *,
    mode: str,
    tokenizer: VN97Tokenizer,
):
    if mode == "text":
        return [
            encode_causal_text(tokenizer, value)
            for value in records
        ]
    return [
        encode_chat_messages(tokenizer, value)
        for value in records
    ]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (
        args.max_input_bytes <= 0
        or args.max_examples <= 0
        or args.max_windows <= 0
        or args.validation_max_input_bytes <= 0
        or args.validation_max_examples <= 0
        or args.validation_max_windows <= 0
        or args.validation_batch_size <= 0
    ):
        raise ValueError("training/validation bounds must be positive")

    train_paths = [Path(value) for value in args.input]
    validation_paths = [Path(value) for value in args.validation_input]
    train_ids = _file_identities(train_paths, label="training input")
    validation_ids = _file_identities(
        validation_paths,
        label="validation input",
    )
    if train_ids.intersection(validation_ids):
        raise ValueError(
            "training and validation inputs must be physically distinct files"
        )

    loaded = load_deployment_checkpoint_file(Path(args.checkpoint))
    tokenizer_bytes = _read_bounded_regular_file(
        Path(args.tokenizer),
        max_bytes=_MAX_TOKENIZER_BYTES,
    )
    try:
        package = VN97TokenizerPackage.from_bytes(tokenizer_bytes)
    except ValueError as exc:
        raise ValueError("VN97TK1 tokenizer is invalid") from exc
    if package.vocab_size != loaded.config.vocab_size:
        raise ValueError(
            "VN97TK1 tokenizer vocabulary does not match VN97CK1 checkpoint"
        )
    tokenizer = VN97Tokenizer(package)

    train_records, train_sha256 = _load_records(
        train_paths,
        mode=args.format,
        max_input_bytes=args.max_input_bytes,
        max_examples=args.max_examples,
    )
    validation_mode = args.validation_format or args.format
    validation_records, validation_sha256 = _load_records(
        validation_paths,
        mode=validation_mode,
        max_input_bytes=args.validation_max_input_bytes,
        max_examples=args.validation_max_examples,
    )

    training_config = VN97TrainingConfig(
        sequence_length=args.sequence_length,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        shuffle=True,
        max_windows=args.max_windows,
    )
    train_windows = build_training_windows(
        _examples(
            train_records,
            mode=args.format,
            tokenizer=tokenizer,
        ),
        training_config,
        pad_token_id=tokenizer.pad_id,
    )
    validation_config = VN97TrainingConfig(
        sequence_length=args.sequence_length,
        stride=args.stride,
        batch_size=args.validation_batch_size,
        epochs=1,
        seed=0,
        shuffle=False,
        max_windows=args.validation_max_windows,
    )
    validation_examples = _examples(
        validation_records,
        mode=validation_mode,
        tokenizer=tokenizer,
    )
    validation_windows = build_training_windows(
        validation_examples,
        validation_config,
        pad_token_id=tokenizer.pad_id,
    )

    criteria = VN97ReleaseCriteria(
        max_validation_loss=args.max_validation_loss,
        min_top1_accuracy=args.min_validation_accuracy,
        min_target_tokens=args.min_validation_target_tokens,
    )
    result = guarded_fine_tune_vn97(
        loaded.model,
        train_windows,
        validation_windows,
        training_config,
        criteria,
        max_validation_loss_increase=args.max_validation_loss_increase,
        validation_batch_size=args.validation_batch_size,
        device=args.device,
    )

    # Nothing is written until both absolute and regression quality gates pass.
    output = Path(args.output_dir)
    if output.is_symlink():
        raise ValueError("output-dir must not be a symlink")
    output.mkdir(parents=True, exist_ok=True)

    tokenizer_path = output / "tokenizer.vn97tk1"
    _atomic_write(tokenizer_path, tokenizer_bytes)
    checkpoint_path = output / "model.vn97ck1"
    checkpoint_sha256 = save_deployment_checkpoint(
        loaded.model,
        checkpoint_path,
    )

    report = {
        "baseline_validation": {
            "mean_loss": result.baseline.mean_loss,
            "target_tokens": result.baseline.target_tokens,
            "top1_accuracy": result.baseline.top1_accuracy,
            "windows": result.baseline.windows,
        },
        "final_validation": {
            "mean_loss": result.final.mean_loss,
            "target_tokens": result.final.target_tokens,
            "top1_accuracy": result.final.top1_accuracy,
            "windows": result.final.windows,
        },
        "output_checkpoint_sha256": checkpoint_sha256,
        "parent_checkpoint_sha256": loaded.checkpoint_sha256,
        "schema": "VN97FT1",
        "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
        "training": {
            "batch_size": training_config.batch_size,
            "dataset_sha256": train_sha256,
            "epochs": training_config.epochs,
            "final_loss": result.training.final_loss,
            "learning_rate": training_config.learning_rate,
            "mean_loss": result.training.mean_loss,
            "max_grad_norm": training_config.max_grad_norm,
            "seed": training_config.seed,
            "steps": result.training.steps,
            "target_tokens": result.training.target_tokens,
            "windows": len(train_windows),
        },
        "validation": {
            "dataset_sha256": validation_sha256,
            "examples": len(validation_examples),
            "max_loss": criteria.max_validation_loss,
            "max_loss_increase": args.max_validation_loss_increase,
            "min_target_tokens": criteria.min_target_tokens,
            "min_top1_accuracy": criteria.min_top1_accuracy,
        },
    }
    _atomic_write(
        output / "finetune-report.json",
        _canonical_json(report),
    )

    print(f"VN97FT1 checkpoint={checkpoint_path}")
    print(f"checkpoint_sha256={checkpoint_sha256}")
    print(f"baseline_validation_loss={result.baseline.mean_loss:.6f}")
    print(f"final_validation_loss={result.final.mean_loss:.6f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-finetune: {exc}", file=sys.stderr)
        raise
