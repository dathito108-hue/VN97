from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from ..cognition_adapter import TorchVN97InferenceEngine
from ..training import VN97TrainingConfig
from ..training_cli import _atomic_write, _load_records
from .bridge import VN97R2InferenceView, assert_tokenizer_compatible
from .checkpoint import load_r2_checkpoint
from .config import r2_cpu_pilot_config, r2_smoke_config
from .data_bridge import build_completion_windows, load_vn97tk1
from .dense_training import (
    R2DenseTrainingConfig,
    evaluate_dense_loss,
    train_dense,
)
from .model import VN97R2Model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CPU/GPU-agnostic dense-first VN97-R2 pilot over canonical "
            "VN97TK1 + chat JSONL. No quantization or ternary mode is used."
        )
    )
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--validation-jsonl", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--profile",
        choices=("smoke", "pilot"),
        default="smoke",
    )
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--max-windows", type=int, default=20_000)
    parser.add_argument("--max-examples", type=int, default=100_000)
    parser.add_argument(
        "--max-input-bytes",
        type=int,
        default=64 * 1024 * 1024,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=9705)
    return parser


def _combined_identity(
    *,
    train_sha: str,
    validation_sha: str,
    tokenizer_sha: str,
    profile: str,
) -> str:
    payload = json.dumps(
        {
            "train": train_sha,
            "validation": validation_sha,
            "tokenizer": tokenizer_sha,
            "profile": profile,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(
        b"VN97R2PILOTDATA\0" + payload
    ).hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = Path(args.output_dir)
    if output.exists() and (
        output.is_symlink()
        or not output.is_dir()
        or any(output.iterdir())
    ):
        raise ValueError("output-dir must be new or empty")
    output.mkdir(parents=True, exist_ok=True)

    tokenizer_path = Path(args.tokenizer).resolve(strict=True)
    tokenizer = load_vn97tk1(tokenizer_path)
    tokenizer_sha = hashlib.sha256(
        tokenizer_path.read_bytes()
    ).hexdigest()

    training_records, train_sha = _load_records(
        [Path(args.train_jsonl)],
        mode="chat",
        max_input_bytes=args.max_input_bytes,
        max_examples=args.max_examples,
    )
    validation_records, validation_sha = _load_records(
        [Path(args.validation_jsonl)],
        mode="chat",
        max_input_bytes=args.max_input_bytes,
        max_examples=args.max_examples,
    )

    if args.profile == "pilot":
        config = r2_cpu_pilot_config(tokenizer.vocab_size)
    else:
        config = r2_smoke_config(tokenizer.vocab_size)
    model = VN97R2Model(config)
    assert_tokenizer_compatible(model, tokenizer)

    window_config = VN97TrainingConfig(
        sequence_length=args.sequence_length,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=1,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        shuffle=True,
        max_windows=args.max_windows,
    )
    training_windows = build_completion_windows(
        tokenizer,
        training_records,
        window_config,
    )
    validation_windows = build_completion_windows(
        tokenizer,
        validation_records,
        window_config,
    )

    dataset_identity = _combined_identity(
        train_sha=train_sha,
        validation_sha=validation_sha,
        tokenizer_sha=tokenizer_sha,
        profile=args.profile,
    )
    dense_config = R2DenseTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        checkpoint_every_steps=args.checkpoint_every,
    )
    checkpoint_path = output / "model.r2.pt"
    result = train_dense(
        model,
        training_windows,
        validation_windows,
        dense_config,
        work_dir=args.work_dir,
        best_checkpoint_path=checkpoint_path,
        dataset_identity=dataset_identity,
        device=args.device,
    )

    best_model, checkpoint_evidence = load_r2_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )
    final_validation = evaluate_dense_loss(
        best_model,
        validation_windows,
        batch_size=args.batch_size,
        device="cpu",
    )

    first_messages = validation_records[0]
    first_user_text = next(
        (
            message.content
            for message in first_messages
            if message.role == "user"
        ),
        "VN97-R2",
    )
    engine = TorchVN97InferenceEngine(
        VN97R2InferenceView(
            best_model,
            profile="deep",
        ),
        tokenizer,
    )
    embedding = engine.embed_text(
        first_user_text,
        vector_dim=best_model.config.d_model,
    )
    if len(embedding) != best_model.config.d_model:
        raise RuntimeError("R2 cognition bridge embedding size mismatch")

    shutil.copyfile(
        tokenizer_path,
        output / "tokenizer.vn97tk1",
    )
    report = {
        "schema": "VN97R2DENSEPILOT1",
        "status": "PASS",
        "profile": args.profile,
        "architecture_id": best_model.config.architecture_id,
        "config_fingerprint": best_model.config.fingerprint(),
        "parameter_count": best_model.parameter_count(),
        "dataset_identity": dataset_identity,
        "train_sha256": train_sha,
        "validation_sha256": validation_sha,
        "tokenizer_sha256": tokenizer_sha,
        "training_windows": len(training_windows),
        "validation_windows": len(validation_windows),
        "training": {
            "steps": result.steps,
            "target_tokens": result.target_tokens,
            "mean_loss": result.mean_loss,
            "final_loss": result.final_loss,
            "best_epoch": result.best_epoch,
            "best_validation_loss": result.best_validation_loss,
            "best_checkpoint_sha256": (
                result.best_checkpoint_sha256
            ),
        },
        "final_validation": final_validation,
        "checkpoint": checkpoint_evidence,
        "cognition_bridge_embedding_dim": len(embedding),
        "quantization_used": False,
    }
    _atomic_write(
        output / "r2-dense-pilot-report.json",
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8") + b"\n",
    )
    print(
        "VN97R2DENSEPILOT "
        f"status=PASS "
        f"profile={args.profile} "
        f"parameters={best_model.parameter_count()} "
        f"steps={result.steps} "
        f"best_epoch={result.best_epoch} "
        f"best_val_loss={result.best_validation_loss:.6f} "
        f"quantization_used=false "
        f"output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
