from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Sequence

import torch

from ..training import VN97TrainingConfig
from ..training_cli import _atomic_write, _load_records
from .config import r2_mobile_1b_config
from .data_bridge import (
    assert_r2_data_compatible,
    build_completion_windows,
    load_vn97tk1,
)
from .production_contract import (
    R2ProductionCorpusManifest,
    R2ProductionShard,
    R2ProductionTrainingRecipe,
)
from .pilot_contract import build_pilot_corpus_evidence
from .production_training import (
    R2MeasuredMemoryEvidence,
    R2ProductionTrainerConfig,
    load_production_stage_model,
    measure_cuda_training_preflight,
    train_production_stage,
)


R2D4_PREFLIGHT_SCHEMA = "VN97R2D4PREFLIGHT1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _regular_file_size(path: Path) -> int:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"expected regular file: {path}")
    return int(resolved.stat().st_size)


def build_production_manifest(
    *,
    stage: str,
    tokenizer_path: Path,
    train_path: Path,
    validation_path: Path,
    train_records: int,
    validation_records: int,
    task_families: Sequence[str],
    parent_checkpoint_sha256: str | None,
) -> R2ProductionCorpusManifest:
    families = tuple(sorted(set(task_families)))
    if not families or any(not item for item in families):
        raise ValueError("at least one non-empty task family is required")
    return R2ProductionCorpusManifest(
        stage=stage,
        tokenizer_sha256=_sha256_file(tokenizer_path),
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        shards=(
            R2ProductionShard(
                split="train",
                sha256=_sha256_file(train_path),
                bytes=_regular_file_size(train_path),
                records=train_records,
                task_families=families,
            ),
            R2ProductionShard(
                split="validation",
                sha256=_sha256_file(validation_path),
                bytes=_regular_file_size(validation_path),
                records=validation_records,
                task_families=families,
            ),
        ),
    )


def save_preflight_bundle(
    path: Path,
    *,
    manifest: R2ProductionCorpusManifest,
    recipe: R2ProductionTrainingRecipe,
    evidence: R2MeasuredMemoryEvidence,
) -> None:
    payload = {
        "schema": R2D4_PREFLIGHT_SCHEMA,
        "manifest_identity": manifest.identity(),
        "recipe_fingerprint": recipe.fingerprint(),
        "manifest": {
            "stage": manifest.stage,
            "tokenizer_sha256": manifest.tokenizer_sha256,
            "parent_checkpoint_sha256": manifest.parent_checkpoint_sha256,
            "shards": [asdict(item) for item in manifest.shards],
        },
        "recipe": asdict(recipe),
        "memory_evidence": evidence.as_dict(),
        "promotion_allowed": False,
    }
    _atomic_write(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8") + b"\n",
    )


def load_preflight_bundle(
    path: Path,
    *,
    manifest: R2ProductionCorpusManifest,
    recipe: R2ProductionTrainingRecipe,
) -> R2MeasuredMemoryEvidence:
    raw = path.resolve(strict=True).read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("R2-D4 preflight bundle is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("R2-D4 preflight bundle must be an object")
    if payload.get("schema") != R2D4_PREFLIGHT_SCHEMA:
        raise RuntimeError("R2-D4 preflight schema mismatch")
    if payload.get("promotion_allowed") is not False:
        raise RuntimeError("R2-D4 preflight must be no-promotion evidence")
    if payload.get("manifest_identity") != manifest.identity():
        raise RuntimeError("R2-D4 preflight manifest identity mismatch")
    if payload.get("recipe_fingerprint") != recipe.fingerprint():
        raise RuntimeError("R2-D4 preflight recipe fingerprint mismatch")
    evidence = payload.get("memory_evidence")
    if not isinstance(evidence, dict):
        raise RuntimeError("R2-D4 measured memory evidence missing")
    return R2MeasuredMemoryEvidence(**evidence)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "VN97-R2D4 production corpus launcher. Performs exact 1B "
            "measured CUDA preflight before any production training."
        )
    )
    parser.add_argument("--mode", choices=("preflight", "train"), required=True)
    parser.add_argument(
        "--stage",
        choices=(
            "dense_pretrain",
            "instruction_reasoning",
            "tool_action",
            "capability",
        ),
        required=True,
    )
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--validation-jsonl", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parent-checkpoint", default=None)
    parser.add_argument(
        "--task-family",
        action="append",
        required=True,
    )
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="fp16",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--max-run-seconds", type=float, default=None)
    parser.add_argument("--seed", type=int, default=9710)
    parser.add_argument("--max-windows", type=int, default=100_000)
    parser.add_argument("--max-examples", type=int, default=1_000_000)
    parser.add_argument(
        "--max-input-bytes",
        type=int,
        default=4 * 1024 * 1024 * 1024,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--safety-fraction", type=float, default=0.90)
    parser.add_argument("--preflight-bundle", default=None)
    return parser


def _build_windows(
    tokenizer,
    records,
    *,
    sequence_length: int,
    micro_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    seed: int,
    max_windows: int,
):
    config = VN97TrainingConfig(
        sequence_length=sequence_length,
        stride=sequence_length,
        batch_size=micro_batch_size,
        epochs=1,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        seed=seed,
        shuffle=True,
        max_windows=max_windows,
    )
    return build_completion_windows(tokenizer, records, config)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    tokenizer_path = Path(args.tokenizer).resolve(strict=True)
    train_path = Path(args.train_jsonl).resolve(strict=True)
    validation_path = Path(args.validation_jsonl).resolve(strict=True)
    work_dir = Path(args.work_dir)
    output_dir = Path(args.output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_vn97tk1(tokenizer_path)
    train_records, _ = _load_records(
        [train_path],
        mode="chat",
        max_input_bytes=args.max_input_bytes,
        max_examples=args.max_examples,
    )
    validation_records, _ = _load_records(
        [validation_path],
        mode="chat",
        max_input_bytes=args.max_input_bytes,
        max_examples=args.max_examples,
    )
    corpus_evidence = build_pilot_corpus_evidence(
        train_records,
        validation_records,
        reject_overlap=True,
        reject_duplicates=True,
    )

    parent_sha = None
    parent_path = None
    if args.parent_checkpoint is not None:
        parent_path = Path(args.parent_checkpoint).resolve(strict=True)
        parent_sha = _sha256_file(parent_path)

    manifest = build_production_manifest(
        stage=args.stage,
        tokenizer_path=tokenizer_path,
        train_path=train_path,
        validation_path=validation_path,
        train_records=len(train_records),
        validation_records=len(validation_records),
        task_families=args.task_family,
        parent_checkpoint_sha256=parent_sha,
    )
    recipe = R2ProductionTrainingRecipe(
        sequence_length=args.sequence_length,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        precision=args.precision,
        activation_checkpointing=True,
        memory_efficient_scan=True,
        optimizer_state_offload=True,
        full_parameter_training=True,
        quantization_used=False,
    )
    trainer = R2ProductionTrainerConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        checkpoint_every_optimizer_steps=args.checkpoint_every,
        require_measured_cuda_preflight=True,
    )

    windows_train = _build_windows(
        tokenizer,
        train_records,
        sequence_length=args.sequence_length,
        micro_batch_size=args.micro_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        max_windows=args.max_windows,
    )
    windows_validation = _build_windows(
        tokenizer,
        validation_records,
        sequence_length=args.sequence_length,
        micro_batch_size=args.micro_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        max_windows=args.max_windows,
    )
    if not windows_train or not windows_validation:
        raise RuntimeError("R2-D4 corpus produced no training/validation windows")

    if args.stage == "dense_pretrain":
        model, parent_evidence = load_production_stage_model(
            manifest,
            config=r2_mobile_1b_config(tokenizer.vocab_size),
            initialization_seed=args.seed,
        )
    else:
        model, parent_evidence = load_production_stage_model(
            manifest,
            parent_checkpoint_path=parent_path,
        )

    assert_r2_data_compatible(model, tokenizer, windows_train)
    assert_r2_data_compatible(model, tokenizer, windows_validation)

    bundle_path = (
        Path(args.preflight_bundle)
        if args.preflight_bundle is not None
        else output_dir / "r2d4-preflight.json"
    )

    if args.mode == "preflight":
        sample = windows_train[: args.micro_batch_size]
        inputs = torch.tensor(
            [item.input_ids for item in sample],
            dtype=torch.long,
        )
        labels = torch.tensor(
            [item.labels for item in sample],
            dtype=torch.long,
        )
        if inputs.shape[0] != args.micro_batch_size:
            raise RuntimeError(
                "R2-D4 preflight requires at least one full micro-batch"
            )
        evidence = measure_cuda_training_preflight(
            model,
            inputs,
            labels,
            recipe,
            safety_fraction=args.safety_fraction,
            device=args.device,
        )
        save_preflight_bundle(
            bundle_path,
            manifest=manifest,
            recipe=recipe,
            evidence=evidence,
        )
        print(
            "VN97R2D4 "
            f"mode=preflight passed={str(evidence.passed).lower()} "
            f"peak_reserved={evidence.peak_reserved_bytes} "
            f"free_before={evidence.free_device_bytes_before} "
            f"bundle={bundle_path}",
            flush=True,
        )
        return 0 if evidence.passed else 2

    evidence = load_preflight_bundle(
        bundle_path,
        manifest=manifest,
        recipe=recipe,
    )
    result = train_production_stage(
        model,
        windows_train,
        windows_validation,
        manifest,
        recipe,
        trainer,
        work_dir=work_dir,
        best_checkpoint_path=output_dir / "model.r2.pt",
        parent_evidence=parent_evidence,
        device=args.device,
        measured_preflight=evidence,
        max_run_seconds=args.max_run_seconds,
    )
    report = {
        "schema": "VN97R2D4TRAIN1",
        "stage": manifest.stage,
        "manifest_identity": manifest.identity(),
        "recipe_fingerprint": recipe.fingerprint(),
        "preflight_bundle_sha256": _sha256_file(bundle_path),
        "corpus_isolation": corpus_evidence.as_dict(),
        "training": asdict(result),
        "quantization_used": False,
    }
    _atomic_write(
        output_dir / "r2d4-training-report.json",
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8") + b"\n",
    )
    print(
        "VN97R2D4 "
        f"mode=train completed={str(result.completed).lower()} "
        f"optimizer_steps={result.optimizer_steps} "
        f"best_val_loss={result.best_validation_loss:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
