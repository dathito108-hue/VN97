from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from ..training_cli import _atomic_write
from .config import r2_mobile_1b_config
from .data_bridge import load_vn97tk1
from .production_contract import R2ProductionTrainingRecipe
from .production_corpus_scale import (
    R2D6_DEFAULT_MIN_TOKENS_PER_PARAMETER,
    R2D6_DEFAULT_TARGET_TOKENS_PER_PARAMETER,
    verify_r2d6_corpus_index,
)
from .production_curriculum import load_r2d8_plan
from .production_streaming import (
    load_r2d5_memory_receipt,
    production_manifest_from_r2d6,
    train_streaming_production_stage,
)
from .production_training import (
    R2ProductionTrainerConfig,
    load_production_stage_model,
)
from .production_virtual_corpus import (
    projection_for_stage,
    verify_r2d12_view,
)


def _plan_preview(path: Path) -> dict[str, object]:
    if path.is_symlink():
        raise ValueError("R2-D7 curriculum plan must not be a symlink")
    try:
        value = json.loads(
            path.resolve(strict=True).read_text(
                encoding="utf-8",
                errors="strict",
            ),
            parse_constant=lambda raw: (
                _ for _ in ()
            ).throw(ValueError(raw)),
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "R2-D7 curriculum plan must be strict UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError("R2-D7 curriculum plan must be an object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "VN97-R2D7 deterministic streaming-shard dense trainer. "
            "Consumes either one verified R2-D6 package or an R2-D12 "
            "virtual multi-batch corpus without materializing the complete "
            "corpus/window set in RAM."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--corpus-package")
    source.add_argument("--virtual-view")
    parser.add_argument("--workspace-root")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preflight-receipt", default=None)
    parser.add_argument("--curriculum-plan", required=True)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
    )
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
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    curriculum_path = Path(args.curriculum_plan)
    virtual_view_id = None
    shard_roots = None

    if args.virtual_view:
        if args.workspace_root is None:
            raise RuntimeError(
                "R2-D7 --virtual-view requires --workspace-root"
            )
        verified_view = verify_r2d12_view(
            Path(args.virtual_view),
            workspace_root=Path(args.workspace_root),
        )
        preview = _plan_preview(curriculum_path)
        weights = preview.get("family_weights")
        if not isinstance(weights, dict) or not weights:
            raise RuntimeError(
                "R2-D7 curriculum plan family_weights are missing"
            )
        index = projection_for_stage(
            verified_view["index"],
            stage=str(preview.get("stage")),
            family_weights=weights,
        )
        package = Path(args.virtual_view).resolve(strict=True)
        tokenizer_path = verified_view["tokenizer_path"]
        shard_roots = verified_view["shard_roots"]
        virtual_view_id = verified_view["view"]["view_id"]
    else:
        if args.workspace_root is not None:
            raise RuntimeError(
                "--workspace-root is only valid with --virtual-view"
            )
        package = Path(args.corpus_package).resolve(strict=True)
        index = verify_r2d6_corpus_index(package)
        tokenizer_path = package / "tokenizer.vn97tk1"

    scale = index.get("scale")
    if not isinstance(scale, dict):
        raise RuntimeError("R2-D7 corpus scale evidence is missing")
    if (
        float(scale.get("minimum_tokens_per_parameter", float("nan")))
        != R2D6_DEFAULT_MIN_TOKENS_PER_PARAMETER
        or float(scale.get("target_tokens_per_parameter", float("nan")))
        != R2D6_DEFAULT_TARGET_TOKENS_PER_PARAMETER
    ):
        raise RuntimeError(
            "R2-D7 production CLI requires the canonical 8/20 "
            "token-per-parameter scale policy"
        )
    if scale.get("scale_floor_passed") is not True:
        raise RuntimeError(
            "R2-D7 refuses to allocate the production model because the "
            "selected corpus projection is below the data-scale floor"
        )
    if index.get("release_held_out") is not True:
        raise RuntimeError("R2-D7 release holdout is not locked")
    if int(index.get("sequence_length", -1)) != args.sequence_length:
        raise RuntimeError(
            "R2-D7 --sequence-length must equal the corpus index"
        )

    curriculum = load_r2d8_plan(
        curriculum_path,
        index,
    )
    if curriculum.get("stage") != "dense_pretrain":
        raise RuntimeError(
            "R2-D7 production CLI requires dense_pretrain curriculum"
        )
    if int(curriculum.get("epochs", -1)) != args.epochs:
        raise RuntimeError(
            "R2-D7 --epochs must match the R2-D8 curriculum"
        )
    if int(curriculum.get("seed", -1)) != args.seed:
        raise RuntimeError(
            "R2-D7 --seed must match the R2-D8 curriculum"
        )

    recipe = R2ProductionTrainingRecipe(
        sequence_length=args.sequence_length,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),
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
        checkpoint_every_optimizer_steps=(
            args.checkpoint_every
        ),
        require_measured_cuda_preflight=True,
    )

    tokenizer = load_vn97tk1(
        Path(tokenizer_path).resolve(strict=True)
    )
    config = r2_mobile_1b_config(tokenizer.vocab_size)
    if index.get("architecture_fingerprint") != config.fingerprint():
        raise RuntimeError(
            "R2-D7 corpus architecture/tokenizer identity mismatch"
        )

    if not str(args.device).startswith("cuda"):
        raise RuntimeError(
            "R2-D7 production CLI requires an explicit CUDA device; "
            "CPU execution is reserved for validation/tests"
        )
    if args.preflight_receipt is None:
        raise RuntimeError(
            "R2-D7 CUDA training requires --preflight-receipt"
        )
    preflight = load_r2d5_memory_receipt(
        Path(args.preflight_receipt),
        architecture_fingerprint=config.fingerprint(),
        recipe=recipe,
    )

    manifest = production_manifest_from_r2d6(index)
    model, parent_evidence = load_production_stage_model(
        manifest,
        config=config,
        initialization_seed=args.seed,
    )
    if parent_evidence is not None:
        raise RuntimeError(
            "R2-D7 dense_pretrain unexpectedly received parent evidence"
        )

    work_dir = Path(args.work_dir)
    output_dir = Path(args.output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = train_streaming_production_stage(
        model,
        package,
        recipe,
        trainer,
        work_dir=work_dir,
        best_checkpoint_path=output_dir / "model.r2.pt",
        device=args.device,
        measured_preflight=preflight,
        max_run_seconds=args.max_run_seconds,
        curriculum_plan=curriculum,
        corpus_index=index,
        shard_roots=shard_roots,
        tokenizer_path=tokenizer_path,
    )

    report = {
        "schema": "VN97R2D7TRAIN1",
        "corpus_index_id": index["index_id"],
        "corpus_index_schema": index.get("schema"),
        "virtual_view_id": virtual_view_id,
        "architecture_fingerprint": config.fingerprint(),
        "production_manifest_identity": manifest.identity(),
        "curriculum_plan_id": curriculum["plan_id"],
        "recipe_fingerprint": recipe.fingerprint(),
        "trainer": asdict(trainer),
        "scale": scale,
        "release_held_out": True,
        "training": {
            **asdict(result),
            "next_cursor": result.next_cursor.as_dict(),
        },
        "quantization_used": False,
    }
    _atomic_write(
        output_dir / "r2d7-training-report.json",
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8") + b"\n",
    )

    print(
        "VN97R2D7 "
        f"completed={str(result.completed).lower()} "
        f"optimizer_steps={result.optimizer_steps} "
        f"micro_steps={result.micro_steps} "
        f"windows={result.consumed_windows} "
        f"next_epoch={result.next_cursor.epoch} "
        f"best_val_loss={result.best_validation_loss:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-r2-stream-train: {exc}", file=sys.stderr)
        raise
