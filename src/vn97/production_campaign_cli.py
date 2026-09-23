from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys

from .deployment_checkpoint import (
    build_deployment_checkpoint,
    load_deployment_checkpoint,
    load_deployment_checkpoint_file,
)
from .mobile_budget import VN97MobileBudget, estimate_vn97_mobile_footprint
from .model_image import build_model_image
from .modality import AudioAdapterConfig, AudioFrameAdapter
from .speech_training import (
    VN97SpeechReleaseCriteria,
    VN97SpeechTrainingConfig,
    evaluate_vn97_speech,
    require_disjoint_speech_splits,
    require_speech_release_quality,
    train_vn97_speech_adapter,
)
from .speech_training_cli import load_speech_manifest
from .tokenizer import VN97Tokenizer, VN97TokenizerPackage
from .training_cli import _atomic_write, _canonical_json


_MAX_REPORT_BYTES = 4 * 1024 * 1024
_MAX_TOKENIZER_BYTES = 64 * 1024 * 1024


def _read_regular_file(path: Path, *, max_bytes: int, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{label} could not be opened safely: {path}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{label} must be a regular file: {path}")
        if not 0 < info.st_size <= max_bytes:
            raise ValueError(f"{label} size is outside bounds")
        out = bytearray()
        while len(out) < info.st_size:
            chunk = os.read(fd, min(1024 * 1024, info.st_size - len(out)))
            if not chunk:
                break
            out.extend(chunk)
        after = os.fstat(fd)
        if (
            len(out) != info.st_size
            or after.st_size != info.st_size
            or after.st_ino != info.st_ino
            or after.st_dev != info.st_dev
        ):
            raise ValueError(f"{label} changed while being read: {path}")
        return bytes(out)
    finally:
        os.close(fd)


def _load_language_campaign_report(path: Path) -> tuple[dict[str, object], str]:
    data = _read_regular_file(
        path,
        max_bytes=_MAX_REPORT_BYTES,
        label="language campaign report",
    )
    duplicates: list[str] = []

    def hook(pairs):
        output = {}
        for key, value in pairs:
            if key in output:
                duplicates.append(key)
            output[key] = value
        return output

    try:
        text = data.decode("utf-8", errors="strict")
        report = json.loads(
            text,
            object_pairs_hook=hook,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(value)
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            "language campaign report must be strict UTF-8 JSON"
        ) from exc
    if duplicates or not isinstance(report, dict):
        raise ValueError(
            "language campaign report must be one object without duplicate keys"
        )
    if report.get("schema") != "VN97CAMP2":
        raise ValueError("language campaign report schema must be VN97CAMP2")
    canonical = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if canonical != text:
        raise ValueError("language campaign report must use canonical JSON")

    for key in (
        "selected_checkpoint_sha256",
        "tokenizer_sha256",
        "selected_candidate_id",
    ):
        if key not in report or not isinstance(report[key], str) or not report[key]:
            raise ValueError(f"language campaign report is missing {key}")
    for key in ("selected_checkpoint_sha256", "tokenizer_sha256"):
        value = report[key]
        if (
            len(value) != 64
            or any(ch not in "0123456789abcdef" for ch in value)
        ):
            raise ValueError(f"language campaign report {key} is invalid")
    return report, hashlib.sha256(data).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize the selected VN97 language campaign winner with one "
            "canonical speech adapter and sealed speech release gate."
        )
    )
    parser.add_argument("--language-campaign-dir", required=True)
    parser.add_argument("--speech-input", required=True)
    parser.add_argument("--speech-validation-input", required=True)
    parser.add_argument("--speech-release-input", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--speech-max-examples", type=int, default=100_000)
    parser.add_argument("--speech-epochs", type=int, default=1)
    parser.add_argument("--speech-learning-rate", type=float, default=3e-4)
    parser.add_argument("--speech-weight-decay", type=float, default=0.01)
    parser.add_argument("--speech-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--speech-max-frames", type=int, default=1500)
    parser.add_argument("--speech-max-target-tokens", type=int, default=512)
    parser.add_argument("--speech-seed", type=int, default=97)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--max-speech-validation-loss", type=float, required=True)
    parser.add_argument(
        "--min-speech-validation-accuracy",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--min-speech-validation-target-tokens",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--min-speech-validation-examples",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--release-max-speech-loss",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--release-min-speech-accuracy",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--release-min-speech-target-tokens",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--release-min-speech-examples",
        type=int,
        default=None,
    )

    parser.add_argument("--tile-rows", type=int, default=16)
    parser.add_argument("--tile-cols", type=int, default=16)
    parser.add_argument(
        "--max-model-image-bytes",
        type=int,
        default=512 * 1024 * 1024,
    )
    parser.add_argument(
        "--max-recurrent-state-bytes",
        type=int,
        default=512 * 1024 * 1024,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (
        args.speech_max_examples <= 0
        or args.speech_epochs <= 0
        or args.speech_max_frames <= 0
        or args.speech_max_target_tokens <= 1
        or not math.isfinite(args.speech_learning_rate)
        or args.speech_learning_rate <= 0.0
        or not math.isfinite(args.speech_weight_decay)
        or args.speech_weight_decay < 0.0
        or not math.isfinite(args.speech_max_grad_norm)
        or args.speech_max_grad_norm <= 0.0
    ):
        raise ValueError("production campaign speech bounds are invalid")

    campaign_dir = Path(args.language_campaign_dir)
    if campaign_dir.is_symlink():
        raise ValueError("language campaign directory must not be a symlink")
    campaign_dir = campaign_dir.resolve(strict=True)
    if not campaign_dir.is_dir():
        raise ValueError("language campaign directory must be a directory")

    language_report, language_report_sha256 = _load_language_campaign_report(
        campaign_dir / "campaign-report.json"
    )
    checkpoint = load_deployment_checkpoint_file(
        campaign_dir / "model.vn97ck1"
    )
    if checkpoint.audio_adapter is not None:
        raise ValueError(
            "M11C language campaign input must not already contain speech weights"
        )
    if (
        checkpoint.checkpoint_sha256
        != language_report["selected_checkpoint_sha256"]
    ):
        raise ValueError(
            "language campaign checkpoint identity does not match report"
        )

    tokenizer_bytes = _read_regular_file(
        campaign_dir / "tokenizer.vn97tk1",
        max_bytes=_MAX_TOKENIZER_BYTES,
        label="language campaign tokenizer",
    )
    tokenizer_sha256 = hashlib.sha256(tokenizer_bytes).hexdigest()
    if tokenizer_sha256 != language_report["tokenizer_sha256"]:
        raise ValueError(
            "language campaign tokenizer identity does not match report"
        )
    tokenizer_package = VN97TokenizerPackage.from_bytes(tokenizer_bytes)
    if tokenizer_package.vocab_size != checkpoint.config.vocab_size:
        raise ValueError(
            "language campaign tokenizer vocabulary does not match checkpoint"
        )
    tokenizer = VN97Tokenizer(tokenizer_package)

    speech_training, speech_training_sha256 = load_speech_manifest(
        Path(args.speech_input),
        max_examples=args.speech_max_examples,
    )
    speech_validation, speech_validation_sha256 = load_speech_manifest(
        Path(args.speech_validation_input),
        max_examples=args.speech_max_examples,
    )
    speech_release, speech_release_sha256 = load_speech_manifest(
        Path(args.speech_release_input),
        max_examples=args.speech_max_examples,
    )
    require_disjoint_speech_splits(
        speech_training,
        speech_validation,
        speech_release,
    )

    adapter = AudioFrameAdapter(
        checkpoint.config.d_model,
        ternary_threshold=checkpoint.config.ternary_threshold,
        config=AudioAdapterConfig(),
        rms_eps=checkpoint.config.rms_eps,
    )
    training_config = VN97SpeechTrainingConfig(
        epochs=args.speech_epochs,
        learning_rate=args.speech_learning_rate,
        weight_decay=args.speech_weight_decay,
        max_grad_norm=args.speech_max_grad_norm,
        seed=args.speech_seed,
        shuffle=True,
        max_frames=args.speech_max_frames,
        max_target_tokens=args.speech_max_target_tokens,
    )
    training = train_vn97_speech_adapter(
        checkpoint.model,
        adapter,
        tokenizer,
        speech_training,
        training_config,
        device=args.device,
    )

    validation_config = VN97SpeechTrainingConfig(
        epochs=1,
        seed=0,
        shuffle=False,
        max_frames=args.speech_max_frames,
        max_target_tokens=args.speech_max_target_tokens,
    )
    validation = evaluate_vn97_speech(
        checkpoint.model,
        adapter,
        tokenizer,
        speech_validation,
        config=validation_config,
        device=args.device,
    )
    validation_criteria = VN97SpeechReleaseCriteria(
        max_validation_loss=args.max_speech_validation_loss,
        min_top1_accuracy=args.min_speech_validation_accuracy,
        min_target_tokens=args.min_speech_validation_target_tokens,
        min_examples=args.min_speech_validation_examples,
    )
    require_speech_release_quality(validation, validation_criteria)

    unified_checkpoint = build_deployment_checkpoint(
        checkpoint.model,
        audio_adapter=adapter,
    )
    unified = load_deployment_checkpoint(unified_checkpoint)
    if unified.audio_adapter is None:
        raise RuntimeError(
            "unified production checkpoint lost speech adapter"
        )

    release = evaluate_vn97_speech(
        unified.model,
        unified.audio_adapter,
        tokenizer,
        speech_release,
        config=validation_config,
        device=args.device,
    )
    release_criteria = VN97SpeechReleaseCriteria(
        max_validation_loss=(
            validation_criteria.max_validation_loss
            if args.release_max_speech_loss is None
            else args.release_max_speech_loss
        ),
        min_top1_accuracy=(
            validation_criteria.min_top1_accuracy
            if args.release_min_speech_accuracy is None
            else args.release_min_speech_accuracy
        ),
        min_target_tokens=(
            validation_criteria.min_target_tokens
            if args.release_min_speech_target_tokens is None
            else args.release_min_speech_target_tokens
        ),
        min_examples=(
            validation_criteria.min_examples
            if args.release_min_speech_examples is None
            else args.release_min_speech_examples
        ),
    )
    try:
        require_speech_release_quality(release, release_criteria)
    except Exception as exc:
        raise RuntimeError(
            "selected language campaign winner failed sealed speech release gate"
        ) from exc

    footprint = estimate_vn97_mobile_footprint(
        unified.config,
        tokenizer_nbytes=len(tokenizer_bytes),
        audio_frame_size=unified.audio_adapter.config.frame_size,
        tile_rows=args.tile_rows,
        tile_cols=args.tile_cols,
        batch_size=1,
    )
    budget = VN97MobileBudget(
        max_model_image_bytes=args.max_model_image_bytes,
        max_recurrent_state_bytes=args.max_recurrent_state_bytes,
    )
    rejection = budget.rejection_status(footprint)
    if rejection is not None:
        raise RuntimeError(
            "selected production intelligence exceeds mobile budget: "
            + rejection
        )

    image = build_model_image(
        unified.model,
        tokenizer=tokenizer_package,
        audio_adapter=unified.audio_adapter,
        tile_rows=args.tile_rows,
        tile_cols=args.tile_cols,
    )
    model_image_sha256 = hashlib.sha256(image.data).hexdigest()

    speech_training_report = {
        "base_checkpoint_sha256": checkpoint.checkpoint_sha256,
        "checkpoint_sha256": unified.checkpoint_sha256,
        "dataset_sha256": speech_training_sha256,
        "examples": training.examples,
        "final_loss": training.final_loss,
        "mean_loss": training.mean_loss,
        "schema": "VN97SPEECHTRAIN1",
        "steps": training.steps,
        "target_tokens": training.target_tokens,
        "tokenizer_sha256": tokenizer_sha256,
        "training": {
            "epochs": training_config.epochs,
            "learning_rate": training_config.learning_rate,
            "max_frames": training_config.max_frames,
            "max_grad_norm": training_config.max_grad_norm,
            "max_target_tokens": training_config.max_target_tokens,
            "seed": training_config.seed,
            "weight_decay": training_config.weight_decay,
        },
    }
    speech_training_report_bytes = _canonical_json(
        speech_training_report
    )

    production_report = {
        "language_campaign_report_sha256": language_report_sha256,
        "deployment": {
            "tile_cols": args.tile_cols,
            "tile_rows": args.tile_rows,
        },
        "language_checkpoint_sha256": checkpoint.checkpoint_sha256,
        "mobile_budget": {
            "max_model_image_bytes": budget.max_model_image_bytes,
            "max_recurrent_state_bytes": budget.max_recurrent_state_bytes,
        },
        "mobile_footprint": footprint.canonical_object(),
        "model_image_sha256": model_image_sha256,
        "schema": "VN97PRODCAMP1",
        "selected_candidate_id": language_report["selected_candidate_id"],
        "speech_release": {
            "dataset_sha256": speech_release_sha256,
            "examples": release.examples,
            "max_loss": release_criteria.max_validation_loss,
            "mean_loss": release.mean_loss,
            "min_target_tokens": release_criteria.min_target_tokens,
            "min_top1_accuracy": release_criteria.min_top1_accuracy,
            "target_tokens": release.target_tokens,
            "top1_accuracy": release.top1_accuracy,
        },
        "speech_training_dataset_sha256": speech_training_sha256,
        "speech_validation": {
            "dataset_sha256": speech_validation_sha256,
            "examples": validation.examples,
            "max_loss": validation_criteria.max_validation_loss,
            "mean_loss": validation.mean_loss,
            "min_target_tokens": validation_criteria.min_target_tokens,
            "min_top1_accuracy": validation_criteria.min_top1_accuracy,
            "target_tokens": validation.target_tokens,
            "top1_accuracy": validation.top1_accuracy,
        },
        "tokenizer_sha256": tokenizer_sha256,
        "unified_checkpoint_sha256": unified.checkpoint_sha256,
    }

    output = Path(args.output_dir)
    if output.is_symlink():
        raise ValueError("production campaign output-dir must not be a symlink")
    output.mkdir(parents=True, exist_ok=True)
    _atomic_write(output / "model.vn97ck1", unified_checkpoint)
    _atomic_write(output / "tokenizer.vn97tk1", tokenizer_bytes)
    _atomic_write(
        output / "speech-training-report.json",
        speech_training_report_bytes,
    )
    _atomic_write(
        output / "production-campaign-report.json",
        _canonical_json(production_report),
    )

    print(
        "VN97PRODCAMP1 "
        f"candidate={language_report['selected_candidate_id']} "
        f"checkpoint_sha256={unified.checkpoint_sha256} "
        f"model_image_sha256={model_image_sha256}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-production-campaign: {exc}", file=sys.stderr)
        raise
