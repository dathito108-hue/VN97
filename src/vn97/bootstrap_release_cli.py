from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys

from .bootstrap_bundle import (
    Ed25519PrivateKeySigner,
    build_bootstrap_bundle,
    write_bootstrap_assets,
)
from .capability_package import CapabilitySource
from .deployment_checkpoint import load_deployment_checkpoint_file
from .device_evidence import (
    VN97DeviceEvidenceCriteria,
    load_device_evidence,
    require_device_evidence,
)
from .evaluation import (
    VN97ReleaseCriteria,
    evaluate_vn97_language,
    require_release_quality,
)
from .speech_training import (
    VN97SpeechReleaseCriteria,
    VN97SpeechTrainingConfig,
    evaluate_vn97_speech,
    require_speech_release_quality,
)
from .speech_training_cli import load_speech_manifest
from .mobile_budget import VN97MobileBudget, estimate_vn97_mobile_footprint
from .model_image import build_model_image
from .release_candidate import (
    VN97LoadedReleaseCandidate,
    load_release_candidate_directory,
)
from .tokenizer import VN97Tokenizer, VN97TokenizerPackage
from .training import (
    VN97TrainingConfig,
    build_training_windows,
    encode_causal_text,
    encode_chat_messages,
)
from .training_cli import _load_records
from .vision_training import (
    VN97VisionReleaseCriteria,
    VN97VisionTrainingConfig,
    evaluate_vn97_vision,
    require_vision_release_quality,
)
from .vision_training_cli import load_vision_manifest


_MAX_TOKENIZER_BYTES = 64 * 1024 * 1024
_MAX_PRIVATE_KEY_FILE_BYTES = 256


def _read_regular_file(
    path: Path,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
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
            raise ValueError(f"{label} changed while being read")
        return bytes(out)
    finally:
        os.close(fd)


def _parse_private_key(data: bytes) -> bytes:
    if len(data) == 32:
        return bytes(data)
    try:
        text = data.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "Ed25519 private key must be 32 raw bytes or 64 lowercase hex characters"
        ) from exc
    if (
        len(text) == 64
        and all(ch in "0123456789abcdef" for ch in text)
    ):
        return bytes.fromhex(text)
    raise ValueError(
        "Ed25519 private key must be 32 raw bytes or 64 lowercase hex characters"
    )


def _load_speech_training_report(
    path: Path,
    *,
    checkpoint_sha256: str,
    tokenizer_sha256: str,
) -> tuple[dict[str, object], str]:
    data = _read_regular_file(
        path,
        max_bytes=1024 * 1024,
        label="speech training report",
    )
    try:
        text = data.decode("utf-8", errors="strict")
        report = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("speech training report must be strict UTF-8 JSON") from exc
    expected_keys = {
        "base_checkpoint_sha256",
        "checkpoint_sha256",
        "dataset_sha256",
        "examples",
        "final_loss",
        "mean_loss",
        "schema",
        "steps",
        "target_tokens",
        "tokenizer_sha256",
        "training",
    }
    if not isinstance(report, dict) or set(report) != expected_keys:
        raise ValueError("speech training report keys are not exact")
    if report["schema"] != "VN97SPEECHTRAIN1":
        raise ValueError("speech training report schema mismatch")
    if report["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError(
            "speech training report checkpoint does not match release checkpoint"
        )
    if report["tokenizer_sha256"] != tokenizer_sha256:
        raise ValueError(
            "speech training report tokenizer does not match release tokenizer"
        )
    for key in (
        "base_checkpoint_sha256",
        "checkpoint_sha256",
        "dataset_sha256",
        "tokenizer_sha256",
    ):
        value = report[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(ch not in "0123456789abcdef" for ch in value)
        ):
            raise ValueError(f"speech training report {key} is invalid")
    if (
        type(report["examples"]) is not int
        or report["examples"] <= 0
        or type(report["steps"]) is not int
        or report["steps"] <= 0
        or type(report["target_tokens"]) is not int
        or report["target_tokens"] <= 0
    ):
        raise ValueError("speech training report work counters are invalid")
    for key in ("final_loss", "mean_loss"):
        value = report[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"speech training report {key} is invalid")
    if not isinstance(report["training"], dict):
        raise ValueError("speech training report training config is invalid")
    canonical = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if canonical != text:
        raise ValueError("speech training report must use canonical JSON")
    return report, hashlib.sha256(data).hexdigest()


def _load_vision_training_report(
    path: Path,
    *,
    checkpoint_sha256: str,
    tokenizer_sha256: str,
) -> tuple[dict[str, object], str]:
    data = _read_regular_file(
        path,
        max_bytes=1024 * 1024,
        label="vision training report",
    )
    try:
        text = data.decode("utf-8", errors="strict")
        report = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "vision training report must be strict UTF-8 JSON"
        ) from exc
    expected = {
        "base_checkpoint_sha256",
        "checkpoint_sha256",
        "dataset_sha256",
        "examples",
        "final_loss",
        "mean_loss",
        "schema",
        "steps",
        "target_tokens",
        "tokenizer_sha256",
        "training",
    }
    if not isinstance(report, dict) or set(report) != expected:
        raise ValueError("vision training report keys are not exact")
    if report["schema"] != "VN97VISIONTRAIN1":
        raise ValueError("vision training report schema mismatch")
    if report["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError(
            "vision training report checkpoint does not match release checkpoint"
        )
    if report["tokenizer_sha256"] != tokenizer_sha256:
        raise ValueError(
            "vision training report tokenizer does not match release tokenizer"
        )
    canonical = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if canonical != text:
        raise ValueError("vision training report must use canonical JSON")
    return report, hashlib.sha256(data).hexdigest()


def _load_production_campaign_report(
    path: Path,
    *,
    checkpoint_sha256: str,
    tokenizer_sha256: str,
    model_image_sha256: str,
) -> str:
    data = _read_regular_file(
        path,
        max_bytes=4 * 1024 * 1024,
        label="production campaign report",
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
            parse_constant=lambda raw: (_ for _ in ()).throw(
                ValueError(raw)
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            "production campaign report must be strict UTF-8 JSON"
        ) from exc
    if duplicates or not isinstance(report, dict):
        raise ValueError(
            "production campaign report must be one object without duplicate keys"
        )
    canonical = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if canonical != text:
        raise ValueError(
            "production campaign report must use canonical JSON"
        )
    if report.get("schema") != "VN97PRODCAMP1":
        raise ValueError(
            "production campaign report schema mismatch"
        )
    expected = {
        "unified_checkpoint_sha256": checkpoint_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "model_image_sha256": model_image_sha256,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(
                f"production campaign report {key} does not match release input"
            )
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create the exact signed VN97 bootstrap assets consumed by the M10J Android app."
        )
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--tokenizer")
    parser.add_argument(
        "--release-candidate-dir",
        help=(
            "verified VN97RC1 candidate directory; resolves checkpoint, "
            "tokenizer, production report, modality reports, device evidence, "
            "and deployment tile geometry"
        ),
    )
    parser.add_argument("--private-key", required=True)
    parser.add_argument("--key-id", required=True)
    parser.add_argument("--capability-version", type=int, required=True)
    parser.add_argument("--source-origin", required=True)
    parser.add_argument("--source-license", required=True)
    parser.add_argument("--assets-dir", required=True)

    parser.add_argument(
        "--validation-input",
        action="append",
        required=True,
        help="held-out local UTF-8 JSONL; repeat for multiple files",
    )
    parser.add_argument(
        "--validation-format",
        choices=("text", "chat"),
        default="chat",
    )
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
    parser.add_argument(
        "--validation-sequence-length",
        type=int,
        default=256,
    )
    parser.add_argument("--validation-stride", type=int, default=None)
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=4,
    )
    parser.add_argument("--validation-device", default="auto")
    parser.add_argument(
        "--speech-training-report",
        help=(
            "canonical VN97SPEECHTRAIN1 report matching the speech-enabled checkpoint"
        ),
    )
    parser.add_argument(
        "--speech-validation-input",
        help=(
            "held-out speech JSONL with 16 kHz mono PCM16 WAV paths; "
            "required for speech-enabled VN97CK1"
        ),
    )
    parser.add_argument(
        "--speech-validation-max-examples",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--speech-max-frames",
        type=int,
        default=1500,
    )
    parser.add_argument(
        "--speech-max-target-tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--max-speech-validation-loss",
        type=float,
    )
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
    parser.add_argument("--vision-training-report")
    parser.add_argument("--vision-validation-input")
    parser.add_argument(
        "--vision-validation-max-examples",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--vision-max-patches",
        type=int,
        default=196,
    )
    parser.add_argument(
        "--vision-max-target-tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--max-vision-validation-loss",
        type=float,
    )
    parser.add_argument(
        "--min-vision-validation-accuracy",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--min-vision-validation-target-tokens",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--min-vision-validation-examples",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-validation-loss",
        type=float,
        required=True,
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

    parser.add_argument("--tile-rows", type=int)
    parser.add_argument("--tile-cols", type=int)
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
    parser.add_argument("--production-campaign-report")
    parser.add_argument(
        "--require-production-campaign-report",
        action="store_true",
    )
    parser.add_argument("--device-evidence")
    parser.add_argument(
        "--require-device-evidence",
        action="store_true",
    )
    parser.add_argument(
        "--device-evidence-min-runs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-text-prefill-p95-ms",
        type=float,
        default=10_000.0,
    )
    parser.add_argument(
        "--max-text-decode-p95-ms-per-token",
        type=float,
        default=2_000.0,
    )
    parser.add_argument(
        "--max-device-peak-pss-kib",
        type=int,
        default=2 * 1024 * 1024,
    )
    parser.add_argument(
        "--max-device-thermal-status",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-speech-prefill-p95-ms",
        type=float,
        default=15_000.0,
    )
    parser.add_argument(
        "--require-device-energy-counter",
        action="store_true",
    )
    parser.add_argument(
        "--max-abs-battery-energy-counter-delta-nwh",
        type=int,
        default=None,
    )
    return parser


def _resolve_release_inputs(
    args: argparse.Namespace,
) -> tuple[
    VN97LoadedReleaseCandidate | None,
    Path,
    Path,
    Path | None,
    Path | None,
    Path | None,
    tuple[Path, ...],
    int,
    int,
]:
    candidate: VN97LoadedReleaseCandidate | None = None
    if args.release_candidate_dir is not None:
        conflicting = {
            "--checkpoint": args.checkpoint,
            "--tokenizer": args.tokenizer,
            "--production-campaign-report":
                args.production_campaign_report,
            "--speech-training-report":
                args.speech_training_report,
            "--vision-training-report":
                args.vision_training_report,
            "--device-evidence":
                args.device_evidence,
        }
        supplied = [
            name
            for name, value in conflicting.items()
            if value is not None
        ]
        if supplied:
            raise ValueError(
                "--release-candidate-dir cannot be combined with: "
                + ", ".join(supplied)
            )
        candidate = load_release_candidate_directory(
            args.release_candidate_dir
        )
        if (
            args.tile_rows is not None
            and args.tile_rows
            != candidate.manifest.tile_rows
        ):
            raise ValueError(
                "--tile-rows does not match VN97RC1"
            )
        if (
            args.tile_cols is not None
            and args.tile_cols
            != candidate.manifest.tile_cols
        ):
            raise ValueError(
                "--tile-cols does not match VN97RC1"
            )
        return (
            candidate,
            candidate.checkpoint_path,
            candidate.tokenizer_path,
            candidate.production_campaign_report_path,
            candidate.speech_training_report_path,
            candidate.vision_training_report_path,
            candidate.device_evidence_paths,
            candidate.manifest.tile_rows,
            candidate.manifest.tile_cols,
        )

    if args.checkpoint is None or args.tokenizer is None:
        raise ValueError(
            "raw release mode requires --checkpoint and --tokenizer"
        )
    return (
        None,
        Path(args.checkpoint),
        Path(args.tokenizer),
        (
            None
            if args.production_campaign_report is None
            else Path(args.production_campaign_report)
        ),
        (
            None
            if args.speech_training_report is None
            else Path(args.speech_training_report)
        ),
        (
            None
            if args.vision_training_report is None
            else Path(args.vision_training_report)
        ),
        (
            tuple()
            if args.device_evidence is None
            else (Path(args.device_evidence),)
        ),
        16 if args.tile_rows is None else args.tile_rows,
        16 if args.tile_cols is None else args.tile_cols,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    (
        release_candidate,
        checkpoint_path,
        tokenizer_path,
        production_campaign_report_path,
        speech_training_report_path,
        vision_training_report_path,
        device_evidence_paths,
        tile_rows,
        tile_cols,
    ) = _resolve_release_inputs(args)
    private_key_path = Path(args.private_key)
    assets_dir = Path(args.assets_dir)

    loaded = load_deployment_checkpoint_file(checkpoint_path)

    tokenizer_bytes = _read_regular_file(
        tokenizer_path,
        max_bytes=_MAX_TOKENIZER_BYTES,
        label="VN97TK1 tokenizer",
    )
    try:
        tokenizer = VN97TokenizerPackage.from_bytes(tokenizer_bytes)
    except ValueError as exc:
        raise ValueError("VN97TK1 tokenizer is invalid") from exc
    if tokenizer.vocab_size != loaded.config.vocab_size:
        raise ValueError(
            "VN97TK1 tokenizer vocabulary does not match VN97CK1 checkpoint"
        )
    tokenizer_sha256 = hashlib.sha256(tokenizer_bytes).hexdigest()

    if (
        args.validation_max_input_bytes <= 0
        or args.validation_max_examples <= 0
        or args.validation_max_windows <= 0
        or args.validation_sequence_length < 2
        or args.validation_batch_size <= 0
    ):
        raise ValueError("validation bounds must be positive")

    validation_records, validation_sha256 = _load_records(
        [Path(value) for value in args.validation_input],
        mode=args.validation_format,
        max_input_bytes=args.validation_max_input_bytes,
        max_examples=args.validation_max_examples,
    )
    runtime_tokenizer = VN97Tokenizer(tokenizer)
    validation_examples = (
        [
            encode_causal_text(runtime_tokenizer, value)
            for value in validation_records
        ]
        if args.validation_format == "text"
        else [
            encode_chat_messages(runtime_tokenizer, value)
            for value in validation_records
        ]
    )
    validation_config = VN97TrainingConfig(
        sequence_length=args.validation_sequence_length,
        stride=args.validation_stride,
        batch_size=args.validation_batch_size,
        epochs=1,
        seed=0,
        shuffle=False,
        max_windows=args.validation_max_windows,
    )
    validation_windows = build_training_windows(
        validation_examples,
        validation_config,
        pad_token_id=runtime_tokenizer.pad_id,
    )
    evaluation = evaluate_vn97_language(
        loaded.model,
        validation_windows,
        batch_size=args.validation_batch_size,
        device=args.validation_device,
    )
    criteria = VN97ReleaseCriteria(
        max_validation_loss=args.max_validation_loss,
        min_top1_accuracy=args.min_validation_accuracy,
        min_target_tokens=args.min_validation_target_tokens,
    )
    require_release_quality(evaluation, criteria)

    speech_evaluation = None
    speech_dataset_sha256 = None
    speech_training_report = None
    speech_training_report_sha256 = None
    if loaded.audio_adapter is not None:
        if speech_training_report_path is None:
            raise ValueError(
                "speech-enabled VN97CK1 requires --speech-training-report"
            )
        speech_training_report, speech_training_report_sha256 = (
            _load_speech_training_report(
                speech_training_report_path,
                checkpoint_sha256=loaded.checkpoint_sha256,
                tokenizer_sha256=tokenizer_sha256,
            )
        )
        if args.speech_validation_input is None:
            raise ValueError(
                "speech-enabled VN97CK1 requires --speech-validation-input"
            )
        if args.max_speech_validation_loss is None:
            raise ValueError(
                "speech-enabled VN97CK1 requires --max-speech-validation-loss"
            )
        if (
            args.speech_validation_max_examples <= 0
            or args.speech_max_frames <= 0
            or args.speech_max_target_tokens <= 1
            or args.min_speech_validation_target_tokens <= 0
            or args.min_speech_validation_examples <= 0
        ):
            raise ValueError("speech validation bounds must be positive")

        speech_examples, speech_dataset_sha256 = load_speech_manifest(
            Path(args.speech_validation_input),
            max_examples=args.speech_validation_max_examples,
        )
        if (
            speech_training_report["dataset_sha256"]
            == speech_dataset_sha256
        ):
            raise ValueError(
                "speech validation dataset must differ from the training dataset"
            )
        speech_config = VN97SpeechTrainingConfig(
            epochs=1,
            seed=0,
            shuffle=False,
            max_frames=args.speech_max_frames,
            max_target_tokens=args.speech_max_target_tokens,
        )
        speech_evaluation = evaluate_vn97_speech(
            loaded.model,
            loaded.audio_adapter,
            runtime_tokenizer,
            speech_examples,
            config=speech_config,
            device=args.validation_device,
        )
        require_speech_release_quality(
            speech_evaluation,
            VN97SpeechReleaseCriteria(
                max_validation_loss=args.max_speech_validation_loss,
                min_top1_accuracy=args.min_speech_validation_accuracy,
                min_target_tokens=args.min_speech_validation_target_tokens,
                min_examples=args.min_speech_validation_examples,
            ),
        )
    elif (
        args.speech_validation_input is not None
        or speech_training_report_path is not None
    ):
        raise ValueError(
            "speech release inputs were provided but VN97CK1 has no audio adapter"
        )

    vision_evaluation = None
    vision_dataset_sha256 = None
    vision_training_report = None
    vision_training_report_sha256 = None
    if loaded.vision_adapter is not None:
        if vision_training_report_path is None:
            raise ValueError(
                "vision-enabled VN97CK1 requires --vision-training-report"
            )
        if args.vision_validation_input is None:
            raise ValueError(
                "vision-enabled VN97CK1 requires --vision-validation-input"
            )
        if args.max_vision_validation_loss is None:
            raise ValueError(
                "vision-enabled VN97CK1 requires --max-vision-validation-loss"
            )
        vision_training_report, vision_training_report_sha256 = (
            _load_vision_training_report(
                vision_training_report_path,
                checkpoint_sha256=loaded.checkpoint_sha256,
                tokenizer_sha256=tokenizer_sha256,
            )
        )
        vision_examples, vision_dataset_sha256 = load_vision_manifest(
            Path(args.vision_validation_input),
            max_examples=args.vision_validation_max_examples,
        )
        if (
            vision_training_report["dataset_sha256"]
            == vision_dataset_sha256
        ):
            raise ValueError(
                "vision validation dataset must differ from training dataset"
            )
        vision_config = VN97VisionTrainingConfig(
            epochs=1,
            seed=0,
            shuffle=False,
            max_patches=args.vision_max_patches,
            max_target_tokens=args.vision_max_target_tokens,
        )
        vision_evaluation = evaluate_vn97_vision(
            loaded.model,
            loaded.vision_adapter,
            runtime_tokenizer,
            vision_examples,
            config=vision_config,
            device=args.validation_device,
        )
        require_vision_release_quality(
            vision_evaluation,
            VN97VisionReleaseCriteria(
                max_validation_loss=args.max_vision_validation_loss,
                min_top1_accuracy=args.min_vision_validation_accuracy,
                min_target_tokens=args.min_vision_validation_target_tokens,
                min_examples=args.min_vision_validation_examples,
            ),
        )
    elif (
        args.vision_validation_input is not None
        or vision_training_report_path is not None
    ):
        raise ValueError(
            "vision release inputs were provided but VN97CK1 has no vision adapter"
        )

    footprint = estimate_vn97_mobile_footprint(
        loaded.config,
        tokenizer_nbytes=len(tokenizer_bytes),
        audio_frame_size=(
            None
            if loaded.audio_adapter is None
            else loaded.audio_adapter.config.frame_size
        ),
        vision_input_features=(
            None
            if loaded.vision_adapter is None
            else loaded.vision_adapter.input_features
        ),
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        batch_size=1,
    )
    mobile_budget = VN97MobileBudget(
        max_model_image_bytes=args.max_model_image_bytes,
        max_recurrent_state_bytes=args.max_recurrent_state_bytes,
    )
    rejection = mobile_budget.rejection_status(footprint)
    if rejection is not None:
        raise ValueError(
            "VN97 production intelligence exceeds mobile release budget: "
            + rejection
        )

    preview_image = build_model_image(
        loaded.model,
        tokenizer=tokenizer,
        audio_adapter=loaded.audio_adapter,
        vision_adapter=loaded.vision_adapter,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
    )
    preview_model_image_sha256 = hashlib.sha256(
        preview_image.data
    ).hexdigest()

    production_campaign_report_sha256 = None
    if (
        args.require_production_campaign_report
        and production_campaign_report_path is None
    ):
        raise ValueError(
            "--require-production-campaign-report needs "
            "--production-campaign-report"
        )
    if production_campaign_report_path is not None:
        production_campaign_report_sha256 = (
            _load_production_campaign_report(
                production_campaign_report_path,
                checkpoint_sha256=loaded.checkpoint_sha256,
                tokenizer_sha256=tokenizer_sha256,
                model_image_sha256=preview_model_image_sha256,
            )
        )

    if release_candidate is not None:
        manifest = release_candidate.manifest
        if (
            preview_model_image_sha256
            != manifest.model_image_sha256
        ):
            raise ValueError(
                "M10N reconstructed VN97MI1 does not match VN97RC1"
            )
        if (
            manifest.speech_enabled
            != (loaded.audio_adapter is not None)
        ):
            raise ValueError(
                "VN97RC1 speech modality flag does not match checkpoint"
            )
        if (
            manifest.vision_enabled
            != (loaded.vision_adapter is not None)
        ):
            raise ValueError(
                "VN97RC1 vision modality flag does not match checkpoint"
            )
        if (
            production_campaign_report_sha256
            != manifest.production_campaign_report_sha256
        ):
            raise ValueError(
                "M10N production report identity does not match VN97RC1"
            )
        if (
            speech_training_report_sha256
            != manifest.speech_training_report_sha256
        ):
            raise ValueError(
                "M10N speech report identity does not match VN97RC1"
            )
        if (
            vision_training_report_sha256
            != manifest.vision_training_report_sha256
        ):
            raise ValueError(
                "M10N vision report identity does not match VN97RC1"
            )

    device_evidence_criteria = VN97DeviceEvidenceCriteria(
        min_runs=args.device_evidence_min_runs,
        max_text_prefill_p95_ms=args.max_text_prefill_p95_ms,
        max_text_decode_p95_ms_per_token=(
            args.max_text_decode_p95_ms_per_token
        ),
        max_peak_pss_kib=args.max_device_peak_pss_kib,
        max_thermal_status=args.max_device_thermal_status,
        max_speech_prefill_p95_ms=(
            args.max_speech_prefill_p95_ms
            if loaded.audio_adapter is not None
            else None
        ),
        require_energy_counter=args.require_device_energy_counter,
        max_abs_battery_energy_counter_delta_nwh=(
            args.max_abs_battery_energy_counter_delta_nwh
        ),
    )
    if (
        args.require_device_evidence
        and not device_evidence_paths
    ):
        raise ValueError(
            "--require-device-evidence needs device evidence"
        )
    verified_device_evidence = []
    for path in device_evidence_paths:
        evidence = load_device_evidence(path)
        require_device_evidence(
            evidence,
            device_evidence_criteria,
            expected_model_image_sha256=
                preview_model_image_sha256,
            speech_enabled=
                loaded.audio_adapter is not None,
        )
        verified_device_evidence.append(
            evidence
        )

    if release_candidate is not None:
        expected_evidence = sorted(
            item.evidence_sha256
            for item in
                release_candidate
                .manifest
                .device_evidence
        )
        actual_evidence = sorted(
            item.evidence_sha256
            for item in
                verified_device_evidence
        )
        if actual_evidence != expected_evidence:
            raise ValueError(
                "M10N device evidence identities do not match VN97RC1"
            )

    device_evidence = (
        verified_device_evidence[0]
        if (
            release_candidate is None
            and len(verified_device_evidence) == 1
        )
        else None
    )

    # Private signing material is not opened until quality, mobile-budget and
    # optional on-device evidence gates pass.
    private_key = _parse_private_key(
        _read_regular_file(
            private_key_path,
            max_bytes=_MAX_PRIVATE_KEY_FILE_BYTES,
            label="Ed25519 private key",
        )
    )
    signer = Ed25519PrivateKeySigner(
        args.key_id,
        private_key,
    )
    source = CapabilitySource(
        args.source_origin,
        loaded.checkpoint_sha256,
        args.source_license,
    )

    bundle = build_bootstrap_bundle(
        loaded.model,
        tokenizer=tokenizer,
        audio_adapter=loaded.audio_adapter,
        vision_adapter=loaded.vision_adapter,
        source=source,
        capability_version=args.capability_version,
        signer=signer,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
    )
    if bundle.model_image_sha256 != preview_model_image_sha256:
        raise RuntimeError(
            "VN97MI1 identity changed between evidence gate and signing"
        )
    output = write_bootstrap_assets(bundle, assets_dir)

    report = {
        "assets_dir": str(output),
        "capability_version": bundle.capability_version,
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "device_evidence": (
            None
            if device_evidence is None
            else {
                "abi": device_evidence.abi,
                "battery_energy_counter_delta_nwh": (
                    device_evidence.battery_energy_counter_delta_nwh
                ),
                "evidence_sha256": device_evidence.evidence_sha256,
                "manufacturer": device_evidence.manufacturer,
                "max_thermal_status": (
                    device_evidence_criteria.max_thermal_status
                ),
                "model": device_evidence.model,
                "peak_pss_kib": device_evidence.peak_pss_kib,
                "runs": device_evidence.runs,
                "sdk_int": device_evidence.sdk_int,
                "speech_prefill_p95_ms": (
                    None
                    if device_evidence.speech_prefill is None
                    else device_evidence.speech_prefill.p95_ms
                ),
                "text_decode_p95_ms_per_token": (
                    device_evidence.text_decode_per_token.p95_ms
                ),
                "text_prefill_p95_ms": (
                    device_evidence.text_prefill.p95_ms
                ),
                "thermal_status_max": (
                    device_evidence.thermal_status_max
                ),
            }
        ),
        "model_image_sha256": bundle.model_image_sha256,
        "mobile_budget": {
            "max_model_image_bytes": mobile_budget.max_model_image_bytes,
            "max_recurrent_state_bytes": mobile_budget.max_recurrent_state_bytes,
        },
        "mobile_footprint": footprint.canonical_object(),
        "package_sha256": bundle.package_sha256,
        "publisher_key_id": bundle.publisher_key_id,
        "publisher_public_key_sha256": hashlib.sha256(
            bundle.publisher_public_key
        ).hexdigest(),
        "production_campaign_report_sha256": (
            production_campaign_report_sha256
        ),
        "release_candidate": (
            None
            if release_candidate is None
            else {
                "device_evidence_sha256": sorted(
                    item.evidence_sha256
                    for item in
                        verified_device_evidence
                ),
                "manifest_sha256":
                    release_candidate.manifest_sha256,
                "selected_candidate_id":
                    release_candidate
                    .manifest
                    .selected_candidate_id,
                "tile_cols":
                    release_candidate
                    .manifest
                    .tile_cols,
                "tile_rows":
                    release_candidate
                    .manifest
                    .tile_rows,
            }
        ),
        "schema": "VN97BOOTREL6",
        "speech_enabled": loaded.audio_adapter is not None,
        "vision_enabled": loaded.vision_adapter is not None,
        "vision_runtime_budget": (
            None
            if loaded.vision_adapter is None
            else {
                "max_patches": args.vision_max_patches,
                "max_generated_tokens": args.vision_max_target_tokens,
                "max_recurrent_steps": (
                    1
                    + args.vision_max_patches
                    + args.vision_max_target_tokens
                ),
            }
        ),
        "vision_training_report_sha256": vision_training_report_sha256,
        "vision_validation": (
            None
            if vision_evaluation is None
            else {
                "dataset_sha256": vision_dataset_sha256,
                "examples": vision_evaluation.examples,
                "max_loss": args.max_vision_validation_loss,
                "mean_loss": vision_evaluation.mean_loss,
                "min_target_tokens": args.min_vision_validation_target_tokens,
                "min_top1_accuracy": args.min_vision_validation_accuracy,
                "target_tokens": vision_evaluation.target_tokens,
                "top1_accuracy": vision_evaluation.top1_accuracy,
            }
        ),
        "speech_runtime_budget": (
            None
            if loaded.audio_adapter is None
            else {
                "max_audio_frames": args.speech_max_frames,
                "max_generated_tokens": args.speech_max_target_tokens,
                "max_recurrent_steps": (
                    1
                    + args.speech_max_frames
                    + args.speech_max_target_tokens
                ),
            }
        ),
        "tokenizer_sha256": tokenizer_sha256,
        "validation": {
            "dataset_sha256": validation_sha256,
            "examples": len(validation_examples),
            "max_loss": criteria.max_validation_loss,
            "mean_loss": evaluation.mean_loss,
            "min_target_tokens": criteria.min_target_tokens,
            "min_top1_accuracy": criteria.min_top1_accuracy,
            "target_tokens": evaluation.target_tokens,
            "top1_accuracy": evaluation.top1_accuracy,
            "windows": evaluation.windows,
        },
        "speech_training_report_sha256": speech_training_report_sha256,
        "speech_validation": (
            None
            if speech_evaluation is None
            else {
                "dataset_sha256": speech_dataset_sha256,
                "examples": speech_evaluation.examples,
                "max_loss": args.max_speech_validation_loss,
                "mean_loss": speech_evaluation.mean_loss,
                "min_target_tokens": args.min_speech_validation_target_tokens,
                "min_top1_accuracy": args.min_speech_validation_accuracy,
                "target_tokens": speech_evaluation.target_tokens,
                "top1_accuracy": speech_evaluation.top1_accuracy,
            }
        ),
    }
    print(_canonical_json(report))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-bootstrap-release: {exc}", file=sys.stderr)
        raise
