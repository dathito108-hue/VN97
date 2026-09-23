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
from .tokenizer import VN97Tokenizer, VN97TokenizerPackage
from .training import (
    VN97TrainingConfig,
    build_training_windows,
    encode_causal_text,
    encode_chat_messages,
)
from .training_cli import _load_records


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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
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

    parser.add_argument("--tile-rows", type=int, default=16)
    parser.add_argument("--tile-cols", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    checkpoint_path = Path(args.checkpoint)
    tokenizer_path = Path(args.tokenizer)
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
        if args.speech_training_report is None:
            raise ValueError(
                "speech-enabled VN97CK1 requires --speech-training-report"
            )
        speech_training_report, speech_training_report_sha256 = (
            _load_speech_training_report(
                Path(args.speech_training_report),
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
        or args.speech_training_report is not None
    ):
        raise ValueError(
            "speech release inputs were provided but VN97CK1 has no audio adapter"
        )

    # Private signing material is not opened until all held-out quality gates pass.
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
        source=source,
        capability_version=args.capability_version,
        signer=signer,
        tile_rows=args.tile_rows,
        tile_cols=args.tile_cols,
    )
    output = write_bootstrap_assets(bundle, assets_dir)

    report = {
        "assets_dir": str(output),
        "capability_version": bundle.capability_version,
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "model_image_sha256": bundle.model_image_sha256,
        "package_sha256": bundle.package_sha256,
        "publisher_key_id": bundle.publisher_key_id,
        "publisher_public_key_sha256": hashlib.sha256(
            bundle.publisher_public_key
        ).hexdigest(),
        "schema": "VN97BOOTREL3",
        "speech_enabled": loaded.audio_adapter is not None,
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
