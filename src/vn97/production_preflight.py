from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .production_run_manifest import (
    VN97ProductionRunManifest,
    VN97ResolvedProductionRun,
    options_to_argv,
)


SCHEMA = "VN97PREFLIGHT1"
MAX_REPORT_BYTES = 1024 * 1024
_HEX64 = set("0123456789abcdef")
_CANDIDATE_STATUSES = {
    "ADMITTED",
    "REJECTED_PARAMETER_BUDGET",
    "REJECTED_LANGUAGE_MODEL_IMAGE_BUDGET",
    "REJECTED_LANGUAGE_RECURRENT_STATE_BUDGET",
    "REJECTED_SPEECH_MODEL_IMAGE_BUDGET",
    "REJECTED_SPEECH_RECURRENT_STATE_BUDGET",
}


class VN97ProductionPreflightError(RuntimeError):
    pass


def _require_sha256(
    value: object,
    *,
    label: str,
) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in _HEX64 for ch in value)
    ):
        raise VN97ProductionPreflightError(
            f"{label} must be lowercase SHA-256"
        )
    return value


@dataclass(frozen=True, order=True)
class VN97PreflightBlocker:
    code: str
    message: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.code, "blocker code"),
            (self.message, "blocker message"),
        ):
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 512
                or any(ord(ch) < 0x20 for ch in value)
            ):
                raise VN97ProductionPreflightError(
                    f"{label} is invalid"
                )

    def canonical_object(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True)
class VN97PreflightCandidate:
    candidate_id: str
    parameter_count: int
    language_model_image_bytes: int
    language_recurrent_state_bytes: int
    speech_model_image_bytes: int
    speech_recurrent_state_bytes: int
    status: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.candidate_id, str)
            or len(self.candidate_id) != 16
            or any(
                ch not in "0123456789abcdef"
                for ch in self.candidate_id
            )
        ):
            raise VN97ProductionPreflightError(
                "preflight candidate id is invalid"
            )
        for value, label in (
            (self.parameter_count, "parameter_count"),
            (
                self.language_model_image_bytes,
                "language_model_image_bytes",
            ),
            (
                self.language_recurrent_state_bytes,
                "language_recurrent_state_bytes",
            ),
            (
                self.speech_model_image_bytes,
                "speech_model_image_bytes",
            ),
            (
                self.speech_recurrent_state_bytes,
                "speech_recurrent_state_bytes",
            ),
        ):
            if type(value) is not int or value <= 0:
                raise VN97ProductionPreflightError(
                    f"{label} must be positive integer"
                )
        if self.status not in _CANDIDATE_STATUSES:
            raise VN97ProductionPreflightError(
                "preflight candidate status is invalid"
            )

    def canonical_object(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "language_model_image_bytes":
                self.language_model_image_bytes,
            "language_recurrent_state_bytes":
                self.language_recurrent_state_bytes,
            "parameter_count": self.parameter_count,
            "speech_model_image_bytes":
                self.speech_model_image_bytes,
            "speech_recurrent_state_bytes":
                self.speech_recurrent_state_bytes,
            "status": self.status,
        }


@dataclass(frozen=True)
class VN97ProductionPreflightReport:
    manifest_sha256: str
    status: str
    tokenizer_sha256: str
    tokenizer_bytes: int
    vocab_size: int
    training_dataset_sha256: str
    validation_dataset_sha256: str
    release_dataset_sha256: str
    training_records: int
    validation_records: int
    release_records: int
    training_windows: int
    validation_windows: int
    release_windows: int
    validation_target_tokens: int
    release_target_tokens: int
    speech_training_dataset_sha256: str
    speech_validation_dataset_sha256: str
    speech_release_dataset_sha256: str
    speech_training_examples: int
    speech_validation_examples: int
    speech_release_examples: int
    speech_validation_target_tokens: int
    speech_release_target_tokens: int
    speech_max_observed_frames: int
    candidates: tuple[VN97PreflightCandidate, ...]
    blockers: tuple[VN97PreflightBlocker, ...]

    def __post_init__(self) -> None:
        _require_sha256(
            self.manifest_sha256,
            label="manifest SHA-256",
        )
        for value, label in (
            (self.tokenizer_sha256, "tokenizer SHA-256"),
            (
                self.training_dataset_sha256,
                "training dataset SHA-256",
            ),
            (
                self.validation_dataset_sha256,
                "validation dataset SHA-256",
            ),
            (
                self.release_dataset_sha256,
                "release dataset SHA-256",
            ),
            (
                self.speech_training_dataset_sha256,
                "speech training dataset SHA-256",
            ),
            (
                self.speech_validation_dataset_sha256,
                "speech validation dataset SHA-256",
            ),
            (
                self.speech_release_dataset_sha256,
                "speech release dataset SHA-256",
            ),
        ):
            _require_sha256(value, label=label)

        for value, label in (
            (self.tokenizer_bytes, "tokenizer_bytes"),
            (self.vocab_size, "vocab_size"),
            (self.training_records, "training_records"),
            (self.validation_records, "validation_records"),
            (self.release_records, "release_records"),
            (self.training_windows, "training_windows"),
            (self.validation_windows, "validation_windows"),
            (self.release_windows, "release_windows"),
            (
                self.validation_target_tokens,
                "validation_target_tokens",
            ),
            (
                self.release_target_tokens,
                "release_target_tokens",
            ),
            (
                self.speech_training_examples,
                "speech_training_examples",
            ),
            (
                self.speech_validation_examples,
                "speech_validation_examples",
            ),
            (
                self.speech_release_examples,
                "speech_release_examples",
            ),
            (
                self.speech_validation_target_tokens,
                "speech_validation_target_tokens",
            ),
            (
                self.speech_release_target_tokens,
                "speech_release_target_tokens",
            ),
            (
                self.speech_max_observed_frames,
                "speech_max_observed_frames",
            ),
        ):
            if type(value) is not int or value <= 0:
                raise VN97ProductionPreflightError(
                    f"{label} must be positive integer"
                )

        if self.status not in {"READY", "BLOCKED"}:
            raise VN97ProductionPreflightError(
                "preflight status is invalid"
            )
        if (
            not self.candidates
            or tuple(
                sorted(
                    self.candidates,
                    key=lambda item: item.candidate_id,
                )
            )
            != self.candidates
        ):
            raise VN97ProductionPreflightError(
                "preflight candidates must be non-empty and sorted"
            )
        if tuple(sorted(self.blockers)) != self.blockers:
            raise VN97ProductionPreflightError(
                "preflight blockers must be sorted"
            )
        ready = (
            not self.blockers
            and any(
                item.status == "ADMITTED"
                for item in self.candidates
            )
        )
        if (self.status == "READY") != ready:
            raise VN97ProductionPreflightError(
                "preflight status does not match blockers/candidates"
            )

    @property
    def ready(self) -> bool:
        return self.status == "READY"

    def canonical_object(self) -> dict[str, object]:
        return {
            "blockers": [
                item.canonical_object()
                for item in self.blockers
            ],
            "candidates": [
                item.canonical_object()
                for item in self.candidates
            ],
            "language": {
                "release_dataset_sha256":
                    self.release_dataset_sha256,
                "release_records":
                    self.release_records,
                "release_target_tokens":
                    self.release_target_tokens,
                "release_windows":
                    self.release_windows,
                "tokenizer_bytes":
                    self.tokenizer_bytes,
                "tokenizer_sha256":
                    self.tokenizer_sha256,
                "training_dataset_sha256":
                    self.training_dataset_sha256,
                "training_records":
                    self.training_records,
                "training_windows":
                    self.training_windows,
                "validation_dataset_sha256":
                    self.validation_dataset_sha256,
                "validation_records":
                    self.validation_records,
                "validation_target_tokens":
                    self.validation_target_tokens,
                "validation_windows":
                    self.validation_windows,
                "vocab_size":
                    self.vocab_size,
            },
            "manifest_sha256":
                self.manifest_sha256,
            "schema": SCHEMA,
            "speech": {
                "max_observed_frames":
                    self.speech_max_observed_frames,
                "release_dataset_sha256":
                    self.speech_release_dataset_sha256,
                "release_examples":
                    self.speech_release_examples,
                "release_target_tokens":
                    self.speech_release_target_tokens,
                "training_dataset_sha256":
                    self.speech_training_dataset_sha256,
                "training_examples":
                    self.speech_training_examples,
                "validation_dataset_sha256":
                    self.speech_validation_dataset_sha256,
                "validation_examples":
                    self.speech_validation_examples,
                "validation_target_tokens":
                    self.speech_validation_target_tokens,
            },
            "status": self.status,
        }

    def to_bytes(self) -> bytes:
        return json.dumps(
            self.canonical_object(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")


def _language_argv(
    manifest: VN97ProductionRunManifest,
    resolved: VN97ResolvedProductionRun,
) -> list[str]:
    argv: list[str] = []
    for path in resolved.language_training:
        argv.extend(["--input", str(path)])
    for path in resolved.language_validation:
        argv.extend(
            ["--validation-input", str(path)]
        )
    for path in resolved.language_release:
        argv.extend(
            ["--release-input", str(path)]
        )
    argv.extend(
        [
            "--campaign",
            str(resolved.campaign_definition),
            "--output-dir",
            str(resolved.language_output_dir),
        ]
    )
    argv.extend(
        options_to_argv(
            manifest.language_options
        )
    )
    return argv


def _speech_argv(
    manifest: VN97ProductionRunManifest,
    resolved: VN97ResolvedProductionRun,
) -> list[str]:
    argv = [
        "--language-campaign-dir",
        str(resolved.language_output_dir),
        "--speech-input",
        str(resolved.speech_training_manifest),
        "--speech-validation-input",
        str(resolved.speech_validation_manifest),
        "--speech-release-input",
        str(resolved.speech_release_manifest),
        "--output-dir",
        str(resolved.production_output_dir),
    ]
    argv.extend(
        options_to_argv(
            manifest.speech_options
        )
    )
    return argv


def _device_blocker(
    value: str,
    *,
    label: str,
) -> VN97PreflightBlocker | None:
    import torch

    try:
        device = torch.device(value)
    except Exception:
        return VN97PreflightBlocker(
            code=f"device.{label}.invalid",
            message=f"{label} device string is invalid: {value}",
        )

    if device.type == "cpu":
        return None
    if device.type == "cuda":
        if not torch.cuda.is_available():
            return VN97PreflightBlocker(
                code=f"device.{label}.unavailable",
                message=f"{label} CUDA device is unavailable",
            )
        index = 0 if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            return VN97PreflightBlocker(
                code=f"device.{label}.index",
                message=f"{label} CUDA device index is unavailable",
            )
        return None
    if device.type == "mps":
        backend = getattr(
            getattr(torch, "backends", None),
            "mps",
            None,
        )
        if (
            backend is None
            or not backend.is_available()
        ):
            return VN97PreflightBlocker(
                code=f"device.{label}.unavailable",
                message=f"{label} MPS device is unavailable",
            )
        return None
    xpu = getattr(torch, "xpu", None)
    if device.type == "xpu":
        if xpu is None or not xpu.is_available():
            return VN97PreflightBlocker(
                code=f"device.{label}.unavailable",
                message=f"{label} XPU device is unavailable",
            )
        return None

    return VN97PreflightBlocker(
        code=f"device.{label}.unsupported",
        message=(
            f"{label} device type is not zero-compute-preflight aware: "
            f"{device.type}"
        ),
    )


def _speech_target_tokens(
    tokenizer,
    examples,
    *,
    max_target_tokens: int,
    frame_size: int,
    hop_size: int,
    max_frames: int,
) -> tuple[int, int]:
    total_targets = 0
    observed_max_frames = 0
    for example in examples:
        content = tokenizer.encode(
            example.transcript
        )
        target_count = len(content) + 2
        if target_count > max_target_tokens:
            raise VN97ProductionPreflightError(
                "speech transcript exceeds configured max_target_tokens"
            )
        samples = int(
            example.waveform.numel()
        )
        if samples < frame_size:
            raise VN97ProductionPreflightError(
                "speech waveform is shorter than one audio frame"
            )
        frames = (
            1
            + (
                samples - frame_size
            ) // hop_size
        )
        if frames <= 0 or frames > max_frames:
            raise VN97ProductionPreflightError(
                "speech example frame count is outside configured bounds"
            )
        total_targets += target_count
        observed_max_frames = max(
            observed_max_frames,
            frames,
        )
    return (
        total_targets,
        observed_max_frames,
    )


def run_production_preflight(
    manifest: VN97ProductionRunManifest,
    resolved: VN97ResolvedProductionRun,
) -> VN97ProductionPreflightReport:
    # Heavy runtime modules are imported only when the real preflight is
    # requested. Importing/parsing VN97PREFLIGHT1 remains stdlib-only.
    from .campaign_cli import (
        _estimate_parameter_count,
        _examples,
        _load_candidates,
        _parser as campaign_parser,
    )
    from .config import VN97Config
    from .dataset_split import (
        require_disjoint_dataset_splits,
    )
    from .evaluation import (
        VN97ReleaseCriteria,
    )
    from .mobile_budget import (
        VN97MobileBudget,
        estimate_vn97_mobile_footprint,
    )
    from .modality import AudioAdapterConfig
    from .production_campaign_cli import (
        _parser as production_parser,
    )
    from .speech_training import (
        VN97SpeechReleaseCriteria,
        VN97SpeechTrainingConfig,
        require_disjoint_speech_splits,
    )
    from .speech_training_cli import (
        load_speech_manifest,
    )
    from .tokenizer import (
        VN97Tokenizer,
        learn_byte_bpe,
    )
    from .training import (
        VN97TrainingConfig,
        build_training_windows,
        render_chat_text,
    )
    from .training_cli import _load_records

    language_args = (
        campaign_parser()
        .parse_args(
            _language_argv(
                manifest,
                resolved,
            )
        )
    )
    speech_args = (
        production_parser()
        .parse_args(
            _speech_argv(
                manifest,
                resolved,
            )
        )
    )

    blockers: list[
        VN97PreflightBlocker
    ] = []
    for value, label in (
        (language_args.device, "language"),
        (speech_args.device, "speech"),
    ):
        blocker = _device_blocker(
            value,
            label=label,
        )
        if blocker is not None:
            blockers.append(blocker)

    train_records, train_sha = (
        _load_records(
            list(
                resolved
                .language_training
            ),
            mode=language_args.format,
            max_input_bytes=
                language_args.max_input_bytes,
            max_examples=
                language_args.max_examples,
        )
    )
    validation_mode = (
        language_args.validation_format
        or language_args.format
    )
    validation_records, validation_sha = (
        _load_records(
            list(
                resolved
                .language_validation
            ),
            mode=validation_mode,
            max_input_bytes=
                language_args
                .validation_max_input_bytes,
            max_examples=
                language_args
                .validation_max_examples,
        )
    )
    release_mode = (
        language_args.release_format
        or validation_mode
    )
    release_records, release_sha = (
        _load_records(
            list(
                resolved
                .language_release
            ),
            mode=release_mode,
            max_input_bytes=
                language_args
                .release_max_input_bytes,
            max_examples=
                language_args
                .release_max_examples,
        )
    )
    require_disjoint_dataset_splits(
        train_records,
        training_mode=
            language_args.format,
        validation_records=
            validation_records,
        validation_mode=
            validation_mode,
        release_records=release_records,
        release_mode=release_mode,
    )

    corpus = (
        train_records
        if language_args.format == "text"
        else [
            render_chat_text(value)
            for value in train_records
        ]
    )
    tokenizer_package = (
        learn_byte_bpe(
            corpus,
            max_learned_tokens=
                language_args
                .learned_tokens,
            min_pair_count=
                language_args
                .min_pair_count,
        )
    )
    tokenizer = VN97Tokenizer(
        tokenizer_package
    )
    tokenizer_bytes = (
        tokenizer_package.to_bytes()
    )
    tokenizer_sha = hashlib.sha256(
        tokenizer_bytes
    ).hexdigest()

    train_examples = _examples(
        train_records,
        mode=language_args.format,
        tokenizer=tokenizer,
    )
    validation_examples = _examples(
        validation_records,
        mode=validation_mode,
        tokenizer=tokenizer,
    )
    release_examples = _examples(
        release_records,
        mode=release_mode,
        tokenizer=tokenizer,
    )

    train_config = VN97TrainingConfig(
        sequence_length=
            language_args.sequence_length,
        stride=language_args.stride,
        batch_size=
            language_args.batch_size,
        epochs=language_args.epochs,
        learning_rate=1e-4,
        weight_decay=
            language_args.weight_decay,
        max_grad_norm=
            language_args.max_grad_norm,
        seed=0,
        shuffle=False,
        max_windows=
            language_args.max_windows,
    )
    validation_config = VN97TrainingConfig(
        sequence_length=
            language_args.sequence_length,
        stride=language_args.stride,
        batch_size=
            language_args
            .validation_batch_size,
        epochs=1,
        seed=0,
        shuffle=False,
        max_windows=
            language_args
            .validation_max_windows,
    )
    release_config = VN97TrainingConfig(
        sequence_length=
            language_args.sequence_length,
        stride=language_args.stride,
        batch_size=
            language_args
            .release_batch_size,
        epochs=1,
        seed=0,
        shuffle=False,
        max_windows=
            language_args
            .release_max_windows,
    )
    train_windows = (
        build_training_windows(
            train_examples,
            train_config,
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
    release_windows = (
        build_training_windows(
            release_examples,
            release_config,
            pad_token_id=
                tokenizer.pad_id,
        )
    )
    validation_targets = sum(
        item.target_tokens
        for item in validation_windows
    )
    release_targets = sum(
        item.target_tokens
        for item in release_windows
    )

    validation_criteria = (
        VN97ReleaseCriteria(
            max_validation_loss=
                language_args
                .max_validation_loss,
            min_top1_accuracy=
                language_args
                .min_validation_accuracy,
            min_target_tokens=
                language_args
                .min_validation_target_tokens,
        )
    )
    release_criteria = (
        VN97ReleaseCriteria(
            max_validation_loss=(
                validation_criteria
                .max_validation_loss
                if language_args
                .release_max_validation_loss
                is None
                else language_args
                .release_max_validation_loss
            ),
            min_top1_accuracy=(
                validation_criteria
                .min_top1_accuracy
                if language_args
                .release_min_validation_accuracy
                is None
                else language_args
                .release_min_validation_accuracy
            ),
            min_target_tokens=(
                validation_criteria
                .min_target_tokens
                if language_args
                .release_min_validation_target_tokens
                is None
                else language_args
                .release_min_validation_target_tokens
            ),
        )
    )
    if (
        validation_targets
        < validation_criteria
        .min_target_tokens
    ):
        blockers.append(
            VN97PreflightBlocker(
                code=
                    "language.validation_target_tokens_impossible",
                message=(
                    "validation supervised target-token count is below "
                    "the configured release-quality minimum"
                ),
            )
        )
    if (
        release_targets
        < release_criteria
        .min_target_tokens
    ):
        blockers.append(
            VN97PreflightBlocker(
                code=
                    "language.release_target_tokens_impossible",
                message=(
                    "sealed release supervised target-token count is below "
                    "the configured release-quality minimum"
                ),
            )
        )

    speech_training_config = (
        VN97SpeechTrainingConfig(
            epochs=
                speech_args.speech_epochs,
            learning_rate=
                speech_args
                .speech_learning_rate,
            weight_decay=
                speech_args
                .speech_weight_decay,
            max_grad_norm=
                speech_args
                .speech_max_grad_norm,
            seed=
                speech_args.speech_seed,
            shuffle=True,
            max_frames=
                speech_args
                .speech_max_frames,
            max_target_tokens=
                speech_args
                .speech_max_target_tokens,
        )
    )

    speech_training, speech_train_sha = (
        load_speech_manifest(
            resolved
            .speech_training_manifest,
            max_examples=
                speech_args
                .speech_max_examples,
        )
    )
    speech_validation, speech_validation_sha = (
        load_speech_manifest(
            resolved
            .speech_validation_manifest,
            max_examples=
                speech_args
                .speech_max_examples,
        )
    )
    speech_release, speech_release_sha = (
        load_speech_manifest(
            resolved
            .speech_release_manifest,
            max_examples=
                speech_args
                .speech_max_examples,
        )
    )
    require_disjoint_speech_splits(
        speech_training,
        speech_validation,
        speech_release,
    )

    audio_config = AudioAdapterConfig()
    speech_train_targets, train_max_frames = (
        _speech_target_tokens(
            tokenizer,
            speech_training,
            max_target_tokens=
                speech_args
                .speech_max_target_tokens,
            frame_size=
                audio_config.frame_size,
            hop_size=
                audio_config.hop_size,
            max_frames=
                speech_args
                .speech_max_frames,
        )
    )
    speech_validation_targets, validation_max_frames = (
        _speech_target_tokens(
            tokenizer,
            speech_validation,
            max_target_tokens=
                speech_args
                .speech_max_target_tokens,
            frame_size=
                audio_config.frame_size,
            hop_size=
                audio_config.hop_size,
            max_frames=
                speech_args
                .speech_max_frames,
        )
    )
    speech_release_targets, release_max_frames = (
        _speech_target_tokens(
            tokenizer,
            speech_release,
            max_target_tokens=
                speech_args
                .speech_max_target_tokens,
            frame_size=
                audio_config.frame_size,
            hop_size=
                audio_config.hop_size,
            max_frames=
                speech_args
                .speech_max_frames,
        )
    )
    if speech_train_targets <= 0:
        raise VN97ProductionPreflightError(
            "speech training produced no target tokens"
        )

    speech_validation_criteria = (
        VN97SpeechReleaseCriteria(
            max_validation_loss=
                speech_args
                .max_speech_validation_loss,
            min_top1_accuracy=
                speech_args
                .min_speech_validation_accuracy,
            min_target_tokens=
                speech_args
                .min_speech_validation_target_tokens,
            min_examples=
                speech_args
                .min_speech_validation_examples,
        )
    )
    speech_release_criteria = (
        VN97SpeechReleaseCriteria(
            max_validation_loss=(
                speech_validation_criteria
                .max_validation_loss
                if speech_args
                .release_max_speech_loss
                is None
                else speech_args
                .release_max_speech_loss
            ),
            min_top1_accuracy=(
                speech_validation_criteria
                .min_top1_accuracy
                if speech_args
                .release_min_speech_accuracy
                is None
                else speech_args
                .release_min_speech_accuracy
            ),
            min_target_tokens=(
                speech_validation_criteria
                .min_target_tokens
                if speech_args
                .release_min_speech_target_tokens
                is None
                else speech_args
                .release_min_speech_target_tokens
            ),
            min_examples=(
                speech_validation_criteria
                .min_examples
                if speech_args
                .release_min_speech_examples
                is None
                else speech_args
                .release_min_speech_examples
            ),
        )
    )
    if (
        len(speech_validation)
        < speech_validation_criteria.min_examples
    ):
        blockers.append(
            VN97PreflightBlocker(
                code=
                    "speech.validation_examples_impossible",
                message=(
                    "speech validation example count is below the "
                    "configured release-quality minimum"
                ),
            )
        )
    if (
        speech_validation_targets
        < speech_validation_criteria
        .min_target_tokens
    ):
        blockers.append(
            VN97PreflightBlocker(
                code=
                    "speech.validation_target_tokens_impossible",
                message=(
                    "speech validation target-token count is below the "
                    "configured release-quality minimum"
                ),
            )
        )
    if (
        len(speech_release)
        < speech_release_criteria.min_examples
    ):
        blockers.append(
            VN97PreflightBlocker(
                code=
                    "speech.release_examples_impossible",
                message=(
                    "sealed speech release example count is below the "
                    "configured release-quality minimum"
                ),
            )
        )
    if (
        speech_release_targets
        < speech_release_criteria
        .min_target_tokens
    ):
        blockers.append(
            VN97PreflightBlocker(
                code=
                    "speech.release_target_tokens_impossible",
                message=(
                    "sealed speech release target-token count is below "
                    "the configured release-quality minimum"
                ),
            )
        )

    language_budget = VN97MobileBudget(
        max_model_image_bytes=
            language_args
            .max_model_image_bytes,
        max_recurrent_state_bytes=
            language_args
            .max_recurrent_state_bytes,
    )
    speech_budget = VN97MobileBudget(
        max_model_image_bytes=
            speech_args
            .max_model_image_bytes,
        max_recurrent_state_bytes=
            speech_args
            .max_recurrent_state_bytes,
    )

    rows: list[
        VN97PreflightCandidate
    ] = []
    potential_winner_final_rejections: list[
        str
    ] = []
    candidates = _load_candidates(
        Path(language_args.campaign)
    )
    for candidate in candidates:
        config = VN97Config(
            vocab_size=
                tokenizer.vocab_size,
            d_model=candidate.d_model,
            n_layers=candidate.n_layers,
            d_state=candidate.d_state,
            embedding_rank=
                candidate.embedding_rank,
        )
        parameter_count = (
            _estimate_parameter_count(
                vocab_size=
                    tokenizer.vocab_size,
                candidate=candidate,
            )
        )
        language_footprint = (
            estimate_vn97_mobile_footprint(
                config,
                tokenizer_nbytes=
                    len(tokenizer_bytes),
                tile_rows=
                    language_args
                    .deployment_tile_rows,
                tile_cols=
                    language_args
                    .deployment_tile_cols,
            )
        )
        speech_footprint = (
            estimate_vn97_mobile_footprint(
                config,
                tokenizer_nbytes=
                    len(tokenizer_bytes),
                audio_frame_size=
                    audio_config.frame_size,
                tile_rows=
                    speech_args.tile_rows,
                tile_cols=
                    speech_args.tile_cols,
                batch_size=1,
            )
        )

        status = "ADMITTED"
        if (
            parameter_count
            > language_args.max_parameters
        ):
            status = (
                "REJECTED_PARAMETER_BUDGET"
            )
        else:
            rejection = (
                language_budget
                .rejection_status(
                    language_footprint
                )
            )
            if rejection == (
                "REJECTED_MODEL_IMAGE_BUDGET"
            ):
                status = (
                    "REJECTED_LANGUAGE_MODEL_IMAGE_BUDGET"
                )
            elif rejection == (
                "REJECTED_RECURRENT_STATE_BUDGET"
            ):
                status = (
                    "REJECTED_LANGUAGE_RECURRENT_STATE_BUDGET"
                )
            else:
                final_rejection = (
                    speech_budget
                    .rejection_status(
                        speech_footprint
                    )
                )
                if final_rejection == (
                    "REJECTED_MODEL_IMAGE_BUDGET"
                ):
                    status = (
                        "REJECTED_SPEECH_MODEL_IMAGE_BUDGET"
                    )
                    potential_winner_final_rejections.append(
                        candidate.candidate_id
                    )
                elif final_rejection == (
                    "REJECTED_RECURRENT_STATE_BUDGET"
                ):
                    status = (
                        "REJECTED_SPEECH_RECURRENT_STATE_BUDGET"
                    )
                    potential_winner_final_rejections.append(
                        candidate.candidate_id
                    )

        rows.append(
            VN97PreflightCandidate(
                candidate_id=
                    candidate.candidate_id,
                parameter_count=
                    parameter_count,
                language_model_image_bytes=
                    language_footprint
                    .model_image_bytes,
                language_recurrent_state_bytes=
                    language_footprint
                    .recurrent_state_bytes,
                speech_model_image_bytes=
                    speech_footprint
                    .model_image_bytes,
                speech_recurrent_state_bytes=
                    speech_footprint
                    .recurrent_state_bytes,
                status=status,
            )
        )

    if not any(
        item.status
        in {
            "ADMITTED",
            "REJECTED_SPEECH_MODEL_IMAGE_BUDGET",
            "REJECTED_SPEECH_RECURRENT_STATE_BUDGET",
        }
        for item in rows
    ):
        blockers.append(
            VN97PreflightBlocker(
                code=
                    "candidate.no_language_admissible",
                message=(
                    "no candidate can enter language training under "
                    "parameter/mobile budgets"
                ),
            )
        )
    if potential_winner_final_rejections:
        blockers.append(
            VN97PreflightBlocker(
                code=
                    "candidate.language_winner_may_fail_speech_budget",
                message=(
                    "one or more language-admissible candidates would "
                    "exceed final speech-enabled mobile budget: "
                    + ",".join(
                        sorted(
                            potential_winner_final_rejections
                        )
                    )
                ),
            )
        )

    sorted_rows = tuple(
        sorted(
            rows,
            key=lambda item:
                item.candidate_id,
        )
    )
    sorted_blockers = tuple(
        sorted(blockers)
    )
    status = (
        "READY"
        if (
            not sorted_blockers
            and any(
                item.status == "ADMITTED"
                for item in sorted_rows
            )
        )
        else "BLOCKED"
    )
    return VN97ProductionPreflightReport(
        manifest_sha256=
            manifest.manifest_sha256,
        status=status,
        tokenizer_sha256=
            tokenizer_sha,
        tokenizer_bytes=
            len(tokenizer_bytes),
        vocab_size=
            tokenizer.vocab_size,
        training_dataset_sha256=
            train_sha,
        validation_dataset_sha256=
            validation_sha,
        release_dataset_sha256=
            release_sha,
        training_records=
            len(train_records),
        validation_records=
            len(validation_records),
        release_records=
            len(release_records),
        training_windows=
            len(train_windows),
        validation_windows=
            len(validation_windows),
        release_windows=
            len(release_windows),
        validation_target_tokens=
            validation_targets,
        release_target_tokens=
            release_targets,
        speech_training_dataset_sha256=
            speech_train_sha,
        speech_validation_dataset_sha256=
            speech_validation_sha,
        speech_release_dataset_sha256=
            speech_release_sha,
        speech_training_examples=
            len(speech_training),
        speech_validation_examples=
            len(speech_validation),
        speech_release_examples=
            len(speech_release),
        speech_validation_target_tokens=
            speech_validation_targets,
        speech_release_target_tokens=
            speech_release_targets,
        speech_max_observed_frames=
            max(
                train_max_frames,
                validation_max_frames,
                release_max_frames,
            ),
        candidates=sorted_rows,
        blockers=sorted_blockers,
    )


def parse_production_preflight_report(
    data: bytes,
) -> VN97ProductionPreflightReport:
    if not 0 < len(data) <= MAX_REPORT_BYTES:
        raise VN97ProductionPreflightError(
            "VN97PREFLIGHT1 byte size is outside bounds"
        )
    duplicates: list[str] = []

    def hook(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                duplicates.append(key)
            out[key] = value
        return out

    try:
        text = data.decode(
            "utf-8",
            errors="strict",
        )
        root = json.loads(
            text,
            object_pairs_hook=hook,
            parse_constant=lambda raw: (
                _ for _ in ()
            ).throw(ValueError(raw)),
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise VN97ProductionPreflightError(
            "VN97PREFLIGHT1 must be strict UTF-8 JSON"
        ) from exc
    if duplicates or not isinstance(root, dict):
        raise VN97ProductionPreflightError(
            "VN97PREFLIGHT1 must be one object without duplicate keys"
        )
    canonical = json.dumps(
        root,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if canonical != data:
        raise VN97ProductionPreflightError(
            "VN97PREFLIGHT1 must use canonical JSON"
        )
    expected = {
        "blockers",
        "candidates",
        "language",
        "manifest_sha256",
        "schema",
        "speech",
        "status",
    }
    if (
        set(root) != expected
        or root["schema"] != SCHEMA
    ):
        raise VN97ProductionPreflightError(
            "VN97PREFLIGHT1 schema/keys mismatch"
        )
    language = root["language"]
    speech = root["speech"]
    if (
        not isinstance(language, dict)
        or set(language)
        != {
            "release_dataset_sha256",
            "release_records",
            "release_target_tokens",
            "release_windows",
            "tokenizer_bytes",
            "tokenizer_sha256",
            "training_dataset_sha256",
            "training_records",
            "training_windows",
            "validation_dataset_sha256",
            "validation_records",
            "validation_target_tokens",
            "validation_windows",
            "vocab_size",
        }
        or not isinstance(speech, dict)
        or set(speech)
        != {
            "max_observed_frames",
            "release_dataset_sha256",
            "release_examples",
            "release_target_tokens",
            "training_dataset_sha256",
            "training_examples",
            "validation_dataset_sha256",
            "validation_examples",
            "validation_target_tokens",
        }
        or not isinstance(root["candidates"], list)
        or not isinstance(root["blockers"], list)
    ):
        raise VN97ProductionPreflightError(
            "VN97PREFLIGHT1 nested keys mismatch"
        )

    candidates: list[
        VN97PreflightCandidate
    ] = []
    for item in root["candidates"]:
        if (
            not isinstance(item, dict)
            or set(item)
            != {
                "candidate_id",
                "language_model_image_bytes",
                "language_recurrent_state_bytes",
                "parameter_count",
                "speech_model_image_bytes",
                "speech_recurrent_state_bytes",
                "status",
            }
        ):
            raise VN97ProductionPreflightError(
                "VN97PREFLIGHT1 candidate keys mismatch"
            )
        candidates.append(
            VN97PreflightCandidate(
                candidate_id=
                    item["candidate_id"],
                parameter_count=
                    item["parameter_count"],
                language_model_image_bytes=
                    item[
                        "language_model_image_bytes"
                    ],
                language_recurrent_state_bytes=
                    item[
                        "language_recurrent_state_bytes"
                    ],
                speech_model_image_bytes=
                    item[
                        "speech_model_image_bytes"
                    ],
                speech_recurrent_state_bytes=
                    item[
                        "speech_recurrent_state_bytes"
                    ],
                status=item["status"],
            )
        )

    blockers: list[
        VN97PreflightBlocker
    ] = []
    for item in root["blockers"]:
        if (
            not isinstance(item, dict)
            or set(item)
            != {"code", "message"}
        ):
            raise VN97ProductionPreflightError(
                "VN97PREFLIGHT1 blocker keys mismatch"
            )
        blockers.append(
            VN97PreflightBlocker(
                code=item["code"],
                message=item["message"],
            )
        )

    try:
        return VN97ProductionPreflightReport(
            manifest_sha256=
                root["manifest_sha256"],
            status=root["status"],
            tokenizer_sha256=
                language["tokenizer_sha256"],
            tokenizer_bytes=
                language["tokenizer_bytes"],
            vocab_size=
                language["vocab_size"],
            training_dataset_sha256=
                language[
                    "training_dataset_sha256"
                ],
            validation_dataset_sha256=
                language[
                    "validation_dataset_sha256"
                ],
            release_dataset_sha256=
                language[
                    "release_dataset_sha256"
                ],
            training_records=
                language["training_records"],
            validation_records=
                language[
                    "validation_records"
                ],
            release_records=
                language["release_records"],
            training_windows=
                language["training_windows"],
            validation_windows=
                language[
                    "validation_windows"
                ],
            release_windows=
                language["release_windows"],
            validation_target_tokens=
                language[
                    "validation_target_tokens"
                ],
            release_target_tokens=
                language[
                    "release_target_tokens"
                ],
            speech_training_dataset_sha256=
                speech[
                    "training_dataset_sha256"
                ],
            speech_validation_dataset_sha256=
                speech[
                    "validation_dataset_sha256"
                ],
            speech_release_dataset_sha256=
                speech[
                    "release_dataset_sha256"
                ],
            speech_training_examples=
                speech["training_examples"],
            speech_validation_examples=
                speech[
                    "validation_examples"
                ],
            speech_release_examples=
                speech["release_examples"],
            speech_validation_target_tokens=
                speech[
                    "validation_target_tokens"
                ],
            speech_release_target_tokens=
                speech[
                    "release_target_tokens"
                ],
            speech_max_observed_frames=
                speech["max_observed_frames"],
            candidates=tuple(candidates),
            blockers=tuple(blockers),
        )
    except (
        TypeError,
        ValueError,
        VN97ProductionPreflightError,
    ) as exc:
        if isinstance(
            exc,
            VN97ProductionPreflightError,
        ):
            raise
        raise VN97ProductionPreflightError(
            str(exc)
        ) from exc
