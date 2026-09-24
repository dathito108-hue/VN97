from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path


PROFILE_ID = "vn97-p3-language-production-v1"
CORPUS_PROFILE_ID = "vn97-production-intelligence-v1"
CAMPAIGN_SCHEMA = "VN97CAMPDEF1"
P2_BASELINE_RUN_IDENTITY = (
    "e0509769f3f819516c7e41c287690536"
    "73f634e41051d8e469b95bd8421db5a2"
)

CANDIDATES = (
    {
        "d_model": 256,
        "d_state": 16,
        "embedding_rank": 128,
        "learning_rate": 0.0003,
        "n_layers": 8,
        "seed": 97,
    },
    {
        "d_model": 320,
        "d_state": 24,
        "embedding_rank": 160,
        "learning_rate": 0.00025,
        "n_layers": 10,
        "seed": 98,
    },
    {
        "d_model": 384,
        "d_state": 24,
        "embedding_rank": 192,
        "learning_rate": 0.0002,
        "n_layers": 12,
        "seed": 99,
    },
    {
        "d_model": 448,
        "d_state": 32,
        "embedding_rank": 224,
        "learning_rate": 0.00018,
        "n_layers": 12,
        "seed": 100,
    },
)

SEQUENCE_LENGTH = 512
BATCH_SIZE = 8
EPOCHS = 2
MAX_TRAIN_WINDOWS = 100_000
MAX_VALIDATION_WINDOWS = 10_000
MAX_RELEASE_WINDOWS = 10_000
LEARNED_TOKENS = 4096
TOKENIZER_TRAIN_RECORDS = 2048
MIN_PAIR_COUNT = 2

MAX_PARAMETERS = 50_000_000
MAX_MODEL_IMAGE_BYTES = 128 * 1024 * 1024
MAX_RECURRENT_STATE_BYTES = 8 * 1024 * 1024
TILE_ROWS = 16
TILE_COLS = 16

MAX_VALIDATION_LOSS = 6.76
MIN_VALIDATION_ACCURACY = 0.052
RELEASE_MAX_VALIDATION_LOSS = 6.79
RELEASE_MIN_VALIDATION_ACCURACY = 0.039
MIN_TARGET_TOKENS = 512

MIN_TRAIN_RECORDS = 10_000
MIN_VALIDATION_RECORDS = 500
MIN_RELEASE_RECORDS = 500


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def candidate_manifest_bytes() -> bytes:
    return _canonical_json(
        {
            "candidates": list(CANDIDATES),
            "schema": CAMPAIGN_SCHEMA,
        }
    ) + b"\n"


def candidate_ids() -> tuple[str, ...]:
    return tuple(
        hashlib.sha256(
            _canonical_json(candidate)
        ).hexdigest()[:16]
        for candidate in CANDIDATES
    )


def profile_object() -> dict[str, object]:
    return {
        "batch_size": BATCH_SIZE,
        "candidate_ids": list(candidate_ids()),
        "candidates": list(CANDIDATES),
        "corpus_profile_id": CORPUS_PROFILE_ID,
        "epochs": EPOCHS,
        "learned_tokens": LEARNED_TOKENS,
        "tokenizer_train_records": TOKENIZER_TRAIN_RECORDS,
        "tokenizer_sampling": "sha256-vn97toksample1",
        "max_model_image_bytes": MAX_MODEL_IMAGE_BYTES,
        "max_parameters": MAX_PARAMETERS,
        "max_recurrent_state_bytes": MAX_RECURRENT_STATE_BYTES,
        "max_release_windows": MAX_RELEASE_WINDOWS,
        "max_train_windows": MAX_TRAIN_WINDOWS,
        "max_validation_loss": MAX_VALIDATION_LOSS,
        "max_validation_windows": MAX_VALIDATION_WINDOWS,
        "min_pair_count": MIN_PAIR_COUNT,
        "min_release_records": MIN_RELEASE_RECORDS,
        "min_target_tokens": MIN_TARGET_TOKENS,
        "min_train_records": MIN_TRAIN_RECORDS,
        "min_validation_accuracy": MIN_VALIDATION_ACCURACY,
        "min_validation_records": MIN_VALIDATION_RECORDS,
        "p2_baseline_run_identity": P2_BASELINE_RUN_IDENTITY,
        "profile_id": PROFILE_ID,
        "release_max_validation_loss": RELEASE_MAX_VALIDATION_LOSS,
        "release_min_validation_accuracy":
            RELEASE_MIN_VALIDATION_ACCURACY,
        "sequence_length": SEQUENCE_LENGTH,
        "tile_cols": TILE_COLS,
        "tile_rows": TILE_ROWS,
    }


def profile_sha256() -> str:
    return hashlib.sha256(
        b"VN97P3PROFILE1\0"
        + _canonical_json(profile_object())
    ).hexdigest()


def campaign_argv(
    *,
    corpus_dir: Path,
    output_dir: Path,
    campaign_path: Path,
    device: str,
) -> list[str]:
    if not device or not device.startswith("cuda"):
        raise ValueError(
            "P3 production campaign requires an explicit CUDA device"
        )
    return [
        "--input",
        str(corpus_dir / "training.jsonl"),
        "--validation-input",
        str(corpus_dir / "validation.jsonl"),
        "--release-input",
        str(corpus_dir / "release.jsonl"),
        "--format",
        "chat",
        "--validation-format",
        "chat",
        "--release-format",
        "chat",
        "--campaign",
        str(campaign_path),
        "--output-dir",
        str(output_dir),
        "--learned-tokens",
        str(LEARNED_TOKENS),
        "--tokenizer-max-records",
        str(TOKENIZER_TRAIN_RECORDS),
        "--min-pair-count",
        str(MIN_PAIR_COUNT),
        "--sequence-length",
        str(SEQUENCE_LENGTH),
        "--batch-size",
        str(BATCH_SIZE),
        "--validation-batch-size",
        str(BATCH_SIZE),
        "--release-batch-size",
        str(BATCH_SIZE),
        "--epochs",
        str(EPOCHS),
        "--max-windows",
        str(MAX_TRAIN_WINDOWS),
        "--validation-max-windows",
        str(MAX_VALIDATION_WINDOWS),
        "--release-max-windows",
        str(MAX_RELEASE_WINDOWS),
        "--max-parameters",
        str(MAX_PARAMETERS),
        "--max-model-image-bytes",
        str(MAX_MODEL_IMAGE_BYTES),
        "--max-recurrent-state-bytes",
        str(MAX_RECURRENT_STATE_BYTES),
        "--deployment-tile-rows",
        str(TILE_ROWS),
        "--deployment-tile-cols",
        str(TILE_COLS),
        "--max-validation-loss",
        str(MAX_VALIDATION_LOSS),
        "--min-validation-accuracy",
        str(MIN_VALIDATION_ACCURACY),
        "--min-validation-target-tokens",
        str(MIN_TARGET_TOKENS),
        "--release-max-validation-loss",
        str(RELEASE_MAX_VALIDATION_LOSS),
        "--release-min-validation-accuracy",
        str(RELEASE_MIN_VALIDATION_ACCURACY),
        "--release-min-validation-target-tokens",
        str(MIN_TARGET_TOKENS),
        "--device",
        device,
    ]


@dataclass(frozen=True)
class VN97P3CampaignReceipt:
    corpus_manifest_id: str
    corpus_manifest_sha256: str
    campaign_report_sha256: str
    selected_candidate_id: str
    checkpoint_sha256: str
    tokenizer_sha256: str
    model_image_sha256: str
    model_image_bytes: int
    validation_mean_loss: float
    validation_top1_accuracy: float
    release_mean_loss: float
    release_top1_accuracy: float
    device: str
    profile_sha256: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.corpus_manifest_id, "corpus manifest ID"),
            (self.corpus_manifest_sha256, "corpus manifest SHA-256"),
            (self.campaign_report_sha256, "campaign report SHA-256"),
            (self.checkpoint_sha256, "checkpoint SHA-256"),
            (self.tokenizer_sha256, "tokenizer SHA-256"),
            (self.model_image_sha256, "model image SHA-256"),
            (self.profile_sha256, "profile SHA-256"),
        ):
            if (
                len(value) != 64
                or any(ch not in "0123456789abcdef" for ch in value)
            ):
                raise ValueError(
                    f"{label} must be lowercase SHA-256"
                )
        if self.selected_candidate_id not in candidate_ids():
            raise ValueError(
                "selected candidate is outside frozen P3 candidate set"
            )
        if self.profile_sha256 != profile_sha256():
            raise ValueError(
                "P3 profile identity mismatch"
            )
        if not self.device.startswith("cuda"):
            raise ValueError(
                "P3 campaign receipt requires a CUDA device"
            )
        if not 0 < self.model_image_bytes <= MAX_MODEL_IMAGE_BYTES:
            raise ValueError(
                "P3 model image size is outside bounds"
            )
        for value, label in (
            (self.validation_mean_loss, "validation mean loss"),
            (self.release_mean_loss, "release mean loss"),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{label} must be finite and non-negative"
                )
        for value, label in (
            (
                self.validation_top1_accuracy,
                "validation top-1 accuracy",
            ),
            (
                self.release_top1_accuracy,
                "release top-1 accuracy",
            ),
        ):
            if (
                not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(
                    f"{label} must be finite in [0, 1]"
                )

    def canonical_object(self) -> dict[str, object]:
        return {
            "campaign_report_sha256": self.campaign_report_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "corpus_manifest_id": self.corpus_manifest_id,
            "corpus_manifest_sha256": self.corpus_manifest_sha256,
            "device": self.device,
            "model_image_bytes": self.model_image_bytes,
            "model_image_sha256": self.model_image_sha256,
            "p2_baseline_run_identity": P2_BASELINE_RUN_IDENTITY,
            "profile_sha256": self.profile_sha256,
            "release_mean_loss": self.release_mean_loss,
            "release_top1_accuracy": self.release_top1_accuracy,
            "schema": "VN97P3RUN1",
            "selected_candidate_id": self.selected_candidate_id,
            "tokenizer_sha256": self.tokenizer_sha256,
            "validation_mean_loss": self.validation_mean_loss,
            "validation_top1_accuracy": self.validation_top1_accuracy,
        }

    def to_bytes(self) -> bytes:
        return _canonical_json(
            self.canonical_object()
        ) + b"\n"
