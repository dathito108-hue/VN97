from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


PROFILE_ID = "vn97-p2-medium-cpu-first-v1"
CORPUS_PROFILE_ID = "vn97-production-intelligence-v1"
CAMPAIGN_SCHEMA = "VN97CAMPDEF1"

CANDIDATE = {
    "d_model": 192,
    "d_state": 16,
    "embedding_rank": 96,
    "learning_rate": 0.0003,
    "n_layers": 6,
    "seed": 97,
}

SEQUENCE_LENGTH = 256
BATCH_SIZE = 2
EPOCHS = 2
MAX_TRAIN_WINDOWS = 2048
MAX_VALIDATION_WINDOWS = 256
MAX_RELEASE_WINDOWS = 256
LEARNED_TOKENS = 4096
MAX_PARAMETERS = 5_000_000
MAX_MODEL_IMAGE_BYTES = 64 * 1024 * 1024
MAX_RECURRENT_STATE_BYTES = 8 * 1024 * 1024
TILE_ROWS = 16
TILE_COLS = 16
MAX_VALIDATION_LOSS = 12.0
MIN_VALIDATION_ACCURACY = 0.0
MIN_TARGET_TOKENS = 64


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def candidate_id() -> str:
    return hashlib.sha256(
        _canonical_json(CANDIDATE)
    ).hexdigest()[:16]


def candidate_manifest_bytes() -> bytes:
    return _canonical_json(
        {
            "candidates": [CANDIDATE],
            "schema": CAMPAIGN_SCHEMA,
        }
    ) + b"\n"


def profile_object() -> dict[str, object]:
    return {
        "batch_size": BATCH_SIZE,
        "candidate": CANDIDATE,
        "candidate_id": candidate_id(),
        "corpus_profile_id": CORPUS_PROFILE_ID,
        "epochs": EPOCHS,
        "learned_tokens": LEARNED_TOKENS,
        "max_model_image_bytes": MAX_MODEL_IMAGE_BYTES,
        "max_parameters": MAX_PARAMETERS,
        "max_recurrent_state_bytes": MAX_RECURRENT_STATE_BYTES,
        "max_release_windows": MAX_RELEASE_WINDOWS,
        "max_train_windows": MAX_TRAIN_WINDOWS,
        "max_validation_loss": MAX_VALIDATION_LOSS,
        "max_validation_windows": MAX_VALIDATION_WINDOWS,
        "min_target_tokens": MIN_TARGET_TOKENS,
        "min_validation_accuracy": MIN_VALIDATION_ACCURACY,
        "profile_id": PROFILE_ID,
        "sequence_length": SEQUENCE_LENGTH,
        "tile_cols": TILE_COLS,
        "tile_rows": TILE_ROWS,
    }


def profile_sha256() -> str:
    return hashlib.sha256(
        b"VN97PILOTPROFILE1\0"
        + _canonical_json(profile_object())
    ).hexdigest()


def campaign_argv(
    *,
    corpus_dir: Path,
    output_dir: Path,
    campaign_path: Path,
    device: str,
) -> list[str]:
    if not device:
        raise ValueError("pilot device must be non-empty")
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
        "--min-pair-count",
        "2",
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
        str(MAX_VALIDATION_LOSS),
        "--release-min-validation-accuracy",
        str(MIN_VALIDATION_ACCURACY),
        "--release-min-validation-target-tokens",
        str(MIN_TARGET_TOKENS),
        "--device",
        device,
    ]


@dataclass(frozen=True)
class VN97PilotReceipt:
    corpus_manifest_id: str
    corpus_manifest_sha256: str
    campaign_report_sha256: str
    candidate_id: str
    checkpoint_sha256: str
    tokenizer_sha256: str
    model_image_sha256: str
    model_image_bytes: int
    device: str
    profile_sha256: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.corpus_manifest_id, "corpus manifest id"),
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
                raise ValueError(f"{label} must be lowercase SHA-256")
        if self.candidate_id != candidate_id():
            raise ValueError("pilot candidate identity mismatch")
        if self.profile_sha256 != profile_sha256():
            raise ValueError("pilot profile identity mismatch")
        if not self.device:
            raise ValueError("pilot device must be non-empty")
        if not 0 < self.model_image_bytes <= MAX_MODEL_IMAGE_BYTES:
            raise ValueError("pilot model image size is outside bounds")

    def canonical_object(self) -> dict[str, object]:
        return {
            "campaign_report_sha256": self.campaign_report_sha256,
            "candidate_id": self.candidate_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "corpus_manifest_id": self.corpus_manifest_id,
            "corpus_manifest_sha256": self.corpus_manifest_sha256,
            "device": self.device,
            "model_image_bytes": self.model_image_bytes,
            "model_image_sha256": self.model_image_sha256,
            "profile_sha256": self.profile_sha256,
            "schema": "VN97PILOT1",
            "tokenizer_sha256": self.tokenizer_sha256,
        }

    def to_bytes(self) -> bytes:
        return _canonical_json(
            self.canonical_object()
        ) + b"\n"
