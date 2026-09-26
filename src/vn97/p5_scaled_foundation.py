from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math


P5A_PROFILE_ID = "vn97-p5a-scaled-foundation-pilot-v1"
P5A_SCHEMA = "VN97P5APROFILE1"

SEQUENCE_LENGTH = 256
LOGICAL_BATCH_SIZE = 8
MICRO_BATCH_SIZE = 1
EPOCHS = 1
MAX_TRAIN_WINDOWS = 6000
P3_TRAIN_RECORDS = 3000
P4_TRAIN_RECORDS = 3000
P3_VALIDATION_MAX_WINDOWS = 1200

MAX_MODEL_IMAGE_BYTES = 128 * 1024 * 1024
MAX_RECURRENT_STATE_BYTES = 8 * 1024 * 1024
TILE_ROWS = 16
TILE_COLS = 16


@dataclass(frozen=True)
class P5AScaleCandidate:
    d_model: int
    n_layers: int
    d_state: int
    embedding_rank: int
    learning_rate: float
    seed: int

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.n_layers <= 0 or self.d_state <= 0:
            raise ValueError("P5A model dimensions must be positive")
        if not 0 < self.embedding_rank < self.d_model:
            raise ValueError(
                "P5A embedding rank must satisfy 0 < rank < d_model"
            )
        if (
            not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
        ):
            raise ValueError(
                "P5A learning rate must be finite and positive"
            )
        if self.seed < 0:
            raise ValueError("P5A seed must be non-negative")

    def canonical_object(self) -> dict[str, object]:
        return {
            "d_model": self.d_model,
            "d_state": self.d_state,
            "embedding_rank": self.embedding_rank,
            "learning_rate": self.learning_rate,
            "n_layers": self.n_layers,
            "seed": self.seed,
        }

    @property
    def candidate_id(self) -> str:
        data = json.dumps(
            self.canonical_object(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(data).hexdigest()[:16]


# Same VN97 architecture at increasing scale. With the current 4104-token
# tokenizer these materialize at roughly 26.3M, 39.0M and 50.6M parameters.
CANDIDATES = (
    P5AScaleCandidate(
        d_model=576,
        n_layers=18,
        d_state=32,
        embedding_rank=288,
        learning_rate=1.5e-4,
        seed=501,
    ),
    P5AScaleCandidate(
        d_model=672,
        n_layers=20,
        d_state=32,
        embedding_rank=336,
        learning_rate=1.2e-4,
        seed=502,
    ),
    P5AScaleCandidate(
        d_model=768,
        n_layers=20,
        d_state=32,
        embedding_rank=384,
        learning_rate=1.0e-4,
        seed=503,
    ),
)


def profile_object() -> dict[str, object]:
    return {
        "candidates": [
            candidate.canonical_object()
            for candidate in CANDIDATES
        ],
        "epochs": EPOCHS,
        "logical_batch_size": LOGICAL_BATCH_SIZE,
        "max_model_image_bytes": MAX_MODEL_IMAGE_BYTES,
        "max_recurrent_state_bytes": MAX_RECURRENT_STATE_BYTES,
        "max_train_windows": MAX_TRAIN_WINDOWS,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "p3_train_records": P3_TRAIN_RECORDS,
        "p3_validation_max_windows": P3_VALIDATION_MAX_WINDOWS,
        "p4_train_records": P4_TRAIN_RECORDS,
        "profile_id": P5A_PROFILE_ID,
        "schema": P5A_SCHEMA,
        "sequence_length": SEQUENCE_LENGTH,
        "tile_cols": TILE_COLS,
        "tile_rows": TILE_ROWS,
    }


def profile_sha256() -> str:
    encoded = json.dumps(
        profile_object(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(
        b"VN97P5APROFILE1\0"
        + encoded
    ).hexdigest()
