from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math


P5D2_SCHEMA = "VN97P5D2"
P5D2_FLOAT_ARTIFACT_SCHEMA = "VN97P5D2FLOAT1"
P5D2_PROFILE_ID = "vn97-p5d2-dual-float-shadow-pilot-v1"

SEQUENCE_LENGTH = 96
LOGICAL_BATCH_SIZE = 4
MICRO_BATCH_SIZE = 1
MAX_TRAIN_WINDOWS = 1200
P3_REPLAY_RECORDS = 600
P3_VALIDATION_MAX_WINDOWS = 200

TEACHER_HOLDOUT_PER_P4_CATEGORY = 10
TEACHER_HOLDOUT_GENERAL_LANGUAGE = 40

CHECKPOINT_INTERVAL_STEPS = 50
PROGRESS_INTERVAL_STEPS = 25
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.01

TARGET_D_MODEL = 1536
TARGET_LAYERS = 32
TARGET_D_STATE = 16
TARGET_EMBEDDING_RANK = 768


@dataclass(frozen=True)
class P5D2Candidate:
    learning_rate: float
    seed: int

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
        ):
            raise ValueError(
                "P5D2 learning rate must be finite and positive"
            )
        if self.seed < 0:
            raise ValueError(
                "P5D2 seed must be non-negative"
            )

    def canonical_object(self) -> dict[str, object]:
        return {
            "learning_rate": self.learning_rate,
            "seed": self.seed,
        }

    @property
    def candidate_id(self) -> str:
        payload = json.dumps(
            self.canonical_object(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(
            b"VN97P5D2CAND\0" + payload
        ).hexdigest()[:16]


# Same deterministic initialization seed; only the optimizer rate differs.
# Both candidates run concurrently on the two Kaggle T4 GPUs.
CANDIDATES = (
    P5D2Candidate(
        learning_rate=8e-5,
        seed=5201,
    ),
    P5D2Candidate(
        learning_rate=4e-5,
        seed=5201,
    ),
)


def profile_object() -> dict[str, object]:
    return {
        "candidates": [
            candidate.canonical_object()
            for candidate in CANDIDATES
        ],
        "checkpoint_interval_steps":
            CHECKPOINT_INTERVAL_STEPS,
        "logical_batch_size":
            LOGICAL_BATCH_SIZE,
        "max_grad_norm":
            MAX_GRAD_NORM,
        "max_train_windows":
            MAX_TRAIN_WINDOWS,
        "micro_batch_size":
            MICRO_BATCH_SIZE,
        "p3_replay_records":
            P3_REPLAY_RECORDS,
        "p3_validation_max_windows":
            P3_VALIDATION_MAX_WINDOWS,
        "profile_id":
            P5D2_PROFILE_ID,
        "schema":
            P5D2_SCHEMA,
        "sequence_length":
            SEQUENCE_LENGTH,
        "target": {
            "d_model":
                TARGET_D_MODEL,
            "d_state":
                TARGET_D_STATE,
            "embedding_rank":
                TARGET_EMBEDDING_RANK,
            "layers":
                TARGET_LAYERS,
        },
        "teacher_holdout": {
            "general_language":
                TEACHER_HOLDOUT_GENERAL_LANGUAGE,
            "per_p4_category":
                TEACHER_HOLDOUT_PER_P4_CATEGORY,
        },
        "training_precision":
            "fp32-float-shadow",
        "weight_decay":
            WEIGHT_DECAY,
    }


def profile_sha256() -> str:
    payload = json.dumps(
        profile_object(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(
        b"VN97P5D2\0" + payload
    ).hexdigest()
