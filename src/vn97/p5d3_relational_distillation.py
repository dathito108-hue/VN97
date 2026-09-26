from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math


P5D3B_SCHEMA = "VN97P5D3B"
P5D3B_FLOAT_ARTIFACT_SCHEMA = "VN97P5D3BFLOAT1"
P5D3B_PROFILE_ID = "vn97-p5d3b-relational-dynamics-distillation-v1"

STUDENT_HIDDEN_DEPTHS = (8, 16, 24, 32)
TEACHER_HIDDEN_DEPTHS = (16, 32, 48, 64)

TRAIN_STEPS = 300
BEHAVIOR_SEQUENCE_LENGTH = 96
P3_VALIDATION_MAX_WINDOWS = 120

BEHAVIOR_WEIGHT = 0.25
GRAM_WEIGHT = 1.0
DYNAMICS_WEIGHT = 0.5
NORM_WEIGHT = 0.1

MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.01
CHECKPOINT_INTERVAL_STEPS = 50
PROGRESS_INTERVAL_STEPS = 25


@dataclass(frozen=True)
class P5D3BCandidate:
    learning_rate: float
    seed: int

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
        ):
            raise ValueError(
                "P5D3B learning rate must be finite and positive"
            )
        if self.seed < 0:
            raise ValueError(
                "P5D3B seed must be non-negative"
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
            b"VN97P5D3BCAND\0" + payload
        ).hexdigest()[:16]


CANDIDATES = (
    P5D3BCandidate(
        learning_rate=2e-5,
        seed=5301,
    ),
    P5D3BCandidate(
        learning_rate=1e-5,
        seed=5301,
    ),
)


def profile_object() -> dict[str, object]:
    return {
        "behavior_sequence_length":
            BEHAVIOR_SEQUENCE_LENGTH,
        "candidates": [
            candidate.canonical_object()
            for candidate in CANDIDATES
        ],
        "checkpoint_interval_steps":
            CHECKPOINT_INTERVAL_STEPS,
        "loss_weights": {
            "behavior":
                BEHAVIOR_WEIGHT,
            "dynamics":
                DYNAMICS_WEIGHT,
            "gram":
                GRAM_WEIGHT,
            "norm":
                NORM_WEIGHT,
        },
        "max_grad_norm":
            MAX_GRAD_NORM,
        "p3_validation_max_windows":
            P3_VALIDATION_MAX_WINDOWS,
        "profile_id":
            P5D3B_PROFILE_ID,
        "schema":
            P5D3B_SCHEMA,
        "student_hidden_depths":
            list(
                STUDENT_HIDDEN_DEPTHS
            ),
        "teacher_hidden_depths":
            list(
                TEACHER_HIDDEN_DEPTHS
            ),
        "train_steps":
            TRAIN_STEPS,
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
        b"VN97P5D3BPROFILE\0" + payload
    ).hexdigest()
