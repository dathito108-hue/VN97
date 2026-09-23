from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable

from .evaluation import VN97EvaluationResult


class VN97CampaignError(RuntimeError):
    pass


@dataclass(frozen=True)
class VN97CampaignCandidate:
    d_model: int
    n_layers: int
    d_state: int
    embedding_rank: int | None
    seed: int
    learning_rate: float

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.n_layers <= 0 or self.d_state <= 0:
            raise ValueError("campaign model dimensions must be positive")
        if self.embedding_rank is not None:
            if not 0 < self.embedding_rank < self.d_model:
                raise ValueError(
                    "campaign embedding_rank must be in [1, d_model)"
                )
        if self.seed < 0:
            raise ValueError("campaign seed must be non-negative")
        if (
            not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
        ):
            raise ValueError(
                "campaign learning_rate must be finite and positive"
            )

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
        blob = json.dumps(
            self.canonical_object(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


@dataclass(frozen=True)
class VN97CampaignObservation:
    candidate: VN97CampaignCandidate
    parameter_count: int
    evaluation: VN97EvaluationResult
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        if self.parameter_count <= 0:
            raise ValueError("campaign parameter_count must be positive")
        if (
            len(self.checkpoint_sha256) != 64
            or any(
                ch not in "0123456789abcdef"
                for ch in self.checkpoint_sha256
            )
        ):
            raise ValueError(
                "campaign checkpoint_sha256 must be lowercase SHA-256"
            )

    @property
    def ranking_key(self) -> tuple[float, float, int, str]:
        return (
            self.evaluation.mean_loss,
            -self.evaluation.top1_accuracy,
            self.parameter_count,
            self.candidate.candidate_id,
        )


def select_best_campaign_candidate(
    observations: Iterable[VN97CampaignObservation],
) -> VN97CampaignObservation:
    values = tuple(observations)
    if not values:
        raise VN97CampaignError(
            "no campaign candidate passed the release quality gate"
        )
    ids = [value.candidate.candidate_id for value in values]
    if len(set(ids)) != len(ids):
        raise VN97CampaignError(
            "campaign observations contain duplicate candidate IDs"
        )
    return min(values, key=lambda value: value.ranking_key)
