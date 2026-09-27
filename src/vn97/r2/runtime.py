from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .config import VN97R2Config


class CognitionMode(str, Enum):
    FAST = "fast"
    DEEP = "deep"


@dataclass(frozen=True)
class RealtimeBudget:
    planner_hz: float = 4.0
    reflex_hz: float = 60.0
    fast_deadline_ms: float = 80.0
    deep_deadline_ms: float = 600.0

    def __post_init__(self) -> None:
        if self.planner_hz <= 0.0 or self.reflex_hz <= 0.0:
            raise ValueError("planner/reflex rates must be positive")
        if self.reflex_hz < self.planner_hz:
            raise ValueError("reflex_hz must be >= planner_hz")
        if not 0.0 < self.fast_deadline_ms < self.deep_deadline_ms:
            raise ValueError(
                "expected 0 < fast_deadline_ms < deep_deadline_ms"
            )


@dataclass(frozen=True)
class CognitionRequest:
    deadline_ms: float
    requires_external_write: bool = False
    requires_multistep_reasoning: bool = False
    risk_score: float = 0.0

    def __post_init__(self) -> None:
        if self.deadline_ms <= 0.0:
            raise ValueError("deadline_ms must be positive")
        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError("risk_score must be in [0, 1]")


@dataclass(frozen=True)
class CognitionDecision:
    mode: CognitionMode
    active_layers: int
    reason: str


class VN97R2ExecutionPolicy:
    """Selects fast/deep depth without routing to another model."""

    def __init__(
        self,
        config: VN97R2Config,
        budget: RealtimeBudget | None = None,
    ) -> None:
        self.config = config
        self.budget = budget or RealtimeBudget()

    def decide(self, request: CognitionRequest) -> CognitionDecision:
        if request.requires_external_write:
            return CognitionDecision(
                mode=CognitionMode.DEEP,
                active_layers=self.config.n_layers,
                reason="external_write_requires_canonical_deep_path",
            )
        if request.risk_score >= 0.5:
            return CognitionDecision(
                mode=CognitionMode.DEEP,
                active_layers=self.config.n_layers,
                reason="risk_requires_deep_path",
            )
        if request.requires_multistep_reasoning:
            return CognitionDecision(
                mode=CognitionMode.DEEP,
                active_layers=self.config.n_layers,
                reason="multistep_reasoning_requires_deep_path",
            )
        if request.deadline_ms <= self.budget.fast_deadline_ms:
            return CognitionDecision(
                mode=CognitionMode.FAST,
                active_layers=self.config.fast_layers,
                reason="latency_budget_selects_fast_path",
            )
        return CognitionDecision(
            mode=CognitionMode.DEEP,
            active_layers=self.config.n_layers,
            reason="default_canonical_deep_path",
        )


@dataclass(frozen=True)
class ReflexPlanContract:
    """Separates cognition cadence from deterministic realtime actuation.

    The model produces/updates a canonical plan at planner_hz. A native
    executor may replay already-approved plan primitives at reflex_hz without
    running another model. Side effects remain subject to the existing M6
    authority layer outside this contract.
    """

    plan_id: str
    generation: int
    valid_for_ms: int
    planner_hz: float
    reflex_hz: float

    def __post_init__(self) -> None:
        if not self.plan_id:
            raise ValueError("plan_id must be non-empty")
        if self.generation < 0:
            raise ValueError("generation must be non-negative")
        if self.valid_for_ms <= 0:
            raise ValueError("valid_for_ms must be positive")
        if self.planner_hz <= 0.0 or self.reflex_hz < self.planner_hz:
            raise ValueError("invalid planner/reflex cadence")
