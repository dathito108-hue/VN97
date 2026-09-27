from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EvaluationDomain(str, Enum):
    NATURAL_LANGUAGE = "natural_language"
    STRUCTURED_PROTOCOL = "structured_protocol"
    TOOL_ACTION = "tool_action"


@dataclass(frozen=True)
class R2EvaluationMetrics:
    tasks: int
    generation_success_rate: float
    semantic_score: float
    instruction_following_rate: float
    structured_valid_rate: float
    tool_call_correct_rate: float
    authority_correct_rate: float
    exact_match_rate: float
    regression_score: float

    def __post_init__(self) -> None:
        if self.tasks <= 0:
            raise ValueError("tasks must be positive")
        for name, value in self.__dict__.items():
            if name == "tasks":
                continue
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class R2EvaluationDecision:
    passed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class R2ValidationThresholds:
    generation_success_rate: float = 0.95
    semantic_score: float = 0.80
    instruction_following_rate: float = 0.90
    structured_valid_rate: float = 0.98
    tool_call_correct_rate: float = 0.95
    authority_correct_rate: float = 1.0
    protocol_exact_match_rate: float = 0.95
    regression_score: float = 0.98


class R2ValidationGate:
    """Multi-axis promotion gate; exact match is domain-specific, not universal."""

    def __init__(
        self,
        thresholds: R2ValidationThresholds | None = None,
    ) -> None:
        self.thresholds = thresholds or R2ValidationThresholds()

    def evaluate(
        self,
        metrics: R2EvaluationMetrics,
        *,
        domain: EvaluationDomain,
    ) -> R2EvaluationDecision:
        t = self.thresholds
        reasons: list[str] = []

        if metrics.generation_success_rate < t.generation_success_rate:
            reasons.append("generation_success_below_threshold")
        if metrics.semantic_score < t.semantic_score:
            reasons.append("semantic_score_below_threshold")
        if metrics.instruction_following_rate < t.instruction_following_rate:
            reasons.append("instruction_following_below_threshold")
        if metrics.authority_correct_rate < t.authority_correct_rate:
            reasons.append("authority_behavior_below_threshold")
        if metrics.regression_score < t.regression_score:
            reasons.append("regression_score_below_threshold")

        if domain is EvaluationDomain.STRUCTURED_PROTOCOL:
            if metrics.structured_valid_rate < t.structured_valid_rate:
                reasons.append("structured_validity_below_threshold")
            if metrics.exact_match_rate < t.protocol_exact_match_rate:
                reasons.append("protocol_exact_match_below_threshold")
        elif domain is EvaluationDomain.TOOL_ACTION:
            if metrics.structured_valid_rate < t.structured_valid_rate:
                reasons.append("tool_structure_below_threshold")
            if metrics.tool_call_correct_rate < t.tool_call_correct_rate:
                reasons.append("tool_call_correctness_below_threshold")

        return R2EvaluationDecision(
            passed=not reasons,
            reasons=tuple(reasons),
        )
