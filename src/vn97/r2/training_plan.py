from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class R2TrainingStage(str, Enum):
    ARCHITECTURE = "architecture"
    DENSE_PRETRAIN = "dense_pretrain"
    INSTRUCTION_REASONING = "instruction_reasoning"
    TOOL_ACTION = "tool_action"
    CAPABILITY = "capability"
    FAST_PATH_ALIGNMENT = "fast_path_alignment"
    FRESH_VALIDATION = "fresh_validation"
    QAT = "qat"
    MOBILE_LOWERING = "mobile_lowering"


@dataclass(frozen=True)
class StageEvidence:
    stage: R2TrainingStage
    generation_valid: bool
    semantic_valid: bool
    tool_protocol_valid: bool
    authority_valid: bool
    regression_valid: bool

    @property
    def passed(self) -> bool:
        return (
            self.generation_valid
            and self.semantic_valid
            and self.tool_protocol_valid
            and self.authority_valid
            and self.regression_valid
        )


class R2PromotionPolicy:
    """Dense-intelligence-first promotion order.

    QAT and ternary/mobile lowering are structurally forbidden until fresh
    dense validation has passed. This prevents compression from being used as
    a repair mechanism for an intelligence core that has not yet demonstrated
    stable generation and task completion.
    """

    _ORDER = (
        R2TrainingStage.ARCHITECTURE,
        R2TrainingStage.DENSE_PRETRAIN,
        R2TrainingStage.INSTRUCTION_REASONING,
        R2TrainingStage.TOOL_ACTION,
        R2TrainingStage.CAPABILITY,
        R2TrainingStage.FAST_PATH_ALIGNMENT,
        R2TrainingStage.FRESH_VALIDATION,
        R2TrainingStage.QAT,
        R2TrainingStage.MOBILE_LOWERING,
    )

    def next_stage(
        self,
        current: R2TrainingStage,
        evidence: StageEvidence,
    ) -> R2TrainingStage:
        if evidence.stage != current:
            raise ValueError("evidence stage does not match current stage")
        if not evidence.passed:
            return current
        index = self._ORDER.index(current)
        if index == len(self._ORDER) - 1:
            return current
        return self._ORDER[index + 1]

    def qat_allowed(
        self,
        fresh_validation: StageEvidence,
    ) -> bool:
        return (
            fresh_validation.stage
            is R2TrainingStage.FRESH_VALIDATION
            and fresh_validation.passed
        )


R2_CANONICAL_TRAINING_SEQUENCE = tuple(
    stage.value for stage in R2PromotionPolicy._ORDER
)
