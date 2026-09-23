from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from .evaluation import (
    VN97EvaluationResult,
    VN97ReleaseCriteria,
    evaluate_vn97_language,
    require_release_quality,
)
from .model import VN97LanguageCore
from .training import (
    VN97TrainingConfig,
    VN97TrainingResult,
    VN97TrainingWindow,
    train_vn97_language,
)


class VN97FineTuneQualityError(RuntimeError):
    pass


@dataclass(frozen=True)
class VN97FineTuneResult:
    baseline: VN97EvaluationResult
    training: VN97TrainingResult
    final: VN97EvaluationResult

    def __post_init__(self) -> None:
        if self.baseline.target_tokens != self.final.target_tokens:
            raise ValueError(
                "baseline/final validation target-token counts must match"
            )


def guarded_fine_tune_vn97(
    model: VN97LanguageCore,
    train_windows: Sequence[VN97TrainingWindow],
    validation_windows: Sequence[VN97TrainingWindow],
    training_config: VN97TrainingConfig,
    release_criteria: VN97ReleaseCriteria,
    *,
    max_validation_loss_increase: float = 0.0,
    validation_batch_size: int | None = None,
    device: str | torch.device = "auto",
) -> VN97FineTuneResult:
    """Fine-tune one canonical VN97 checkpoint and gate deployment on held-out quality.

    This intentionally starts a fresh optimizer. VN97CK1 is a deployment checkpoint,
    not an optimizer/training-state serialization format.
    """
    if not isinstance(model, VN97LanguageCore):
        raise TypeError("model must be VN97LanguageCore")
    if not train_windows:
        raise ValueError("train_windows must not be empty")
    if not validation_windows:
        raise ValueError("validation_windows must not be empty")
    if (
        not math.isfinite(max_validation_loss_increase)
        or max_validation_loss_increase < 0.0
    ):
        raise ValueError(
            "max_validation_loss_increase must be finite and non-negative"
        )
    validation_batch = (
        training_config.batch_size
        if validation_batch_size is None
        else validation_batch_size
    )
    if validation_batch <= 0:
        raise ValueError("validation_batch_size must be positive")

    baseline = evaluate_vn97_language(
        model,
        validation_windows,
        batch_size=validation_batch,
        device=device,
    )
    training = train_vn97_language(
        model,
        train_windows,
        training_config,
        device=device,
    )
    final = evaluate_vn97_language(
        model,
        validation_windows,
        batch_size=validation_batch,
        device=device,
    )

    require_release_quality(final, release_criteria)
    allowed_loss = baseline.mean_loss + max_validation_loss_increase
    if final.mean_loss > allowed_loss:
        raise VN97FineTuneQualityError(
            "fine-tune validation loss regressed beyond allowed increase"
        )

    return VN97FineTuneResult(
        baseline=baseline,
        training=training,
        final=final,
    )
