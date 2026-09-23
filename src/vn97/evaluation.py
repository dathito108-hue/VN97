from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn.functional as F

from .model import VN97LanguageCore
from .training import IGNORE_INDEX, VN97TrainingWindow


class VN97ReleaseQualityError(RuntimeError):
    pass


@dataclass(frozen=True)
class VN97EvaluationResult:
    windows: int
    target_tokens: int
    mean_loss: float
    top1_accuracy: float

    def __post_init__(self) -> None:
        if self.windows <= 0 or self.target_tokens <= 0:
            raise ValueError("evaluation result must contain supervised work")
        if not math.isfinite(self.mean_loss) or self.mean_loss < 0.0:
            raise ValueError("evaluation mean_loss must be finite and non-negative")
        if (
            not math.isfinite(self.top1_accuracy)
            or not 0.0 <= self.top1_accuracy <= 1.0
        ):
            raise ValueError("evaluation top1_accuracy must be in [0, 1]")


@dataclass(frozen=True)
class VN97ReleaseCriteria:
    max_validation_loss: float
    min_top1_accuracy: float = 0.0
    min_target_tokens: int = 64

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.max_validation_loss)
            or self.max_validation_loss <= 0.0
        ):
            raise ValueError("max_validation_loss must be finite and positive")
        if (
            not math.isfinite(self.min_top1_accuracy)
            or not 0.0 <= self.min_top1_accuracy <= 1.0
        ):
            raise ValueError("min_top1_accuracy must be in [0, 1]")
        if self.min_target_tokens <= 0:
            raise ValueError("min_target_tokens must be positive")


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    return torch.device(device)


def evaluate_vn97_language(
    model: VN97LanguageCore,
    windows: Sequence[VN97TrainingWindow],
    *,
    batch_size: int = 4,
    device: str | torch.device = "auto",
) -> VN97EvaluationResult:
    if not isinstance(model, VN97LanguageCore):
        raise TypeError("model must be VN97LanguageCore")
    if not windows:
        raise ValueError("evaluation windows must not be empty")
    if batch_size <= 0:
        raise ValueError("evaluation batch_size must be positive")

    lengths = {len(window.input_ids) for window in windows}
    if len(lengths) != 1:
        raise ValueError("evaluation windows must have one fixed sequence length")

    resolved = _resolve_device(device)
    model.to(resolved)
    model.eval()

    total_loss = 0.0
    total_targets = 0
    total_correct = 0

    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = windows[start : start + batch_size]
            inputs = torch.tensor(
                [window.input_ids for window in batch],
                dtype=torch.long,
                device=resolved,
            )
            labels = torch.tensor(
                [window.labels for window in batch],
                dtype=torch.long,
                device=resolved,
            )
            logits, _ = model(inputs)
            if (
                logits.ndim != 3
                or logits.shape[:2] != inputs.shape
                or logits.shape[-1] != model.config.vocab_size
            ):
                raise RuntimeError(
                    "VN97 model returned invalid evaluation logits shape"
                )

            active = labels != IGNORE_INDEX
            target_count = int(active.sum().item())
            if target_count <= 0:
                raise RuntimeError(
                    "evaluation batch contains no supervised target tokens"
                )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(
                    "VN97 validation loss became non-finite"
                )
            predictions = logits.argmax(dim=-1)
            correct = int(
                ((predictions == labels) & active).sum().item()
            )
            total_loss += float(loss.detach().cpu())
            total_targets += target_count
            total_correct += correct

    return VN97EvaluationResult(
        windows=len(windows),
        target_tokens=total_targets,
        mean_loss=total_loss / total_targets,
        top1_accuracy=total_correct / total_targets,
    )


def require_release_quality(
    result: VN97EvaluationResult,
    criteria: VN97ReleaseCriteria,
) -> None:
    if result.target_tokens < criteria.min_target_tokens:
        raise VN97ReleaseQualityError(
            "validation target-token count is below release minimum"
        )
    if result.mean_loss > criteria.max_validation_loss:
        raise VN97ReleaseQualityError(
            "validation loss exceeds release maximum"
        )
    if result.top1_accuracy < criteria.min_top1_accuracy:
        raise VN97ReleaseQualityError(
            "validation top-1 accuracy is below release minimum"
        )
