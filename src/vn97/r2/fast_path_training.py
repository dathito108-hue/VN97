from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

import torch
import torch.nn.functional as F

from ..training import IGNORE_INDEX, VN97TrainingWindow
from .model import VN97R2Model


@dataclass(frozen=True)
class R2FastPathConfig:
    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    temperature: float = 2.0
    distill_weight: float = 0.5
    deep_preservation_weight: float = 0.5
    seed: int = 9704

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.distill_weight < 0.0:
            raise ValueError("distill_weight must be non-negative")
        if self.deep_preservation_weight < 0.0:
            raise ValueError(
                "deep_preservation_weight must be non-negative"
            )


@dataclass(frozen=True)
class R2FastPathResult:
    steps: int
    target_tokens: int
    mean_loss: float
    final_loss: float


def _batch(
    windows: Sequence[VN97TrainingWindow],
    indices: Sequence[int],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [windows[index] for index in indices]
    return (
        torch.tensor(
            [row.input_ids for row in rows],
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            [row.labels for row in rows],
            dtype=torch.long,
            device=device,
        ),
    )


def _configure_fast_alignment_trainable(
    model: VN97R2Model,
) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable: list[torch.nn.Parameter] = []
    for index in range(model.config.fast_layers):
        for parameter in model.layers[index].parameters():
            parameter.requires_grad_(True)
            trainable.append(parameter)
    return trainable


def align_fast_path(
    model: VN97R2Model,
    windows: Sequence[VN97TrainingWindow],
    config: R2FastPathConfig,
    *,
    device: str | torch.device = "auto",
) -> R2FastPathResult:
    """Align the early-exit path against the same model's deep path.

    Only the leading fast layers are updated. Embedding, tied lexical head,
    final norm, and later layers remain frozen. A supervised deep-path term
    constrains shared early layers so fast alignment cannot ignore full-depth
    task performance.
    """
    if not windows:
        raise ValueError("windows must not be empty")
    resolved = (
        torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if device == "auto"
        else torch.device(device)
    )
    torch.manual_seed(config.seed)
    if resolved.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    model.to(resolved)
    trainable = _configure_fast_alignment_trainable(model)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    rng = random.Random(config.seed)
    losses: list[float] = []
    target_tokens = 0
    steps = 0

    for epoch in range(config.epochs):
        order = list(range(len(windows)))
        rng.shuffle(order)
        for start in range(0, len(order), config.batch_size):
            indices = order[start : start + config.batch_size]
            inputs, labels = _batch(
                windows,
                indices,
                device=resolved,
            )
            mask = labels != IGNORE_INDEX
            if not bool(mask.any()):
                continue

            optimizer.zero_grad(set_to_none=True)
            deep_logits, _ = model(inputs, profile="deep")
            fast_logits, _ = model(inputs, profile="fast")

            deep_selected = deep_logits[mask].float()
            fast_selected = fast_logits[mask].float()
            targets = labels[mask]

            fast_ce = F.cross_entropy(
                fast_selected,
                targets,
            )
            deep_ce = F.cross_entropy(
                deep_selected,
                targets,
            )

            temperature = config.temperature
            teacher = F.softmax(
                deep_selected.detach() / temperature,
                dim=-1,
            )
            student = F.log_softmax(
                fast_selected / temperature,
                dim=-1,
            )
            distill = F.kl_div(
                student,
                teacher,
                reduction="batchmean",
            ) * (temperature * temperature)

            loss = (
                fast_ce
                + config.distill_weight * distill
                + config.deep_preservation_weight * deep_ce
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(
                    "R2 fast-path alignment loss became non-finite"
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                config.max_grad_norm,
            )
            if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
                raise RuntimeError(
                    "R2 fast-path alignment gradient became non-finite"
                )
            optimizer.step()

            value = float(loss.detach().cpu())
            losses.append(value)
            target_tokens += int(mask.sum().item())
            steps += 1

    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.eval()

    if not losses:
        raise RuntimeError("R2 fast-path alignment completed without work")
    return R2FastPathResult(
        steps=steps,
        target_tokens=target_tokens,
        mean_loss=sum(losses) / len(losses),
        final_loss=losses[-1],
    )
