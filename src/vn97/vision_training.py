from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence

import torch
import torch.nn.functional as F

from .model import VN97LanguageCore
from .modality import Modality, VisionPatchAdapter
from .tokenizer import VN97Tokenizer


@dataclass(frozen=True)
class VN97VisionExample:
    image: torch.Tensor
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.image, torch.Tensor):
            raise TypeError("vision image must be a torch.Tensor")
        if self.image.ndim != 3 or self.image.shape[0] != 3:
            raise ValueError("vision image must have shape [3,H,W]")
        if self.image.shape[1] <= 0 or self.image.shape[2] <= 0:
            raise ValueError("vision image dimensions must be positive")
        if not bool(torch.isfinite(self.image).all()):
            raise ValueError("vision image contains non-finite values")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("vision description must be non-empty text")


@dataclass(frozen=True)
class VN97VisionTrainingConfig:
    epochs: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 97
    shuffle: bool = True
    max_patches: int = 196
    max_target_tokens: int = 512

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be finite and positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.max_patches <= 0 or self.max_target_tokens <= 1:
            raise ValueError("vision bounds must be positive")


@dataclass(frozen=True)
class VN97VisionTrainingResult:
    steps: int
    examples: int
    target_tokens: int
    mean_loss: float
    final_loss: float


@dataclass(frozen=True)
class VN97VisionEvaluationResult:
    examples: int
    target_tokens: int
    mean_loss: float
    top1_accuracy: float


@dataclass(frozen=True)
class VN97VisionReleaseCriteria:
    max_validation_loss: float
    min_top1_accuracy: float = 0.0
    min_target_tokens: int = 32
    min_examples: int = 1

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_validation_loss) or self.max_validation_loss <= 0.0:
            raise ValueError("max_validation_loss must be finite and positive")
        if not math.isfinite(self.min_top1_accuracy) or not 0.0 <= self.min_top1_accuracy <= 1.0:
            raise ValueError("min_top1_accuracy must be in [0,1]")
        if self.min_target_tokens <= 0 or self.min_examples <= 0:
            raise ValueError("vision release minimums must be positive")


class VN97VisionReleaseQualityError(RuntimeError):
    pass


def _device(value: str | torch.device) -> torch.device:
    if isinstance(value, torch.device):
        return value
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _targets(
    tokenizer: VN97Tokenizer,
    description: str,
    max_target_tokens: int,
) -> torch.Tensor:
    ids = (tokenizer.text_id, *tokenizer.encode(description), tokenizer.eos_id)
    if len(ids) > max_target_tokens:
        raise ValueError("vision description exceeds max_target_tokens")
    return torch.tensor(ids, dtype=torch.long)


def _teacher_logits(
    model: VN97LanguageCore,
    adapter: VisionPatchAdapter,
    tokenizer: VN97Tokenizer,
    example: VN97VisionExample,
    *,
    max_patches: int,
    max_target_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    image = example.image.to(device=device, dtype=torch.float32).unsqueeze(0)
    embeddings = adapter(image)
    if embeddings.ndim != 3 or embeddings.shape[0] != 1:
        raise RuntimeError("VN97 vision adapter returned invalid shape")
    patch_count = int(embeddings.shape[1])
    if patch_count <= 0 or patch_count > max_patches:
        raise ValueError("vision patch count is outside configured bounds")
    if embeddings.shape[2] != model.config.d_model:
        raise RuntimeError("VN97 vision embedding width does not match core")

    targets = _targets(
        tokenizer,
        example.description,
        max_target_tokens,
    ).to(device)
    control = torch.tensor(
        [[int(Modality.VISION)]],
        dtype=torch.long,
        device=device,
    )
    pieces = [model.embedding(control), embeddings]
    if targets.numel() > 1:
        pieces.append(model.embedding(targets[:-1].unsqueeze(0)))
    inputs = torch.cat(pieces, dim=1)
    logits, _ = model.forward_embeddings(inputs)
    start = patch_count
    selected = logits[:, start : start + targets.numel(), :]
    if selected.shape != (1, targets.numel(), model.config.vocab_size):
        raise RuntimeError("VN97 vision teacher-forcing alignment is invalid")
    return selected.squeeze(0), targets


def train_vn97_vision_adapter(
    model: VN97LanguageCore,
    adapter: VisionPatchAdapter,
    tokenizer: VN97Tokenizer,
    examples: Sequence[VN97VisionExample],
    config: VN97VisionTrainingConfig,
    *,
    device: str | torch.device = "auto",
) -> VN97VisionTrainingResult:
    if not examples:
        raise ValueError("vision examples must not be empty")
    resolved = _device(device)
    torch.manual_seed(config.seed)
    model.to(resolved).eval()
    adapter.to(resolved).train()
    previous = {name: p.requires_grad for name, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    rng = random.Random(config.seed)
    losses: list[float] = []
    target_tokens = 0
    steps = 0
    try:
        for _ in range(config.epochs):
            indices = list(range(len(examples)))
            if config.shuffle:
                rng.shuffle(indices)
            for index in indices:
                optimizer.zero_grad(set_to_none=True)
                logits, targets = _teacher_logits(
                    model,
                    adapter,
                    tokenizer,
                    examples[index],
                    max_patches=config.max_patches,
                    max_target_tokens=config.max_target_tokens,
                    device=resolved,
                )
                loss = F.cross_entropy(logits, targets)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("VN97 vision training loss became non-finite")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    adapter.parameters(),
                    config.max_grad_norm,
                )
                if not bool(torch.isfinite(torch.as_tensor(norm))):
                    raise RuntimeError("VN97 vision gradient norm became non-finite")
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                target_tokens += int(targets.numel())
                steps += 1
    finally:
        for name, p in model.named_parameters():
            p.requires_grad_(previous[name])
        model.eval()
        adapter.eval()
    return VN97VisionTrainingResult(
        steps=steps,
        examples=len(examples),
        target_tokens=target_tokens,
        mean_loss=sum(losses) / len(losses),
        final_loss=losses[-1],
    )


@torch.no_grad()
def evaluate_vn97_vision(
    model: VN97LanguageCore,
    adapter: VisionPatchAdapter,
    tokenizer: VN97Tokenizer,
    examples: Sequence[VN97VisionExample],
    *,
    config: VN97VisionTrainingConfig = VN97VisionTrainingConfig(
        epochs=1,
        shuffle=False,
    ),
    device: str | torch.device = "auto",
) -> VN97VisionEvaluationResult:
    if not examples:
        raise ValueError("vision evaluation examples must not be empty")
    resolved = _device(device)
    model.to(resolved).eval()
    adapter.to(resolved).eval()
    total_loss = 0.0
    correct = 0
    target_tokens = 0
    for example in examples:
        logits, targets = _teacher_logits(
            model,
            adapter,
            tokenizer,
            example,
            max_patches=config.max_patches,
            max_target_tokens=config.max_target_tokens,
            device=resolved,
        )
        loss = F.cross_entropy(logits, targets, reduction="sum")
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("VN97 vision evaluation loss became non-finite")
        total_loss += float(loss.cpu())
        correct += int((logits.argmax(dim=-1) == targets).sum().item())
        target_tokens += int(targets.numel())
    return VN97VisionEvaluationResult(
        examples=len(examples),
        target_tokens=target_tokens,
        mean_loss=total_loss / target_tokens,
        top1_accuracy=correct / target_tokens,
    )


def require_vision_release_quality(
    result: VN97VisionEvaluationResult,
    criteria: VN97VisionReleaseCriteria,
) -> None:
    failures: list[str] = []
    if result.examples < criteria.min_examples:
        failures.append("examples below minimum")
    if result.target_tokens < criteria.min_target_tokens:
        failures.append("target_tokens below minimum")
    if result.mean_loss > criteria.max_validation_loss:
        failures.append("mean_loss exceeds maximum")
    if result.top1_accuracy < criteria.min_top1_accuracy:
        failures.append("top1_accuracy below minimum")
    if failures:
        raise VN97VisionReleaseQualityError(
            "VN97 vision release quality gate failed: " + "; ".join(failures)
        )
