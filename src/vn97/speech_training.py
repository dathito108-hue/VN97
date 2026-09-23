from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence

import torch
import torch.nn.functional as F

from .model import VN97LanguageCore
from .modality import AudioFrameAdapter, Modality
from .tokenizer import VN97Tokenizer


@dataclass(frozen=True)
class VN97SpeechExample:
    waveform: torch.Tensor
    transcript: str

    def __post_init__(self) -> None:
        if not isinstance(self.waveform, torch.Tensor):
            raise TypeError("speech waveform must be a torch.Tensor")
        if self.waveform.ndim != 1 or self.waveform.numel() <= 0:
            raise ValueError("speech waveform must have shape [samples]")
        if not bool(torch.isfinite(self.waveform).all()):
            raise ValueError("speech waveform contains non-finite values")
        if not isinstance(self.transcript, str) or not self.transcript.strip():
            raise ValueError("speech transcript must be non-empty text")


@dataclass(frozen=True)
class VN97SpeechTrainingConfig:
    epochs: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 97
    shuffle: bool = True
    max_frames: int = 1500
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
        if self.max_frames <= 0 or self.max_target_tokens <= 1:
            raise ValueError("speech bounds must be positive")


@dataclass(frozen=True)
class VN97SpeechTrainingResult:
    steps: int
    examples: int
    target_tokens: int
    mean_loss: float
    final_loss: float

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.examples <= 0 or self.target_tokens <= 0:
            raise ValueError("speech training result must contain work")
        if not math.isfinite(self.mean_loss) or not math.isfinite(self.final_loss):
            raise ValueError("speech training losses must be finite")


@dataclass(frozen=True)
class VN97SpeechEvaluationResult:
    examples: int
    target_tokens: int
    mean_loss: float
    top1_accuracy: float

    def __post_init__(self) -> None:
        if self.examples <= 0 or self.target_tokens <= 0:
            raise ValueError("speech evaluation must contain targets")
        if not math.isfinite(self.mean_loss) or self.mean_loss < 0.0:
            raise ValueError("speech evaluation loss must be finite and non-negative")
        if (
            not math.isfinite(self.top1_accuracy)
            or not 0.0 <= self.top1_accuracy <= 1.0
        ):
            raise ValueError("speech top1_accuracy must be in [0, 1]")


@dataclass(frozen=True)
class VN97SpeechReleaseCriteria:
    max_validation_loss: float
    min_top1_accuracy: float = 0.0
    min_target_tokens: int = 32
    min_examples: int = 1

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
        if self.min_target_tokens <= 0 or self.min_examples <= 0:
            raise ValueError("speech release minimums must be positive")


class VN97SpeechReleaseQualityError(RuntimeError):
    pass


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _target_ids(
    tokenizer: VN97Tokenizer,
    transcript: str,
    *,
    max_target_tokens: int,
) -> torch.Tensor:
    content = tokenizer.encode(transcript)
    ids = (tokenizer.text_id, *content, tokenizer.eos_id)
    if len(ids) > max_target_tokens:
        raise ValueError("speech transcript exceeds max_target_tokens")
    return torch.tensor(ids, dtype=torch.long)


def _speech_teacher_forced_logits(
    model: VN97LanguageCore,
    audio_adapter: AudioFrameAdapter,
    tokenizer: VN97Tokenizer,
    example: VN97SpeechExample,
    *,
    max_frames: int,
    max_target_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    waveform = example.waveform.to(
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)
    audio_embeddings = audio_adapter(waveform)
    if audio_embeddings.ndim != 3 or audio_embeddings.shape[0] != 1:
        raise RuntimeError("VN97 audio adapter returned invalid embedding shape")
    frame_count = int(audio_embeddings.shape[1])
    if frame_count <= 0 or frame_count > max_frames:
        raise ValueError("speech example frame count is outside configured bounds")
    if audio_embeddings.shape[2] != model.config.d_model:
        raise RuntimeError("VN97 audio embedding width does not match language core")

    targets = _target_ids(
        tokenizer,
        example.transcript,
        max_target_tokens=max_target_tokens,
    ).to(device)
    audio_control = torch.tensor(
        [[int(Modality.AUDIO)]],
        dtype=torch.long,
        device=device,
    )
    prefix = model.embedding(audio_control)

    pieces = [prefix, audio_embeddings]
    if targets.numel() > 1:
        pieces.append(model.embedding(targets[:-1].unsqueeze(0)))
    inputs = torch.cat(pieces, dim=1)

    logits, _ = model.forward_embeddings(inputs)
    start = frame_count
    selected = logits[:, start : start + targets.numel(), :]
    if selected.shape != (1, targets.numel(), model.config.vocab_size):
        raise RuntimeError("VN97 speech teacher-forcing alignment is invalid")
    return selected.squeeze(0), targets


def train_vn97_speech_adapter(
    model: VN97LanguageCore,
    audio_adapter: AudioFrameAdapter,
    tokenizer: VN97Tokenizer,
    examples: Sequence[VN97SpeechExample],
    config: VN97SpeechTrainingConfig,
    *,
    device: str | torch.device = "auto",
) -> VN97SpeechTrainingResult:
    if not isinstance(model, VN97LanguageCore):
        raise TypeError("model must be VN97LanguageCore")
    if not isinstance(audio_adapter, AudioFrameAdapter):
        raise TypeError("audio_adapter must be AudioFrameAdapter")
    if audio_adapter.projection.out_features != model.config.d_model:
        raise ValueError("audio adapter d_model must match language core")
    if not examples:
        raise ValueError("speech examples must not be empty")

    resolved = _resolve_device(device)
    torch.manual_seed(config.seed)
    if resolved.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    model.to(resolved)
    audio_adapter.to(resolved)
    model.eval()
    audio_adapter.train()

    original_requires_grad = {
        name: parameter.requires_grad
        for name, parameter in model.named_parameters()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        audio_adapter.parameters(),
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
                logits, targets = _speech_teacher_forced_logits(
                    model,
                    audio_adapter,
                    tokenizer,
                    examples[index],
                    max_frames=config.max_frames,
                    max_target_tokens=config.max_target_tokens,
                    device=resolved,
                )
                loss = F.cross_entropy(logits, targets)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("VN97 speech training loss became non-finite")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    audio_adapter.parameters(),
                    config.max_grad_norm,
                )
                if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
                    raise RuntimeError(
                        "VN97 speech training gradient norm became non-finite"
                    )
                optimizer.step()

                losses.append(float(loss.detach().cpu()))
                target_tokens += int(targets.numel())
                steps += 1
    finally:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(original_requires_grad[name])
        model.eval()
        audio_adapter.eval()

    if not losses:
        raise RuntimeError("VN97 speech training completed without updates")
    return VN97SpeechTrainingResult(
        steps=steps,
        examples=len(examples),
        target_tokens=target_tokens,
        mean_loss=sum(losses) / len(losses),
        final_loss=losses[-1],
    )


@torch.no_grad()
def evaluate_vn97_speech(
    model: VN97LanguageCore,
    audio_adapter: AudioFrameAdapter,
    tokenizer: VN97Tokenizer,
    examples: Sequence[VN97SpeechExample],
    *,
    config: VN97SpeechTrainingConfig = VN97SpeechTrainingConfig(
        epochs=1,
        shuffle=False,
    ),
    device: str | torch.device = "auto",
) -> VN97SpeechEvaluationResult:
    if not examples:
        raise ValueError("speech evaluation examples must not be empty")
    resolved = _resolve_device(device)
    model.to(resolved).eval()
    audio_adapter.to(resolved).eval()

    loss_sum = 0.0
    correct = 0
    target_tokens = 0
    for example in examples:
        logits, targets = _speech_teacher_forced_logits(
            model,
            audio_adapter,
            tokenizer,
            example,
            max_frames=config.max_frames,
            max_target_tokens=config.max_target_tokens,
            device=resolved,
        )
        losses = F.cross_entropy(
            logits,
            targets,
            reduction="sum",
        )
        if not bool(torch.isfinite(losses)):
            raise RuntimeError("VN97 speech evaluation loss became non-finite")
        loss_sum += float(losses.cpu())
        predictions = logits.argmax(dim=-1)
        correct += int((predictions == targets).sum().item())
        target_tokens += int(targets.numel())

    return VN97SpeechEvaluationResult(
        examples=len(examples),
        target_tokens=target_tokens,
        mean_loss=loss_sum / target_tokens,
        top1_accuracy=correct / target_tokens,
    )


def require_speech_release_quality(
    result: VN97SpeechEvaluationResult,
    criteria: VN97SpeechReleaseCriteria,
) -> None:
    failures: list[str] = []
    if result.examples < criteria.min_examples:
        failures.append(
            f"examples {result.examples} < {criteria.min_examples}"
        )
    if result.target_tokens < criteria.min_target_tokens:
        failures.append(
            f"target_tokens {result.target_tokens} < {criteria.min_target_tokens}"
        )
    if result.mean_loss > criteria.max_validation_loss:
        failures.append(
            f"mean_loss {result.mean_loss:.6f} > "
            f"{criteria.max_validation_loss:.6f}"
        )
    if result.top1_accuracy < criteria.min_top1_accuracy:
        failures.append(
            f"top1_accuracy {result.top1_accuracy:.6f} < "
            f"{criteria.min_top1_accuracy:.6f}"
        )
    if failures:
        raise VN97SpeechReleaseQualityError(
            "VN97 speech release quality gate failed: " + "; ".join(failures)
        )
