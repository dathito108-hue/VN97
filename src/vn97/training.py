from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from .model import VN97LanguageCore
from .tokenizer import VN97Tokenizer


IGNORE_INDEX = -100
_ROLE_MARKERS = {
    "system": "\n<|system|>\n",
    "user": "\n<|user|>\n",
    "assistant": "\n<|assistant|>\n",
}


@dataclass(frozen=True)
class VN97ChatMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in _ROLE_MARKERS:
            raise ValueError("chat role must be system, user, or assistant")
        if not isinstance(self.content, str) or not self.content:
            raise ValueError("chat content must be non-empty text")


@dataclass(frozen=True)
class VN97TrainingExample:
    token_ids: tuple[int, ...]
    target_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.target_mask):
            raise ValueError("token_ids and target_mask lengths must match")
        if len(self.token_ids) < 2:
            raise ValueError("training example must contain at least two tokens")
        if any(type(value) is not int or value < 0 for value in self.token_ids):
            raise ValueError("training token IDs must be non-negative integers")
        if not any(self.target_mask[1:]):
            raise ValueError("training example must contain at least one target token")


@dataclass(frozen=True)
class VN97TrainingWindow:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    target_tokens: int

    def __post_init__(self) -> None:
        if len(self.input_ids) != len(self.labels) or not self.input_ids:
            raise ValueError("training window input/label shapes must match")
        if self.target_tokens <= 0:
            raise ValueError("training window must contain target tokens")
        if sum(label != IGNORE_INDEX for label in self.labels) != self.target_tokens:
            raise ValueError("training window target count mismatch")


@dataclass(frozen=True)
class VN97TrainingConfig:
    sequence_length: int = 256
    stride: int | None = None
    batch_size: int = 4
    epochs: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 97
    shuffle: bool = True

    def __post_init__(self) -> None:
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if self.stride is not None and not 1 <= self.stride <= self.sequence_length:
            raise ValueError("stride must be in [1, sequence_length]")
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be finite and positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True)
class VN97TrainingResult:
    steps: int
    target_tokens: int
    mean_loss: float
    final_loss: float

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.target_tokens <= 0:
            raise ValueError("training result must contain work")
        if not math.isfinite(self.mean_loss) or not math.isfinite(self.final_loss):
            raise ValueError("training result losses must be finite")


def encode_causal_text(
    tokenizer: VN97Tokenizer,
    text: str,
) -> VN97TrainingExample:
    if not isinstance(text, str) or not text:
        raise ValueError("causal training text must be non-empty")
    content = tokenizer.encode(text)
    tokens = (
        tokenizer.bos_id,
        tokenizer.text_id,
        *content,
        tokenizer.eos_id,
    )
    mask = (
        False,
        False,
        *([True] * len(content)),
        True,
    )
    return VN97TrainingExample(tuple(tokens), tuple(mask))


def encode_chat_messages(
    tokenizer: VN97Tokenizer,
    messages: Sequence[VN97ChatMessage],
) -> VN97TrainingExample:
    if not messages:
        raise ValueError("chat training example must contain messages")

    tokens: list[int] = [tokenizer.bos_id, tokenizer.text_id]
    mask: list[bool] = [False, False]
    assistant_targets = 0

    for message in messages:
        prefix = tokenizer.encode(_ROLE_MARKERS[message.role])
        content = tokenizer.encode(message.content)
        suffix = tokenizer.encode("\n")
        target = message.role == "assistant"

        tokens.extend(prefix)
        mask.extend([target] * len(prefix))
        tokens.extend(content)
        mask.extend([target] * len(content))
        tokens.extend(suffix)
        mask.extend([target] * len(suffix))

        if target:
            assistant_targets += len(prefix) + len(content) + len(suffix)

    tokens.append(tokenizer.eos_id)
    eos_target = messages[-1].role == "assistant"
    mask.append(eos_target)
    if eos_target:
        assistant_targets += 1

    if assistant_targets == 0:
        raise ValueError("chat training example must contain an assistant target")
    return VN97TrainingExample(tuple(tokens), tuple(mask))


def render_chat_text(
    messages: Sequence[VN97ChatMessage],
) -> str:
    if not messages:
        raise ValueError("chat messages must not be empty")
    return "".join(
        _ROLE_MARKERS[message.role] + message.content + "\n"
        for message in messages
    )


def make_training_windows(
    example: VN97TrainingExample,
    *,
    sequence_length: int,
    stride: int | None = None,
    pad_token_id: int = 0,
) -> tuple[VN97TrainingWindow, ...]:
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    if stride is None:
        stride = sequence_length
    if not 1 <= stride <= sequence_length:
        raise ValueError("stride must be in [1, sequence_length]")
    if pad_token_id < 0:
        raise ValueError("pad_token_id must be non-negative")

    output: list[VN97TrainingWindow] = []
    token_ids = example.token_ids
    target_mask = example.target_mask

    for start in range(0, len(token_ids) - 1, stride):
        inputs = list(token_ids[start : start + sequence_length])
        targets = list(token_ids[start + 1 : start + sequence_length + 1])
        masks = list(target_mask[start + 1 : start + sequence_length + 1])
        labels = [
            token if enabled else IGNORE_INDEX
            for token, enabled in zip(targets, masks)
        ]

        if not any(label != IGNORE_INDEX for label in labels):
            if start + sequence_length >= len(token_ids) - 1:
                break
            continue

        pad_count = sequence_length - len(inputs)
        if pad_count < 0:
            raise AssertionError("training window exceeded sequence length")
        inputs.extend([pad_token_id] * pad_count)
        labels.extend([IGNORE_INDEX] * (sequence_length - len(labels)))

        output.append(
            VN97TrainingWindow(
                input_ids=tuple(inputs),
                labels=tuple(labels),
                target_tokens=sum(label != IGNORE_INDEX for label in labels),
            )
        )
        if start + sequence_length >= len(token_ids) - 1:
            break

    if not output:
        raise ValueError("example produced no supervised training windows")
    return tuple(output)


def build_training_windows(
    examples: Iterable[VN97TrainingExample],
    config: VN97TrainingConfig,
    *,
    pad_token_id: int = 0,
) -> tuple[VN97TrainingWindow, ...]:
    output: list[VN97TrainingWindow] = []
    for example in examples:
        output.extend(
            make_training_windows(
                example,
                sequence_length=config.sequence_length,
                stride=config.stride,
                pad_token_id=pad_token_id,
            )
        )
    if not output:
        raise ValueError("training set produced no windows")
    return tuple(output)


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def train_vn97_language(
    model: VN97LanguageCore,
    windows: Sequence[VN97TrainingWindow],
    config: VN97TrainingConfig,
    *,
    device: str | torch.device = "auto",
) -> VN97TrainingResult:
    if not isinstance(model, VN97LanguageCore):
        raise TypeError("model must be VN97LanguageCore")
    if not windows:
        raise ValueError("windows must not be empty")
    if any(len(window.input_ids) != config.sequence_length for window in windows):
        raise ValueError("all windows must match training sequence_length")

    resolved = _resolve_device(device)
    torch.manual_seed(config.seed)
    if resolved.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    model.to(resolved)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    rng = random.Random(config.seed)
    losses: list[float] = []
    trained_tokens = 0
    steps = 0

    for epoch in range(config.epochs):
        indices = list(range(len(windows)))
        if config.shuffle:
            rng.shuffle(indices)

        for batch_start in range(0, len(indices), config.batch_size):
            batch_indices = indices[batch_start : batch_start + config.batch_size]
            batch = [windows[index] for index in batch_indices]
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

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(inputs)
            if logits.ndim != 3 or logits.shape[:2] != inputs.shape:
                raise RuntimeError("VN97 model returned invalid training logits shape")
            if logits.shape[-1] != model.config.vocab_size:
                raise RuntimeError("VN97 model logits vocabulary mismatch")

            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("VN97 training loss became non-finite")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.max_grad_norm,
            )
            if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
                raise RuntimeError("VN97 training gradient norm became non-finite")
            optimizer.step()

            value = float(loss.detach().cpu())
            losses.append(value)
            trained_tokens += int((labels != IGNORE_INDEX).sum().item())
            steps += 1

    model.eval()
    if not losses or steps <= 0 or trained_tokens <= 0:
        raise RuntimeError("VN97 training completed without supervised updates")
    return VN97TrainingResult(
        steps=steps,
        target_tokens=trained_tokens,
        mean_loss=sum(losses) / len(losses),
        final_loss=losses[-1],
    )
