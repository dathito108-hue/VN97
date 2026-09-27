from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import torch

from ..tokenizer import VN97Tokenizer, VN97TokenizerPackage
from ..training import (
    VN97ChatMessage,
    VN97TrainingConfig,
    VN97TrainingWindow,
    build_training_windows,
    encode_causal_text,
    encode_chat_completion_messages,
)
from .bridge import assert_tokenizer_compatible
from .model import VN97R2Model


def load_vn97tk1(path: str | Path) -> VN97Tokenizer:
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("VN97TK1 path must be a regular file")
    package = VN97TokenizerPackage.from_bytes(resolved.read_bytes())
    return VN97Tokenizer(package)


def build_completion_windows(
    tokenizer: VN97Tokenizer,
    conversations: Iterable[Sequence[VN97ChatMessage]],
    config: VN97TrainingConfig,
) -> tuple[VN97TrainingWindow, ...]:
    examples = [
        encode_chat_completion_messages(tokenizer, tuple(messages))
        for messages in conversations
    ]
    return build_training_windows(
        examples,
        config,
        pad_token_id=tokenizer.pad_id,
    )


def build_causal_windows(
    tokenizer: VN97Tokenizer,
    texts: Iterable[str],
    config: VN97TrainingConfig,
) -> tuple[VN97TrainingWindow, ...]:
    examples = [
        encode_causal_text(tokenizer, text)
        for text in texts
    ]
    return build_training_windows(
        examples,
        config,
        pad_token_id=tokenizer.pad_id,
    )


def windows_to_tensors(
    windows: Sequence[VN97TrainingWindow],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not windows:
        raise ValueError("windows must not be empty")
    width = len(windows[0].input_ids)
    if any(
        len(window.input_ids) != width
        or len(window.labels) != width
        for window in windows
    ):
        raise ValueError("training windows must have a common width")
    return (
        torch.tensor(
            [window.input_ids for window in windows],
            dtype=torch.long,
        ),
        torch.tensor(
            [window.labels for window in windows],
            dtype=torch.long,
        ),
    )


def assert_r2_data_compatible(
    model: VN97R2Model,
    tokenizer: VN97Tokenizer,
    windows: Sequence[VN97TrainingWindow],
) -> None:
    assert_tokenizer_compatible(model, tokenizer)
    if not windows:
        raise ValueError("windows must not be empty")
    for window in windows:
        for token in window.input_ids:
            if token < 0 or token >= model.config.vocab_size:
                raise ValueError(
                    "training input token outside R2 vocabulary"
                )
        for label in window.labels:
            if label != -100 and (
                label < 0 or label >= model.config.vocab_size
            ):
                raise ValueError(
                    "training label outside R2 vocabulary"
                )
