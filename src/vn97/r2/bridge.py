from __future__ import annotations

import torch
import torch.nn as nn

from .model import VN97R2Model, VN97R2State


class VN97R2InferenceView(nn.Module):
    """Weight-sharing compatibility view for existing VN97 inference callers.

    The view owns no independent intelligence weights. It fixes one execution
    profile so legacy callers that invoke model(input_ids, state) can use R2
    without learning about the fast/deep keyword immediately.
    """

    def __init__(
        self,
        model: VN97R2Model,
        *,
        profile: str | int = "deep",
    ) -> None:
        super().__init__()
        self.model = model
        self.profile = profile
        self.config = model.config
        self.active_layers = model.resolve_active_layers(profile)

    def forward(
        self,
        input_ids: torch.Tensor,
        state: VN97R2State | None = None,
    ) -> tuple[torch.Tensor, VN97R2State]:
        return self.model(
            input_ids,
            state,
            profile=self.active_layers,
        )

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        state: VN97R2State | None = None,
    ) -> tuple[torch.Tensor, VN97R2State]:
        return self.model.forward_hidden(
            input_ids,
            state,
            profile=self.active_layers,
        )


def assert_tokenizer_compatible(
    model: VN97R2Model,
    tokenizer,
) -> None:
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size != model.config.vocab_size:
        raise ValueError(
            "VN97-R2/tokenizer vocabulary mismatch: "
            f"model={model.config.vocab_size}, tokenizer={vocab_size}"
        )
    for name in ("bos_id", "eos_id", "pad_id"):
        token_id = getattr(tokenizer, name, None)
        if not isinstance(token_id, int):
            raise ValueError(f"tokenizer missing integer {name}")
        if not 0 <= token_id < model.config.vocab_size:
            raise ValueError(f"tokenizer {name} outside model vocabulary")
