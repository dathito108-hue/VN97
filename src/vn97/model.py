from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from .config import VN97Config
from .embedding import FactorizedEmbedding, FactorizedLMHead
from .ssm import RMSNorm, VN97Block


VN97States = list[torch.Tensor]


class VN97LanguageCore(nn.Module):
    """Reference recurrent language core for the VN97 system architecture."""

    def __init__(self, config: VN97Config) -> None:
        super().__init__()
        self.config = config
        if config.embedding_rank is None:
            self.embedding = nn.Embedding(
                config.vocab_size, config.d_model
            )
            self.lm_head = nn.Linear(
                config.d_model, config.vocab_size, bias=False
            )
            self.lm_head.weight = self.embedding.weight
        else:
            self.embedding = FactorizedEmbedding(
                config.vocab_size,
                config.d_model,
                config.embedding_rank,
            )
            self.lm_head = FactorizedLMHead(
                self.embedding
            )
        self.layers = nn.ModuleList(
            [
                VN97Block(
                    config.d_model,
                    config.d_state,
                    ternary_threshold=config.ternary_threshold,
                    dt_min=config.dt_min,
                    dt_max=config.dt_max,
                    min_decay=config.min_decay,
                    max_decay=config.max_decay,
                    rms_eps=config.rms_eps,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = RMSNorm(
            config.d_model, eps=config.rms_eps
        )

    def initial_states(
        self, batch_size: int, *, device, dtype
    ) -> VN97States:
        return [
            layer.core.initial_state(
                batch_size, device=device, dtype=dtype
            )
            for layer in self.layers
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        states: Optional[
            Sequence[Optional[torch.Tensor]]
        ] = None,
    ) -> tuple[torch.Tensor, VN97States]:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape [batch, seq]"
            )
        x = self.embedding(input_ids)

        if states is None:
            layer_states: Sequence[
                Optional[torch.Tensor]
            ] = [None] * len(self.layers)
        else:
            if len(states) != len(self.layers):
                raise ValueError(
                    "number of recurrent states must match n_layers"
                )
            layer_states = states

        new_states: VN97States = []
        for layer, state in zip(
            self.layers, layer_states
        ):
            x, new_state = layer(x, state)
            new_states.append(new_state)

        logits = self.lm_head(
            self.final_norm(x)
        )
        return logits, new_states

    @torch.no_grad()
    def generate_greedy(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        if max_new_tokens < 0:
            raise ValueError(
                "max_new_tokens must be non-negative"
            )
        if max_new_tokens == 0:
            return input_ids.new_empty(
                (input_ids.shape[0], 0)
            )

        self.eval()
        logits, states = self(input_ids)
        token = logits[:, -1].argmax(
            dim=-1, keepdim=True
        )
        generated = [token]

        for _ in range(max_new_tokens - 1):
            logits, states = self(
                token, states
            )
            token = logits[:, -1].argmax(
                dim=-1, keepdim=True
            )
            generated.append(token)

        return torch.cat(generated, dim=1)
