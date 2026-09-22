from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FactorizedEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        rank: int,
    ) -> None:
        super().__init__()
        if (
            vocab_size <= 1
            or d_model <= 0
            or not 0 < rank < d_model
        ):
            raise ValueError(
                "expected vocab_size>1 and 0<rank<d_model"
            )
        self.num_embeddings = vocab_size
        self.embedding_dim = d_model
        self.rank = rank
        self.token_factors = nn.Parameter(
            torch.empty(vocab_size, rank)
        )
        self.projection = nn.Parameter(
            torch.empty(rank, d_model)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.token_factors,
            mean=0.0,
            std=1.0
            / math.sqrt(self.rank),
        )
        nn.init.orthogonal_(
            self.projection
        )

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        factors = F.embedding(
            input_ids,
            self.token_factors,
        )
        return factors @ self.projection

    def effective_weight(
        self,
    ) -> torch.Tensor:
        return (
            self.token_factors
            @ self.projection
        )

    def parameter_count(self) -> int:
        return (
            self.token_factors.numel()
            + self.projection.numel()
        )


class FactorizedLMHead(nn.Module):
    """Output head tied exactly to a FactorizedEmbedding."""

    def __init__(
        self,
        embedding: FactorizedEmbedding,
    ) -> None:
        super().__init__()
        self.token_factors = (
            embedding.token_factors
        )
        self.projection = (
            embedding.projection
        )
        self.in_features = (
            embedding.embedding_dim
        )
        self.out_features = (
            embedding.num_embeddings
        )

    def forward(
        self,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        reduced = (
            hidden
            @ self.projection.transpose(
                0, 1
            )
        )
        return F.linear(
            reduced,
            self.token_factors,
        )

    def effective_weight(
        self,
    ) -> torch.Tensor:
        return (
            self.token_factors
            @ self.projection
        )
