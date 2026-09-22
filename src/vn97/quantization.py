from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PerChannelTernarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, threshold: float) -> torch.Tensor:
        scale = weight.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
        normalized = weight / scale
        ternary = torch.where(
            normalized.abs() >= threshold,
            normalized.sign(),
            torch.zeros_like(normalized),
        )
        return ternary * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


def quantize_ternary_per_channel(
    weight: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError(
            "ternary linear weights must be rank-2 [out_features, in_features]"
        )
    return _PerChannelTernarySTE.apply(weight, float(threshold))


class TernaryLinear(nn.Module):
    """Numerical reference for a future packed ternary native linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def quantized_weight(self) -> torch.Tensor:
        return quantize_ternary_per_channel(self.weight, self.threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.quantized_weight(), self.bias)
