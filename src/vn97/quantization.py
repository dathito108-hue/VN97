from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from .packing import PackedTernaryMatrix


def ternary_symbols_and_scales(
    weight: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return int8 {-1, 0, +1} symbols and one positive scale per output row."""
    if weight.ndim != 2:
        raise ValueError(
            "ternary linear weights must be rank-2 [out_features, in_features]"
        )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    scales = weight.abs().mean(dim=1).clamp_min(1e-8)
    normalized = weight / scales.unsqueeze(1)
    symbols = torch.where(
        normalized.abs() >= threshold,
        normalized.sign(),
        torch.zeros_like(normalized),
    ).to(torch.int8)
    return symbols, scales


class _PerChannelTernarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, threshold: float) -> torch.Tensor:
        symbols, scales = ternary_symbols_and_scales(weight, float(threshold))
        return symbols.to(dtype=weight.dtype) * scales.unsqueeze(1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


def quantize_ternary_per_channel(
    weight: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    return _PerChannelTernarySTE.apply(weight, float(threshold))


class TernaryLinear(nn.Module):
    """Training/reference linear layer whose deployment representation is packed ternary."""

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
        self.float_shadow_enabled = False
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

    def effective_weight(self) -> torch.Tensor:
        if self.float_shadow_enabled:
            return self.weight
        return self.quantized_weight()

    def set_float_shadow(self, enabled: bool) -> None:
        self.float_shadow_enabled = bool(enabled)

    def export_packed(
        self,
        *,
        tile_rows: int = 16,
        tile_cols: int = 16,
    ) -> "PackedTernaryMatrix":
        from .packing import pack_ternary_weight

        return pack_ternary_weight(
            self.weight.detach(),
            threshold=self.threshold,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.bias)


def set_float_shadow_mode(
    module: nn.Module,
    enabled: bool,
) -> int:
    """Switch all VN97 ternary projections between dense shadow and ternary STE.

    The flag is runtime-only and is intentionally absent from state_dict().
    Deployment therefore remains ternary by default even when a checkpoint was
    trained in float-shadow mode.
    """
    count = 0
    for child in module.modules():
        if isinstance(child, TernaryLinear):
            child.set_float_shadow(
                enabled
            )
            count += 1
    return count
