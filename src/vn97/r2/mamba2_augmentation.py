from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


@dataclass
class VN97MultiTimescaleLayerState:
    fast: torch.Tensor
    working: torch.Tensor
    slow: torch.Tensor

    def detach(self) -> "VN97MultiTimescaleLayerState":
        return VN97MultiTimescaleLayerState(
            fast=self.fast.detach(),
            working=self.working.detach(),
            slow=self.slow.detach(),
        )


class VN97Mamba2LayerAugmentation(nn.Module):
    """Parameter-efficient VN97 augmentation around one preserved Mamba-2 block.

    The original block output is not modified when highway_alpha and memory_alpha
    are zero. Both are initialized to exact zero, so a freshly transferred G0
    model has an exact baseline path while the shadow states may already update.
    """

    def __init__(
        self,
        d_model: int,
        *,
        fast_decay: float = 0.50,
        working_decay: float = 0.95,
        slow_decay: float = 0.995,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        for label, value in (
            ("fast_decay", fast_decay),
            ("working_decay", working_decay),
            ("slow_decay", slow_decay),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{label} must be inside (0, 1)")
        if not fast_decay < working_decay < slow_decay:
            raise ValueError(
                "timescales must satisfy fast < working < slow decay"
            )

        self.d_model = int(d_model)
        decays = torch.tensor(
            [fast_decay, working_decay, slow_decay],
            dtype=torch.float32,
        )
        decay_logits = torch.logit(decays)
        self.decay_logits = nn.Parameter(
            decay_logits[:, None].expand(3, d_model).clone()
        )
        self.highway_mix_logits = nn.Parameter(
            torch.zeros(3, d_model, dtype=torch.float32)
        )
        self.highway_alpha = nn.Parameter(
            torch.zeros(d_model, dtype=torch.float32)
        )
        self.memory_gate_logit = nn.Parameter(
            torch.zeros(d_model, dtype=torch.float32)
        )
        self.memory_alpha = nn.Parameter(
            torch.zeros(d_model, dtype=torch.float32)
        )

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> VN97MultiTimescaleLayerState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        shape = (batch_size, self.d_model)
        zeros = torch.zeros(shape, device=device, dtype=dtype)
        return VN97MultiTimescaleLayerState(
            fast=zeros.clone(),
            working=zeros.clone(),
            slow=zeros.clone(),
        )

    def zero_impact(self) -> bool:
        return bool(
            torch.count_nonzero(self.highway_alpha).item() == 0
            and torch.count_nonzero(self.memory_alpha).item() == 0
        )

    def forward(
        self,
        hidden: torch.Tensor,
        state: VN97MultiTimescaleLayerState,
        *,
        retrieved_memory: torch.Tensor | None = None,
        enabled: bool = True,
    ) -> tuple[torch.Tensor, VN97MultiTimescaleLayerState]:
        if hidden.ndim != 2 or hidden.shape[-1] != self.d_model:
            raise ValueError(
                "hidden must be [batch, d_model], got "
                f"{tuple(hidden.shape)}"
            )
        for label, value in (
            ("fast", state.fast),
            ("working", state.working),
            ("slow", state.slow),
        ):
            if value.shape != hidden.shape:
                raise ValueError(f"{label} state shape mismatch")

        decays = torch.sigmoid(self.decay_logits).to(
            device=hidden.device,
            dtype=hidden.dtype,
        )
        current = (state.fast, state.working, state.slow)
        updated = tuple(
            decay[index] * current[index]
            + (1.0 - decay[index]) * hidden
            for index, decay in enumerate(decays)
        )
        next_state = VN97MultiTimescaleLayerState(
            fast=updated[0],
            working=updated[1],
            slow=updated[2],
        )

        if not enabled:
            return hidden, next_state

        mix = torch.softmax(
            self.highway_mix_logits,
            dim=0,
        ).to(device=hidden.device, dtype=hidden.dtype)
        highway = (
            mix[0] * next_state.fast
            + mix[1] * next_state.working
            + mix[2] * next_state.slow
        )
        output = hidden + self.highway_alpha.to(
            hidden.dtype
        ) * highway

        if retrieved_memory is not None:
            if retrieved_memory.shape != hidden.shape:
                raise ValueError(
                    "retrieved_memory must be [batch, d_model]"
                )
            gate = torch.sigmoid(self.memory_gate_logit).to(
                device=hidden.device,
                dtype=hidden.dtype,
            )
            output = output + (
                self.memory_alpha.to(hidden.dtype)
                * gate
                * retrieved_memory
            )
        return output, next_state


class VN97Mamba2AugmentationBank(nn.Module):
    """One zero-impact augmentation module per preserved Mamba-2 layer."""

    def __init__(self, d_model: int, n_layers: int) -> None:
        super().__init__()
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.layers = nn.ModuleList(
            VN97Mamba2LayerAugmentation(d_model)
            for _ in range(n_layers)
        )

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> tuple[VN97MultiTimescaleLayerState, ...]:
        return tuple(
            layer.initial_state(
                batch_size,
                device=device,
                dtype=dtype,
            )
            for layer in self.layers
        )

    def zero_impact(self) -> bool:
        return all(layer.zero_impact() for layer in self.layers)

    def trainable_parameter_count(self) -> int:
        return sum(
            int(parameter.numel())
            for parameter in self.parameters()
            if parameter.requires_grad
        )


@dataclass(frozen=True)
class VN97RecurrentReasoningPolicy:
    fast_passes: int = 1
    normal_passes: int = 2
    deep_passes: int = 4
    max_adaptive_passes: int = 6

    def __post_init__(self) -> None:
        values = (
            self.fast_passes,
            self.normal_passes,
            self.deep_passes,
            self.max_adaptive_passes,
        )
        if any(value <= 0 for value in values):
            raise ValueError("reasoning pass counts must be positive")
        if not (
            self.fast_passes
            <= self.normal_passes
            <= self.deep_passes
            <= self.max_adaptive_passes
        ):
            raise ValueError("reasoning pass counts must be monotonic")

    def passes_for(
        self,
        mode: Literal["fast", "normal", "deep", "adaptive"],
    ) -> int:
        if mode == "fast":
            return self.fast_passes
        if mode == "normal":
            return self.normal_passes
        if mode == "deep":
            return self.deep_passes
        if mode == "adaptive":
            return self.max_adaptive_passes
        raise ValueError(f"unsupported reasoning mode: {mode}")
