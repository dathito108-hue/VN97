from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantization import TernaryLinear


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inv_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * inv_rms


class SelectiveSSM(nn.Module):
    """Stable diagonal selective state-space layer with constant recurrent state."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        *,
        ternary_threshold: float,
        dt_min: float,
        dt_max: float,
        min_decay: float,
        max_decay: float,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_min = dt_min
        self.dt_max = dt_max

        def linear(i: int, o: int, bias: bool = False) -> TernaryLinear:
            return TernaryLinear(
                i, o, bias=bias, threshold=ternary_threshold
            )

        self.in_proj = linear(d_model, d_model * 2)
        self.dt_proj = linear(d_model, d_model, bias=True)
        self.b_proj = linear(d_model, d_state)
        self.c_proj = linear(d_model, d_state)
        self.out_proj = linear(d_model, d_model)

        rates = torch.logspace(
            math.log10(min_decay),
            math.log10(max_decay),
            steps=d_state,
            dtype=torch.float32,
        )
        self.a_log = nn.Parameter(rates.log().repeat(d_model, 1))
        self.act = nn.SiLU()

    def initial_state(
        self, batch_size: int, *, device, dtype
    ) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.d_model,
            self.d_state,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"expected x [batch, seq, {self.d_model}], got {tuple(x.shape)}"
            )

        batch_size, seq_len, _ = x.shape
        signal, gate = self.in_proj(x).chunk(2, dim=-1)
        dt = F.softplus(self.dt_proj(signal))
        dt = dt.clamp(min=self.dt_min, max=self.dt_max)

        a = -torch.exp(self.a_log).to(dtype=x.dtype, device=x.device)
        if state is None:
            state = self.initial_state(
                batch_size, device=x.device, dtype=x.dtype
            )
        expected = (batch_size, self.d_model, self.d_state)
        if tuple(state.shape) != expected:
            raise ValueError(
                f"expected state {expected}, got {tuple(state.shape)}"
            )

        outputs: list[torch.Tensor] = []
        for t in range(seq_len):
            xt = signal[:, t, :]
            dt_t = dt[:, t, :]
            b_t = self.b_proj(xt)
            c_t = self.c_proj(xt)

            z = a.unsqueeze(0) * dt_t.unsqueeze(-1)
            d_a = torch.exp(z)
            zoh = torch.expm1(z) / a.unsqueeze(0)
            drive = zoh * b_t.unsqueeze(1) * xt.unsqueeze(-1)
            state = d_a * state + drive
            outputs.append(
                (state * c_t.unsqueeze(1)).sum(dim=-1)
            )

        y = torch.stack(outputs, dim=1)
        y = y * self.act(gate)
        return self.out_proj(y), state


class VN97Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        *,
        ternary_threshold: float,
        dt_min: float,
        dt_max: float,
        min_decay: float,
        max_decay: float,
        rms_eps: float,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model, eps=rms_eps)
        self.core = SelectiveSSM(
            d_model,
            d_state,
            ternary_threshold=ternary_threshold,
            dt_min=dt_min,
            dt_max=dt_max,
            min_decay=min_decay,
            max_decay=max_decay,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y, state = self.core(self.norm(x), state)
        return x + y, state
