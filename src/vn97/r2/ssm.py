from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..scan import affine_prefix_scan
from .config import VN97R2Config


class R2RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inv_rms = torch.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        ).to(dtype=x.dtype)
        return x * inv_rms * self.weight


@dataclass
class R2LayerState:
    conv: torch.Tensor
    ssm: torch.Tensor

    def detach(self) -> "R2LayerState":
        return R2LayerState(
            conv=self.conv.detach(),
            ssm=self.ssm.detach(),
        )


class SelectiveSSMBlock(nn.Module):
    """VN97-native Mamba-like selective SSM.

    Full-sequence execution uses an associative affine prefix scan. Recurrent
    token execution uses the mathematically identical constant-state update.
    This makes training/prefill parallelizable while preserving a small mobile
    autoregressive state.
    """

    def __init__(self, config: VN97R2Config) -> None:
        super().__init__()
        self.config = config
        self.norm = R2RMSNorm(config.d_model, config.rms_eps)

        self.in_proj = nn.Linear(
            config.d_model,
            2 * config.d_inner,
            bias=False,
        )
        self.conv_weight = nn.Parameter(
            torch.empty(config.d_inner, config.d_conv)
        )
        self.conv_bias = nn.Parameter(torch.zeros(config.d_inner))

        self.x_proj = nn.Linear(
            config.d_inner,
            config.dt_rank + 2 * config.d_state,
            bias=False,
        )
        self.dt_proj = nn.Linear(
            config.dt_rank,
            config.d_inner,
            bias=True,
        )

        rates = torch.logspace(
            math.log10(0.01),
            math.log10(16.0),
            steps=config.d_state,
            dtype=torch.float32,
        )
        self.a_log = nn.Parameter(
            rates.log().repeat(config.d_inner, 1)
        )
        self.d_skip = nn.Parameter(torch.ones(config.d_inner))
        self.out_proj = nn.Linear(
            config.d_inner,
            config.d_model,
            bias=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.xavier_uniform_(self.dt_proj.weight)
        nn.init.zeros_(self.dt_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))
        nn.init.zeros_(self.conv_bias)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> R2LayerState:
        conv_width = max(self.config.d_conv - 1, 0)
        return R2LayerState(
            conv=torch.zeros(
                batch_size,
                self.config.d_inner,
                conv_width,
                device=device,
                dtype=dtype,
            ),
            ssm=torch.zeros(
                batch_size,
                self.config.d_inner,
                self.config.d_state,
                device=device,
                dtype=dtype,
            ),
        )

    def _validate_step_state(
        self,
        x: torch.Tensor,
        state: Optional[R2LayerState],
    ) -> R2LayerState:
        if x.ndim != 2 or x.shape[-1] != self.config.d_model:
            raise ValueError(
                "step expects [batch, d_model], got "
                f"{tuple(x.shape)}"
            )
        return self._validate_shapes(
            batch_size=x.shape[0],
            device=x.device,
            dtype=x.dtype,
            state=state,
        )

    def _validate_sequence_state(
        self,
        x: torch.Tensor,
        state: Optional[R2LayerState],
    ) -> R2LayerState:
        if x.ndim != 3 or x.shape[-1] != self.config.d_model:
            raise ValueError(
                "forward expects [batch, seq, d_model], got "
                f"{tuple(x.shape)}"
            )
        if x.shape[1] <= 0:
            raise ValueError("sequence length must be positive")
        return self._validate_shapes(
            batch_size=x.shape[0],
            device=x.device,
            dtype=x.dtype,
            state=state,
        )

    def _validate_shapes(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        state: Optional[R2LayerState],
    ) -> R2LayerState:
        if state is None:
            return self.initial_state(
                batch_size,
                device=device,
                dtype=dtype,
            )

        expected_conv = (
            batch_size,
            self.config.d_inner,
            max(self.config.d_conv - 1, 0),
        )
        expected_ssm = (
            batch_size,
            self.config.d_inner,
            self.config.d_state,
        )
        if tuple(state.conv.shape) != expected_conv:
            raise ValueError(
                f"invalid conv state {tuple(state.conv.shape)}; "
                f"expected {expected_conv}"
            )
        if tuple(state.ssm.shape) != expected_ssm:
            raise ValueError(
                f"invalid SSM state {tuple(state.ssm.shape)}; "
                f"expected {expected_ssm}"
            )
        return state

    def _causal_conv_step(
        self,
        u: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.d_conv == 1:
            window = u.unsqueeze(-1)
            new_state = conv_state
        else:
            window = torch.cat(
                (conv_state, u.unsqueeze(-1)),
                dim=-1,
            )
            new_state = window[:, :, 1:]

        conv = (
            window * self.conv_weight.unsqueeze(0)
        ).sum(dim=-1) + self.conv_bias
        return F.silu(conv), new_state

    def _causal_conv_sequence(
        self,
        u: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # u: [batch, sequence, d_inner]
        channels = u.transpose(1, 2)
        if self.config.d_conv == 1:
            source = channels
            new_state = conv_state
        else:
            source = torch.cat(
                (conv_state, channels),
                dim=-1,
            )
            new_state = source[:, :, -(self.config.d_conv - 1):]

        convolved = F.conv1d(
            source,
            self.conv_weight.unsqueeze(1),
            bias=self.conv_bias,
            stride=1,
            padding=0,
            groups=self.config.d_inner,
        )
        if convolved.shape[-1] != u.shape[1]:
            raise RuntimeError(
                "R2 causal convolution produced wrong sequence length"
            )
        return F.silu(convolved.transpose(1, 2)), new_state

    def _dynamics(
        self,
        u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dynamics = self.x_proj(u)
        dt_low_rank, b, c = torch.split(
            dynamics,
            [
                self.config.dt_rank,
                self.config.d_state,
                self.config.d_state,
            ],
            dim=-1,
        )
        dt = F.softplus(self.dt_proj(dt_low_rank)).clamp(
            min=self.config.dt_min,
            max=self.config.dt_max,
        )
        return dt, b, c

    def _continuous_a(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return -torch.exp(self.a_log).to(
            device=device,
            dtype=dtype,
        )

    def step(
        self,
        x: torch.Tensor,
        state: Optional[R2LayerState] = None,
    ) -> tuple[torch.Tensor, R2LayerState]:
        residual = x
        x = self.norm(x)
        state = self._validate_step_state(x, state)

        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u, next_conv = self._causal_conv_step(u, state.conv)
        dt, b, c = self._dynamics(u)

        a = self._continuous_a(
            device=u.device,
            dtype=u.dtype,
        )
        z = dt.unsqueeze(-1) * a.unsqueeze(0)
        d_a = torch.exp(z)
        zoh = torch.expm1(z) / a.unsqueeze(0)
        drive = (
            zoh
            * b.unsqueeze(1)
            * u.unsqueeze(-1)
        )
        next_ssm = d_a * state.ssm + drive

        y = (
            next_ssm * c.unsqueeze(1)
        ).sum(dim=-1)
        y = y + self.d_skip * u
        y = y * F.silu(gate)
        y = self.out_proj(y)
        return residual + y, R2LayerState(
            conv=next_conv,
            ssm=next_ssm,
        )

    def forward_reference(
        self,
        x: torch.Tensor,
        state: Optional[R2LayerState] = None,
    ) -> tuple[torch.Tensor, R2LayerState]:
        """Token-by-token numerical reference used for equivalence tests."""
        self._validate_sequence_state(x, state)
        outputs: list[torch.Tensor] = []
        current = state
        for index in range(x.shape[1]):
            y, current = self.step(x[:, index], current)
            outputs.append(y)
        if current is None:
            raise AssertionError("reference state was not produced")
        return torch.stack(outputs, dim=1), current

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[R2LayerState] = None,
    ) -> tuple[torch.Tensor, R2LayerState]:
        residual = x
        normalized = self.norm(x)
        state = self._validate_sequence_state(normalized, state)

        u, gate = self.in_proj(normalized).chunk(2, dim=-1)
        u, next_conv = self._causal_conv_sequence(
            u,
            state.conv,
        )
        dt, b, c = self._dynamics(u)

        a = self._continuous_a(
            device=u.device,
            dtype=u.dtype,
        )
        z = (
            dt.unsqueeze(-1)
            * a.unsqueeze(0).unsqueeze(0)
        )
        d_a = torch.exp(z)
        zoh = (
            torch.expm1(z)
            / a.unsqueeze(0).unsqueeze(0)
        )
        drive = (
            zoh
            * b.unsqueeze(2)
            * u.unsqueeze(-1)
        )

        states, next_ssm = affine_prefix_scan(
            d_a,
            drive,
            state.ssm,
        )
        y = (
            states * c.unsqueeze(2)
        ).sum(dim=-1)
        y = y + self.d_skip * u
        y = y * F.silu(gate)
        y = self.out_proj(y)

        return residual + y, R2LayerState(
            conv=next_conv,
            ssm=next_ssm,
        )
