from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Mamba2ReferenceConfig:
    d_model: int
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    head_dim: int = 64
    n_groups: int = 1
    rms_eps: float = 1e-5
    norm_before_gate: bool = False

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.d_state <= 0 or self.d_conv <= 0:
            raise ValueError("Mamba-2 dimensions must be positive")
        if self.expand <= 0 or self.head_dim <= 0 or self.n_groups <= 0:
            raise ValueError("Mamba-2 expansion/head/group values must be positive")
        if self.d_inner % self.head_dim:
            raise ValueError("expanded width must be divisible by head_dim")
        if self.d_inner % self.n_groups:
            raise ValueError("expanded width must be divisible by n_groups")
        if self.rms_eps <= 0.0:
            raise ValueError("rms_eps must be positive")

    @property
    def d_inner(self) -> int:
        return self.d_model * self.expand

    @property
    def n_heads(self) -> int:
        return self.d_inner // self.head_dim

    @property
    def conv_dim(self) -> int:
        return self.d_inner + 2 * self.n_groups * self.d_state

    @property
    def in_proj_dim(self) -> int:
        return (
            2 * self.d_inner
            + 2 * self.n_groups * self.d_state
            + self.n_heads
        )


@dataclass
class Mamba2LayerReferenceState:
    conv: torch.Tensor
    ssm: torch.Tensor

    def detach(self) -> "Mamba2LayerReferenceState":
        return Mamba2LayerReferenceState(
            conv=self.conv.detach(),
            ssm=self.ssm.detach(),
        )


def initial_layer_state(
    config: Mamba2ReferenceConfig,
    batch_size: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Mamba2LayerReferenceState:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return Mamba2LayerReferenceState(
        conv=torch.zeros(
            batch_size,
            config.conv_dim,
            config.d_conv,
            device=device,
            dtype=dtype,
        ),
        ssm=torch.zeros(
            batch_size,
            config.n_heads,
            config.head_dim,
            config.d_state,
            device=device,
            dtype=dtype,
        ),
    )


def rms_norm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    dtype = x.dtype
    source = x.float()
    scale = torch.rsqrt(
        source.square().mean(dim=-1, keepdim=True) + eps
    )
    return (
        source
        * scale
        * weight.float()
    ).to(dtype=dtype)


def rms_norm_gated_ref(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
    n_groups: int,
    norm_before_gate: bool,
) -> torch.Tensor:
    if x.shape != z.shape:
        raise ValueError("gated RMSNorm x/z shapes differ")
    if x.shape[-1] % n_groups:
        raise ValueError("gated RMSNorm width is not divisible by n_groups")
    dtype = x.dtype
    source = x.float()
    gate = z.float()
    if not norm_before_gate:
        source = source * F.silu(gate)
    group_size = source.shape[-1] // n_groups
    grouped = source.reshape(*source.shape[:-1], n_groups, group_size)
    inv = torch.rsqrt(
        grouped.square().mean(dim=-1, keepdim=True) + eps
    )
    out = (grouped * inv).reshape_as(source) * weight.float()
    if norm_before_gate:
        out = out * F.silu(gate)
    return out.to(dtype=dtype)


def _required(
    tensors: Mapping[str, torch.Tensor],
    name: str,
) -> torch.Tensor:
    tensor = tensors.get(name)
    if tensor is None:
        raise ValueError(f"missing Mamba-2 tensor: {name}")
    return tensor


def mamba2_mixer_step_ref(
    hidden: torch.Tensor,
    state: Mamba2LayerReferenceState,
    tensors: Mapping[str, torch.Tensor],
    config: Mamba2ReferenceConfig,
) -> tuple[torch.Tensor, Mamba2LayerReferenceState]:
    """CPU-auditable one-token reference matching official Mamba2.step.

    This intentionally mirrors the unfused path: in-projection, causal
    depthwise convolution state update, SSD recurrent update, D skip, gated
    group RMSNorm and output projection.
    """

    if hidden.ndim != 2 or hidden.shape[-1] != config.d_model:
        raise ValueError(
            "hidden must be [batch, d_model], got "
            f"{tuple(hidden.shape)}"
        )
    batch = hidden.shape[0]
    if state.conv.shape != (
        batch,
        config.conv_dim,
        config.d_conv,
    ):
        raise ValueError("Mamba-2 convolution state shape mismatch")
    if state.ssm.shape != (
        batch,
        config.n_heads,
        config.head_dim,
        config.d_state,
    ):
        raise ValueError("Mamba-2 SSD state shape mismatch")

    in_proj = _required(tensors, "in_proj.weight")
    if in_proj.shape != (config.in_proj_dim, config.d_model):
        raise ValueError("Mamba-2 in_proj shape mismatch")
    zxbcdt = F.linear(hidden, in_proj)

    d_mlp = (
        zxbcdt.shape[-1]
        - 2 * config.d_inner
        - 2 * config.n_groups * config.d_state
        - config.n_heads
    ) // 2
    if d_mlp != 0:
        raise ValueError(
            "VN97 G0.2 reference currently requires d_intermediate=0"
        )

    z, xbc, dt = torch.split(
        zxbcdt,
        [
            config.d_inner,
            config.d_inner + 2 * config.n_groups * config.d_state,
            config.n_heads,
        ],
        dim=-1,
    )

    conv_weight = _required(tensors, "conv1d.weight")
    if conv_weight.shape != (
        config.conv_dim,
        1,
        config.d_conv,
    ):
        raise ValueError("Mamba-2 conv1d weight shape mismatch")
    conv_bias = _required(tensors, "conv1d.bias")
    if conv_bias.shape != (config.conv_dim,):
        raise ValueError("Mamba-2 conv1d bias shape mismatch")

    next_conv = torch.roll(state.conv, shifts=-1, dims=-1)
    next_conv = next_conv.clone()
    next_conv[:, :, -1] = xbc
    xbc = torch.sum(
        next_conv * conv_weight[:, 0, :].to(dtype=next_conv.dtype),
        dim=-1,
    )
    xbc = F.silu(
        xbc + conv_bias.to(device=xbc.device, dtype=xbc.dtype)
    )

    x, b_value, c_value = torch.split(
        xbc,
        [
            config.d_inner,
            config.n_groups * config.d_state,
            config.n_groups * config.d_state,
        ],
        dim=-1,
    )
    if config.n_groups != 1:
        raise ValueError(
            "auditable recurrent G0.2 reference currently requires n_groups=1"
        )

    a_log = _required(tensors, "A_log")
    dt_bias = _required(tensors, "dt_bias")
    d_skip = _required(tensors, "D")
    if (
        a_log.shape != (config.n_heads,)
        or dt_bias.shape != (config.n_heads,)
        or d_skip.shape != (config.n_heads,)
    ):
        raise ValueError("Mamba-2 A/dt/D head shape mismatch")

    a = -torch.exp(a_log.float())
    dt_value = F.softplus(
        dt + dt_bias.to(device=dt.device, dtype=dt.dtype)
    )
    d_a = torch.exp(dt_value.float() * a[None, :])

    x_heads = x.reshape(
        batch,
        config.n_heads,
        config.head_dim,
    )
    b_group = b_value.reshape(batch, config.d_state)
    c_group = c_value.reshape(batch, config.d_state)
    d_b_x = torch.einsum(
        "bh,bn,bhp->bhpn",
        dt_value.float(),
        b_group.float(),
        x_heads.float(),
    )
    next_ssm = (
        state.ssm.float()
        * d_a[:, :, None, None]
        + d_b_x
    ).to(dtype=state.ssm.dtype)

    y = torch.einsum(
        "bhpn,bn->bhp",
        next_ssm.to(dtype=x_heads.dtype),
        c_group.to(dtype=x_heads.dtype),
    )
    y = y + d_skip.to(
        device=y.device,
        dtype=y.dtype,
    )[None, :, None] * x_heads
    y = y.reshape(batch, config.d_inner)

    norm_weight = _required(tensors, "norm.weight")
    if norm_weight.shape != (config.d_inner,):
        raise ValueError("Mamba-2 gated RMSNorm weight shape mismatch")
    y = rms_norm_gated_ref(
        y,
        z,
        norm_weight,
        eps=config.rms_eps,
        n_groups=config.n_groups,
        norm_before_gate=config.norm_before_gate,
    )

    out_proj = _required(tensors, "out_proj.weight")
    if out_proj.shape != (config.d_model, config.d_inner):
        raise ValueError("Mamba-2 out_proj shape mismatch")
    out = F.linear(y, out_proj)
    return out, Mamba2LayerReferenceState(
        conv=next_conv,
        ssm=next_ssm,
    )


def mamba2_block_step_ref(
    hidden: torch.Tensor,
    residual: torch.Tensor | None,
    state: Mamba2LayerReferenceState,
    tensors: Mapping[str, torch.Tensor],
    config: Mamba2ReferenceConfig,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    Mamba2LayerReferenceState,
]:
    block_norm = _required(tensors, "block_norm.weight")
    if block_norm.shape != (config.d_model,):
        raise ValueError("Mamba-2 block RMSNorm weight shape mismatch")

    residual_next = hidden if residual is None else hidden + residual
    normalized = rms_norm_ref(
        residual_next.to(dtype=block_norm.dtype),
        block_norm,
        eps=config.rms_eps,
    )
    # Official Mamba2 2.7B uses residual_in_fp32=true.
    residual_next = residual_next.float()
    mixer_tensors = {
        name: value
        for name, value in tensors.items()
        if name != "block_norm.weight"
    }
    out, next_state = mamba2_mixer_step_ref(
        normalized,
        state,
        mixer_tensors,
        config,
    )
    return out, residual_next, next_state


def mamba2_model_step_ref(
    token_ids: torch.Tensor,
    states: Sequence[Mamba2LayerReferenceState],
    state_dict: Mapping[str, torch.Tensor],
    *,
    config: Mamba2ReferenceConfig,
    n_layers: int,
) -> tuple[torch.Tensor, tuple[Mamba2LayerReferenceState, ...]]:
    if token_ids.ndim != 1:
        raise ValueError("token_ids must be [batch]")
    if len(states) != n_layers:
        raise ValueError("state layer count mismatch")

    embedding = _required(state_dict, "backbone.embedding.weight")
    hidden = F.embedding(token_ids, embedding)
    residual: torch.Tensor | None = None
    next_states: list[Mamba2LayerReferenceState] = []

    for index in range(n_layers):
        prefix = f"backbone.layers.{index}"
        mixer = f"{prefix}.mixer"
        local = {
            "block_norm.weight": _required(
                state_dict,
                f"{prefix}.norm.weight",
            ),
            "in_proj.weight": _required(
                state_dict,
                f"{mixer}.in_proj.weight",
            ),
            "conv1d.weight": _required(
                state_dict,
                f"{mixer}.conv1d.weight",
            ),
            "conv1d.bias": _required(
                state_dict,
                f"{mixer}.conv1d.bias",
            ),
            "dt_bias": _required(
                state_dict,
                f"{mixer}.dt_bias",
            ),
            "A_log": _required(
                state_dict,
                f"{mixer}.A_log",
            ),
            "D": _required(
                state_dict,
                f"{mixer}.D",
            ),
            "norm.weight": _required(
                state_dict,
                f"{mixer}.norm.weight",
            ),
            "out_proj.weight": _required(
                state_dict,
                f"{mixer}.out_proj.weight",
            ),
        }
        hidden, residual, next_state = mamba2_block_step_ref(
            hidden,
            residual,
            states[index],
            local,
            config,
        )
        next_states.append(next_state)

    residual = hidden + residual if residual is not None else hidden
    final_norm = _required(state_dict, "backbone.norm_f.weight")
    hidden = rms_norm_ref(
        residual.to(dtype=final_norm.dtype),
        final_norm,
        eps=config.rms_eps,
    )
    lm_head = state_dict.get(
        "lm_head.weight",
        embedding,
    )
    logits = F.linear(hidden, lm_head)
    return logits, tuple(next_states)
