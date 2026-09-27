from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VN97R2Config
from .ssm import R2LayerState, R2RMSNorm, SelectiveSSMBlock


@dataclass
class VN97R2State:
    layers: tuple[R2LayerState, ...]
    active_layers: int

    def detach(self) -> "VN97R2State":
        return VN97R2State(
            layers=tuple(layer.detach() for layer in self.layers),
            active_layers=self.active_layers,
        )


class VN97R2Model(nn.Module):
    """Single-backbone VN97-R2 language model.

    The same weights support a fast early-exit path and a full-depth path.
    There is no second model/backend. Runtime callers choose how many leading
    blocks are active for a whole recurrent stream.
    """

    def __init__(self, config: VN97R2Config) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )
        self.layers = nn.ModuleList(
            SelectiveSSMBlock(config)
            for _ in range(config.n_layers)
        )
        self.final_norm = R2RMSNorm(
            config.d_model,
            config.rms_eps,
        )
        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=config.d_model ** -0.5,
        )

    def parameter_count(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(
            int(p.numel())
            for p in self.parameters()
            if p.requires_grad
        )

    def resolve_active_layers(
        self,
        profile: str | int | None,
    ) -> int:
        if profile is None or profile == "deep":
            return self.config.n_layers
        if profile == "fast":
            return self.config.fast_layers
        if isinstance(profile, int):
            if not 1 <= profile <= self.config.n_layers:
                raise ValueError(
                    "active layer count must be in "
                    f"[1, {self.config.n_layers}]"
                )
            return profile
        raise ValueError(
            "profile must be 'fast', 'deep', an integer, or None"
        )

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        profile: str | int | None = None,
    ) -> VN97R2State:
        active_layers = self.resolve_active_layers(profile)
        return VN97R2State(
            layers=tuple(
                self.layers[index].initial_state(
                    batch_size,
                    device=device,
                    dtype=dtype,
                )
                for index in range(active_layers)
            ),
            active_layers=active_layers,
        )

    def _validate_state(
        self,
        state: Optional[VN97R2State],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        active_layers: int,
    ) -> VN97R2State:
        if state is None:
            return self.initial_state(
                batch_size,
                device=device,
                dtype=dtype,
                profile=active_layers,
            )
        if state.active_layers != active_layers:
            raise ValueError(
                "cannot change fast/deep layer count inside one recurrent "
                "stream; start a new state when switching profile"
            )
        if len(state.layers) != active_layers:
            raise ValueError("state layer count does not match active_layers")
        return state

    def _project_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.final_norm(hidden)
        return F.linear(hidden, self.embedding.weight)

    def step_hidden(
        self,
        hidden: torch.Tensor,
        state: Optional[VN97R2State] = None,
        *,
        profile: str | int | None = None,
    ) -> tuple[torch.Tensor, VN97R2State]:
        if hidden.ndim != 2 or hidden.shape[-1] != self.config.d_model:
            raise ValueError(
                "step_hidden expects [batch, d_model], got "
                f"{tuple(hidden.shape)}"
            )
        active_layers = self.resolve_active_layers(profile)
        current = self._validate_state(
            state,
            batch_size=hidden.shape[0],
            device=hidden.device,
            dtype=hidden.dtype,
            active_layers=active_layers,
        )

        next_states: list[R2LayerState] = []
        x = hidden
        for index in range(active_layers):
            x, next_layer_state = self.layers[index].step(
                x,
                current.layers[index],
            )
            next_states.append(next_layer_state)

        return x, VN97R2State(
            layers=tuple(next_states),
            active_layers=active_layers,
        )

    def step(
        self,
        token_ids: torch.Tensor,
        state: Optional[VN97R2State] = None,
        *,
        profile: str | int | None = None,
    ) -> tuple[torch.Tensor, VN97R2State]:
        if token_ids.ndim == 2 and token_ids.shape[1] == 1:
            token_ids = token_ids[:, 0]
        if token_ids.ndim != 1:
            raise ValueError(
                "step expects token ids [batch] or [batch, 1], got "
                f"{tuple(token_ids.shape)}"
            )
        hidden = self.embedding(token_ids)
        hidden, next_state = self.step_hidden(
            hidden,
            state,
            profile=profile,
        )
        return self._project_logits(hidden), next_state

    def forward_hidden(
        self,
        input_ids: torch.Tensor,
        state: Optional[VN97R2State] = None,
        *,
        profile: str | int | None = None,
    ) -> tuple[torch.Tensor, VN97R2State]:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must be [batch, seq], got "
                f"{tuple(input_ids.shape)}"
            )
        if input_ids.shape[1] <= 0:
            raise ValueError("sequence length must be positive")

        active_layers = self.resolve_active_layers(profile)
        hidden = self.embedding(input_ids)
        current = self._validate_state(
            state,
            batch_size=input_ids.shape[0],
            device=hidden.device,
            dtype=hidden.dtype,
            active_layers=active_layers,
        )

        outputs: list[torch.Tensor] = []
        for position in range(input_ids.shape[1]):
            token_hidden, current = self.step_hidden(
                hidden[:, position],
                current,
                profile=active_layers,
            )
            outputs.append(token_hidden)
        return torch.stack(outputs, dim=1), current

    def forward(
        self,
        input_ids: torch.Tensor,
        state: Optional[VN97R2State] = None,
        *,
        profile: str | int | None = None,
    ) -> tuple[torch.Tensor, VN97R2State]:
        hidden, state = self.forward_hidden(
            input_ids,
            state,
            profile=profile,
        )
        return self._project_logits(hidden), state

    @torch.inference_mode()
    def generate_greedy(
        self,
        prompt_ids: Iterable[int],
        *,
        max_new_tokens: int,
        eos_id: int | None = None,
        profile: str | int | None = None,
        device: torch.device | str | None = None,
    ) -> list[int]:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        prompt = list(int(token) for token in prompt_ids)
        if not prompt:
            raise ValueError("prompt must contain at least one token")
        if any(token < 0 or token >= self.config.vocab_size for token in prompt):
            raise ValueError("prompt token out of vocabulary range")

        resolved_device = (
            next(self.parameters()).device
            if device is None
            else torch.device(device)
        )
        input_ids = torch.tensor(
            [prompt],
            dtype=torch.long,
            device=resolved_device,
        )
        logits, state = self.forward(
            input_ids,
            profile=profile,
        )
        generated: list[int] = []
        next_logits = logits[:, -1]

        for _ in range(max_new_tokens):
            token = int(next_logits.argmax(dim=-1).item())
            generated.append(token)
            if eos_id is not None and token == eos_id:
                break
            next_token = torch.tensor(
                [token],
                dtype=torch.long,
                device=resolved_device,
            )
            next_logits, state = self.step(
                next_token,
                state,
                profile=profile,
            )
        return generated
