"""Trainable research cell. No production loader, weight transfer or activation."""
from dataclasses import dataclass
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F
from adaptive_state import StateConfig


@dataclass(frozen=True)
class TensorState:
    architecture: str
    fast: torch.Tensor
    slow: torch.Tensor

    def detach(self):
        return TensorState(self.architecture, self.fast.detach(), self.slow.detach())


class TrainableAdaptiveState(nn.Module):
    def __init__(self, input_width, config):
        super().__init__()
        if type(input_width) is not int or input_width <= 0:
            raise ValueError("input width must be positive integer")
        if not isinstance(config, StateConfig):
            raise ValueError("config must be StateConfig")
        if config.slow_rate <= 1e-6:
            raise ValueError("trainable slow rate must exceed numerical floor")
        self.config = config
        self.input_width = input_width
        self.architecture = hashlib.sha256(
            f"VN97TRAINABLEASC1:{input_width}:{config.fingerprint()}".encode()
        ).hexdigest()
        self.projection = nn.Linear(input_width, 4 * config.width)
        # Scalar rates first, matching the reference; channels have learned gates.
        inverse_softplus = lambda x: x + math.log(-math.expm1(-x))
        self.raw_slow = nn.Parameter(torch.tensor(inverse_softplus(config.slow_rate - 1e-6)))
        self.raw_gap = nn.Parameter(torch.tensor(inverse_softplus(config.fast_rate - config.slow_rate)))

    def rates(self):
        slow = F.softplus(self.raw_slow) + 1e-6
        fast = slow + F.softplus(self.raw_gap) + 1e-6
        return fast, slow

    def selectors(self, x):
        drive, duration, write, read = self.projection(x).chunk(4, dim=-1)
        return drive, torch.sigmoid(duration) * self.config.max_dt, torch.sigmoid(write), torch.sigmoid(read)

    def initial_state(self, batch_size):
        p = self.projection.weight
        zeros = p.new_zeros(batch_size, self.config.width)
        return TensorState(self.architecture, zeros, zeros.clone())

    def _validate(self, x, state):
        if x.ndim != 3 or x.shape[-1] != self.input_width or x.shape[0] <= 0:
            raise ValueError("input must be [positive batch, time, input_width]")
        p = self.projection.weight
        if x.dtype != p.dtype or x.device != p.device or x.dtype not in (torch.float32, torch.float64):
            raise ValueError("research input must match model FP32/FP64 dtype and device")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("input must be finite")
        if state.architecture != self.architecture:
            raise ValueError("foreign architecture state")
        for value in (state.fast, state.slow):
            if value.shape != (x.shape[0], self.config.width) or value.dtype != x.dtype or value.device != x.device:
                raise ValueError("state shape/dtype/device mismatch")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("state must be finite")

    def forward(self, x, state=None, *, execution="scan"):
        if execution not in ("scan", "sequential"):
            raise ValueError("unknown execution mode")
        if x.ndim != 3:
            raise ValueError("input must have three axes")
        state = self.initial_state(x.shape[0]) if state is None else state
        self._validate(x, state)
        if x.shape[1] == 0:
            return x.new_empty(x.shape[0], 0, self.config.width), state
        drive, dt, write, read = self.selectors(x)
        fast, slow = self.rates()
        elapsed = torch.cat((dt, dt * write), dim=-1)
        rate = torch.cat((fast.expand(self.config.width), slow.expand(self.config.width)))
        exponent = -elapsed * rate
        a = torch.exp(exponent)
        b = -torch.expm1(exponent) / rate * torch.cat((drive, drive), dim=-1)
        initial = torch.cat((state.fast, state.slow), dim=-1)
        if execution == "scan":
            offset = 1
            while offset < x.shape[1]:
                old_a, old_b = a, b
                a = torch.cat((old_a[:, :offset], old_a[:, offset:] * old_a[:, :-offset]), dim=1)
                b = torch.cat((old_b[:, :offset], old_a[:, offset:] * old_b[:, :-offset] + old_b[:, offset:]), dim=1)
                offset *= 2
            history = a * initial.unsqueeze(1) + b
        else:
            current = initial
            values = []
            for index in range(x.shape[1]):
                current = a[:, index] * current + b[:, index]
                values.append(current)
            history = torch.stack(values, dim=1)
        if not bool(torch.isfinite(history).all()):
            raise ValueError("non-finite recurrent result")
        fast_values, slow_values = history.chunk(2, dim=-1)
        output = (1 - read) * fast_values + read * slow_values
        return output, TensorState(self.architecture, fast_values[:, -1], slow_values[:, -1])

    def step(self, x, state=None):
        if x.ndim != 2:
            raise ValueError("step input must be [batch, input_width]")
        output, state = self.forward(x.unsqueeze(1), state, execution="sequential")
        return output[:, 0], state
