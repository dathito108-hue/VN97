"""VN97 adaptive-state candidate: scalar mathematical reference, not a runtime.

Input-conditioned affine updates preserve scan compatibility. No weight import,
training, runtime dispatch or production activation is performed here.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite numeric")
    return float(value)


def _vector(values, width, label):
    if len(values) != width:
        raise ValueError(f"{label} width mismatch")
    return tuple(_number(x, label) for x in values)


@dataclass(frozen=True)
class StateConfig:
    width: int
    fast_rate: float = 1.0
    slow_rate: float = 0.05
    max_dt: float = 4.0
    architecture: str = "VN97-ADAPTIVE-STATE-EXPERIMENT-1"

    def __post_init__(self):
        if type(self.width) is not int or self.width <= 0:
            raise ValueError("width must be positive integer")
        fast = _number(self.fast_rate, "fast_rate")
        slow = _number(self.slow_rate, "slow_rate")
        if not 0 < slow < fast:
            raise ValueError("rates require 0 < slow < fast")
        if _number(self.max_dt, "max_dt") <= 0:
            raise ValueError("max_dt must be positive")
        if self.architecture != "VN97-ADAPTIVE-STATE-EXPERIMENT-1":
            raise ValueError("unsupported architecture")

    def fingerprint(self):
        body = asdict(self)
        for field in ("fast_rate", "slow_rate", "max_dt"):
            body[field] = float(body[field])
        raw = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return hashlib.sha256(b"VN97ASC1\0" + raw).hexdigest()

    def recurrent_bytes(self, bytes_per_value=2):
        if type(bytes_per_value) is not int or bytes_per_value not in (2, 4):
            raise ValueError("state storage must be FP16 or FP32")
        return 2 * self.width * bytes_per_value


@dataclass(frozen=True)
class State:
    fingerprint: str
    fast: tuple
    slow: tuple


@dataclass(frozen=True)
class Frame:
    drive: tuple
    dt: tuple
    slow_write: tuple
    slow_read: tuple


@dataclass(frozen=True)
class Affine:
    scale: tuple
    bias: tuple


def compose(after, before):
    """Chronological composition: after(before(state))."""
    width = len(after.scale)
    if any(len(x) != width for x in (after.bias, before.scale, before.bias)):
        raise ValueError("affine dimensions differ")
    return Affine(
        tuple(a * b for a, b in zip(after.scale, before.scale)),
        tuple(a * b + c for a, b, c in zip(after.scale, before.bias, after.bias)),
    )


def prefix_scan(updates):
    """Tree-depth inclusive scan reference; this Python loop is not a fast kernel."""
    result = list(updates)
    offset = 1
    while offset < len(result):
        previous = result
        result = [
            compose(previous[i], previous[i - offset]) if i >= offset else previous[i]
            for i in range(len(previous))
        ]
        offset *= 2
    return result


class AdaptiveStateCell:
    def __init__(self, config):
        self.config = config
        self.identity = config.fingerprint()

    def initial_state(self):
        zeros = (0.0,) * self.config.width
        return State(self.identity, zeros, zeros)

    def _state(self, state):
        if state.fingerprint != self.identity:
            raise ValueError("state belongs to a different architecture configuration")
        return (_vector(state.fast, self.config.width, "fast state")
                + _vector(state.slow, self.config.width, "slow state"))

    def _update(self, frame):
        width = self.config.width
        drive = _vector(frame.drive, width, "drive")
        dt = _vector(frame.dt, width, "dt")
        write = _vector(frame.slow_write, width, "slow_write")
        read = _vector(frame.slow_read, width, "slow_read")
        if any(not 0 <= x <= self.config.max_dt for x in dt):
            raise ValueError("dt outside budget")
        if any(not 0 <= x <= 1 for x in write + read):
            raise ValueError("write/read gates must be in [0, 1]")
        scale, bias = [], []
        for rate, gate in ((self.config.fast_rate, (1.0,) * width), (self.config.slow_rate, write)):
            for value, duration, opened in zip(drive, dt, gate):
                elapsed = duration * opened
                exponent = -rate * elapsed
                scale.append(math.exp(exponent))
                # Exact ZOH integral for dh/dt = -rate*h + drive.
                gain = -math.expm1(exponent) / rate
                bias.append(gain * value)
        return Affine(tuple(scale), tuple(bias)), read

    def _apply(self, update, initial):
        values = tuple(a * s + b for a, s, b in zip(update.scale, initial, update.bias))
        if not all(math.isfinite(x) for x in values):
            raise ValueError("state overflow")
        width = self.config.width
        return State(self.identity, values[:width], values[width:])

    @staticmethod
    def _read(state, read):
        return tuple((1 - gate) * fast + gate * slow
                     for fast, slow, gate in zip(state.fast, state.slow, read))

    def step(self, frame, state=None):
        initial = self._state(state if state is not None else self.initial_state())
        update, read = self._update(frame)
        next_state = self._apply(update, initial)
        return self._read(next_state, read), next_state

    def sequence(self, frames, state=None, *, execution="scan"):
        if execution not in ("scan", "sequential"):
            raise ValueError("execution must be scan or sequential")
        state = state if state is not None else self.initial_state()
        initial = self._state(state)
        if execution == "sequential":
            outputs = []
            for frame in frames:
                output, state = self.step(frame, state)
                outputs.append(output)
            return outputs, state
        compiled = [self._update(frame) for frame in frames]
        outputs = []
        for update, (_, read) in zip(prefix_scan([x[0] for x in compiled]), compiled):
            state = self._apply(update, initial)
            outputs.append(self._read(state, read))
        return outputs, state
