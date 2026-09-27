from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


R2_ARCHITECTURE_ID = "VN97-R2-SSM1"


@dataclass(frozen=True)
class VN97R2Config:
    vocab_size: int
    d_model: int
    d_inner: int
    n_layers: int
    d_state: int
    d_conv: int
    dt_rank: int
    fast_layers: int
    rms_eps: float = 1e-5
    dt_min: float = 1e-4
    dt_max: float = 1.0
    architecture_id: str = R2_ARCHITECTURE_ID

    def __post_init__(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "d_inner": self.d_inner,
            "n_layers": self.n_layers,
            "d_state": self.d_state,
            "d_conv": self.d_conv,
            "dt_rank": self.dt_rank,
            "fast_layers": self.fast_layers,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.vocab_size <= 3:
            raise ValueError("vocab_size must reserve special tokens and payload tokens")
        if self.d_inner < self.d_model:
            raise ValueError("d_inner must be >= d_model")
        if self.fast_layers > self.n_layers:
            raise ValueError("fast_layers must be <= n_layers")
        if not 0.0 < self.dt_min < self.dt_max:
            raise ValueError("expected 0 < dt_min < dt_max")
        if self.rms_eps <= 0.0:
            raise ValueError("rms_eps must be positive")
        if self.architecture_id != R2_ARCHITECTURE_ID:
            raise ValueError("unsupported R2 architecture id")

    def canonical_dict(self) -> dict[str, object]:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.canonical_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(b"VN97R2CFG\0" + payload).hexdigest()

    def recurrent_state_elements(self, batch_size: int = 1) -> int:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        per_layer = (
            self.d_inner * self.d_state
            + self.d_inner * max(self.d_conv - 1, 0)
        )
        return batch_size * self.n_layers * per_layer

    def recurrent_state_bytes(
        self,
        batch_size: int = 1,
        *,
        bytes_per_value: int = 2,
    ) -> int:
        if bytes_per_value <= 0:
            raise ValueError("bytes_per_value must be positive")
        return self.recurrent_state_elements(batch_size) * bytes_per_value

    def estimated_parameter_count(self) -> int:
        embedding = self.vocab_size * self.d_model
        final_norm = self.d_model
        per_layer = (
            self.d_model
            + self.d_model * (2 * self.d_inner)
            + self.d_inner * self.d_conv
            + self.d_inner
            + self.d_inner * (self.dt_rank + 2 * self.d_state)
            + self.dt_rank * self.d_inner
            + self.d_inner
            + self.d_inner * self.d_state
            + self.d_inner
            + self.d_inner * self.d_model
        )
        return embedding + final_norm + self.n_layers * per_layer

    def estimated_weight_bytes(self, bits_per_weight: float) -> int:
        if bits_per_weight <= 0.0:
            raise ValueError("bits_per_weight must be positive")
        return int(self.estimated_parameter_count() * bits_per_weight / 8.0)


def r2_smoke_config(vocab_size: int = 320) -> VN97R2Config:
    return VN97R2Config(
        vocab_size=vocab_size,
        d_model=64,
        d_inner=128,
        n_layers=4,
        d_state=8,
        d_conv=4,
        dt_rank=8,
        fast_layers=2,
    )


def r2_cpu_pilot_config(vocab_size: int = 4096) -> VN97R2Config:
    """~60M class reference pilot; intended for architecture validation, not mobile deployment."""
    return VN97R2Config(
        vocab_size=vocab_size,
        d_model=768,
        d_inner=1536,
        n_layers=16,
        d_state=32,
        d_conv=4,
        dt_rank=48,
        fast_layers=6,
    )


def r2_mobile_1b_config(vocab_size: int) -> VN97R2Config:
    """Production-scale target in the ~1B-1.3B range depending on vocabulary size."""
    return VN97R2Config(
        vocab_size=vocab_size,
        d_model=2048,
        d_inner=4096,
        n_layers=40,
        d_state=64,
        d_conv=4,
        dt_rank=128,
        fast_layers=12,
    )
