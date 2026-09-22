from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VN97Config:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 6
    d_state: int = 16
    ternary_threshold: float = 0.5
    dt_min: float = 1e-4
    dt_max: float = 1.0
    min_decay: float = 0.01
    max_decay: float = 16.0
    rms_eps: float = 1e-5
    embedding_rank: int | None = None

    def __post_init__(self) -> None:
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must be > 1")
        if self.d_model <= 0 or self.n_layers <= 0 or self.d_state <= 0:
            raise ValueError("d_model, n_layers and d_state must be positive")
        if not 0.0 <= self.ternary_threshold <= 1.0:
            raise ValueError("ternary_threshold must be in [0, 1]")
        if not 0.0 < self.dt_min < self.dt_max:
            raise ValueError("expected 0 < dt_min < dt_max")
        if not 0.0 < self.min_decay <= self.max_decay:
            raise ValueError("expected 0 < min_decay <= max_decay")
        if self.embedding_rank is not None and not 0 < self.embedding_rank < self.d_model:
            raise ValueError("embedding_rank must satisfy 0 < rank < d_model")

    def recurrent_state_elements(self, batch_size: int = 1) -> int:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return batch_size * self.n_layers * self.d_model * self.d_state

    def recurrent_state_bytes(self, batch_size: int = 1, bytes_per_value: int = 4) -> int:
        if bytes_per_value <= 0:
            raise ValueError("bytes_per_value must be positive")
        return self.recurrent_state_elements(batch_size) * bytes_per_value

    def tied_embedding_parameters(self) -> int:
        if self.embedding_rank is None:
            return self.vocab_size * self.d_model
        return (
            self.vocab_size * self.embedding_rank
            + self.embedding_rank * self.d_model
        )

    def full_tied_embedding_parameters(self) -> int:
        return self.vocab_size * self.d_model

    def embedding_compression_ratio(self) -> float:
        return self.tied_embedding_parameters() / self.full_tied_embedding_parameters()
