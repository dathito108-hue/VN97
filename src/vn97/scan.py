from __future__ import annotations

import math

import torch
from torch.utils.checkpoint import checkpoint


def affine_scan_rounds(sequence_length: int) -> int:
    if sequence_length < 0:
        raise ValueError("sequence_length must be non-negative")
    if sequence_length <= 1:
        return 0
    return math.ceil(math.log2(sequence_length))


def affine_prefix_scan(
    decay: torch.Tensor,
    drive: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inclusive prefix scan for h_t = decay_t * h_(t-1) + drive_t.

    decay and drive have identical shape [batch, sequence, ...]. The scan composes
    affine state transitions in logarithmic Python control-flow depth instead of
    iterating once per token.
    """
    if decay.shape != drive.shape:
        raise ValueError("decay and drive must have identical shapes")
    if decay.ndim < 2:
        raise ValueError(
            "decay and drive must include batch and sequence dimensions"
        )
    if decay.shape[1] == 0:
        raise ValueError("sequence length must be positive")

    prefix_a = decay
    prefix_b = drive
    offset = 1
    seq_len = decay.shape[1]

    while offset < seq_len:
        left_a = prefix_a[:, :-offset]
        left_b = prefix_b[:, :-offset]
        right_a = prefix_a[:, offset:]
        right_b = prefix_b[:, offset:]

        composed_a = right_a * left_a
        composed_b = right_b + right_a * left_b

        prefix_a = torch.cat(
            (prefix_a[:, :offset], composed_a),
            dim=1,
        )
        prefix_b = torch.cat(
            (prefix_b[:, :offset], composed_b),
            dim=1,
        )
        offset <<= 1

    if initial_state is None:
        states = prefix_b
    else:
        expected = decay.shape[:1] + decay.shape[2:]
        if tuple(initial_state.shape) != tuple(expected):
            raise ValueError(
                "expected initial_state shape "
                f"{tuple(expected)}, got {tuple(initial_state.shape)}"
            )
        states = (
            prefix_a * initial_state.unsqueeze(1)
            + prefix_b
        )

    return states, states[:, -1]



def affine_prefix_scan_memory_efficient(
    decay: torch.Tensor,
    drive: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    chunk_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact affine scan with bounded per-chunk autograd retention.

    The sequence is split into small chunks. Each chunk still uses the
    associative logarithmic scan, while activation checkpointing recomputes
    the chunk scan during backward instead of retaining every scan round for
    the complete sequence. The carry state links chunks exactly, so this is
    numerically the same recurrence, not truncated BPTT.
    """
    if decay.shape != drive.shape:
        raise ValueError("decay and drive must have identical shapes")
    if decay.ndim < 2:
        raise ValueError(
            "decay and drive must include batch and sequence dimensions"
        )
    if decay.shape[1] == 0:
        raise ValueError("sequence length must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    expected = decay.shape[:1] + decay.shape[2:]
    if initial_state is None:
        current = torch.zeros(
            expected,
            device=decay.device,
            dtype=decay.dtype,
        )
    else:
        if tuple(initial_state.shape) != tuple(expected):
            raise ValueError(
                "expected initial_state shape "
                f"{tuple(expected)}, got {tuple(initial_state.shape)}"
            )
        current = initial_state

    outputs: list[torch.Tensor] = []

    def _chunk_scan(
        chunk_decay: torch.Tensor,
        chunk_drive: torch.Tensor,
        chunk_initial: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return affine_prefix_scan(
            chunk_decay,
            chunk_drive,
            chunk_initial,
        )

    for start in range(0, decay.shape[1], chunk_size):
        end = min(start + chunk_size, decay.shape[1])
        chunk_decay = decay[:, start:end]
        chunk_drive = drive[:, start:end]

        needs_grad = (
            torch.is_grad_enabled()
            and (
                chunk_decay.requires_grad
                or chunk_drive.requires_grad
                or current.requires_grad
            )
        )
        if needs_grad:
            chunk_states, current = checkpoint(
                _chunk_scan,
                chunk_decay,
                chunk_drive,
                current,
                use_reentrant=False,
            )
        else:
            chunk_states, current = _chunk_scan(
                chunk_decay,
                chunk_drive,
                current,
            )
        outputs.append(chunk_states)

    return torch.cat(outputs, dim=1), current
