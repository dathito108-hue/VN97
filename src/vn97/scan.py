from __future__ import annotations

import math

import torch


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
