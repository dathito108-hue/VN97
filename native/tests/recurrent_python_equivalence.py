from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


FLOAT_P = ctypes.POINTER(ctypes.c_float)


def ptr(tensor: torch.Tensor) -> FLOAT_P:
    assert tensor.dtype == torch.float32
    assert tensor.device.type == "cpu"
    assert tensor.is_contiguous()
    return ctypes.cast(tensor.data_ptr(), FLOAT_P)


def scan(decay, drive, initial):
    prefix_a = decay
    prefix_b = drive
    offset = 1
    while offset < decay.shape[1]:
        left_a = prefix_a[:, :-offset]
        left_b = prefix_b[:, :-offset]
        right_a = prefix_a[:, offset:]
        right_b = prefix_b[:, offset:]
        prefix_a = torch.cat(
            (prefix_a[:, :offset], right_a * left_a), dim=1
        )
        prefix_b = torch.cat(
            (prefix_b[:, :offset], right_b + right_a * left_b), dim=1
        )
        offset <<= 1
    return prefix_a * initial.unsqueeze(1) + prefix_b


def sequential(decay, drive, readout, gate, initial):
    state = initial.clone()
    outputs = []
    for t in range(decay.shape[1]):
        state = decay[:, t] * state + drive[:, t]
        y = (state * readout[:, t].unsqueeze(1)).sum(dim=-1)
        outputs.append(y * F.silu(gate[:, t]))
    return torch.stack(outputs, dim=1), state


def configure(lib):
    lib.vn97_recurrent_backend_available.argtypes = [ctypes.c_int]
    lib.vn97_recurrent_backend_available.restype = ctypes.c_int
    lib.vn97_recurrent_prefill_f32.argtypes = [
        FLOAT_P, FLOAT_P, FLOAT_P, FLOAT_P, FLOAT_P, FLOAT_P,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_size_t, ctypes.c_int,
    ]
    lib.vn97_recurrent_prefill_f32.restype = ctypes.c_int
    lib.vn97_recurrent_step_f32.argtypes = [
        FLOAT_P, FLOAT_P, FLOAT_P, FLOAT_P, FLOAT_P, FLOAT_P,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int,
    ]
    lib.vn97_recurrent_step_f32.restype = ctypes.c_int


def main():
    lib = ctypes.CDLL(str(Path(sys.argv[1]).resolve()))
    configure(lib)

    torch.manual_seed(97)
    batch, sequence, d_model, d_state = 2, 7, 5, 9
    decay = torch.sigmoid(
        torch.randn(batch, sequence, d_model, d_state)
    ).contiguous()
    drive = torch.randn(
        batch, sequence, d_model, d_state
    ).contiguous()
    readout = torch.randn(batch, sequence, d_state).contiguous()
    gate = torch.randn(batch, sequence, d_model).contiguous()
    initial = torch.randn(batch, d_model, d_state).contiguous()

    sequential_y, sequential_state = sequential(
        decay, drive, readout, gate, initial
    )
    states = scan(decay, drive, initial)
    scan_y = (
        (states * readout.unsqueeze(2)).sum(dim=-1) * F.silu(gate)
    )
    torch.testing.assert_close(
        scan_y, sequential_y, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        states[:, -1], sequential_state, rtol=1e-5, atol=1e-6
    )

    native_state = initial.clone()
    native_y = torch.empty(batch, sequence, d_model)
    status = lib.vn97_recurrent_prefill_f32(
        ptr(decay), ptr(drive), ptr(readout), ptr(gate),
        ptr(native_state), ptr(native_y),
        batch, sequence, d_model, d_state, 1,
    )
    assert status == 0
    torch.testing.assert_close(
        native_y, scan_y, rtol=2e-5, atol=2e-6
    )
    torch.testing.assert_close(
        native_state, states[:, -1], rtol=2e-5, atol=2e-6
    )

    step_state = initial.clone()
    step_y = []
    for t in range(sequence):
        decay_t = decay[:, t].contiguous()
        drive_t = drive[:, t].contiguous()
        readout_t = readout[:, t].contiguous()
        gate_t = gate[:, t].contiguous()
        out_t = torch.empty(batch, d_model)
        status = lib.vn97_recurrent_step_f32(
            ptr(decay_t), ptr(drive_t), ptr(readout_t), ptr(gate_t),
            ptr(step_state), ptr(out_t),
            batch, d_model, d_state, 1,
        )
        assert status == 0
        step_y.append(out_t)
    torch.testing.assert_close(
        torch.stack(step_y, dim=1), native_y, rtol=2e-5, atol=2e-6
    )
    torch.testing.assert_close(
        step_state, native_state, rtol=2e-5, atol=2e-6
    )

    assert lib.vn97_recurrent_backend_available(1) == 1


if __name__ == "__main__":
    main()
