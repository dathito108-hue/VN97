from __future__ import annotations

import ctypes
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from vn97.scan import affine_prefix_scan


FLOAT_P = ctypes.POINTER(ctypes.c_float)


def ptr(tensor: torch.Tensor) -> FLOAT_P:
    assert tensor.dtype == torch.float32
    assert tensor.device.type == "cpu"
    assert tensor.is_contiguous()
    return ctypes.cast(tensor.data_ptr(), FLOAT_P)


def sequential_reference(
    decay: torch.Tensor,
    drive: torch.Tensor,
    readout: torch.Tensor,
    gate: torch.Tensor,
    initial: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = initial.clone()
    outputs = []
    for t in range(decay.shape[1]):
        state = decay[:, t] * state + drive[:, t]
        y = (state * readout[:, t].unsqueeze(1)).sum(dim=-1)
        outputs.append(y * F.silu(gate[:, t]))
    return torch.stack(outputs, dim=1), state


def configure(lib: ctypes.CDLL) -> None:
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


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: recurrent_python_equivalence.py <shared-library>"
        )

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

    sequential_y, sequential_state = sequential_reference(
        decay, drive, readout, gate, initial
    )
    scan_states, scan_final = affine_prefix_scan(
        decay, drive, initial
    )
    scan_y = (
        (scan_states * readout.unsqueeze(2)).sum(dim=-1)
        * F.silu(gate)
    )

    torch.testing.assert_close(
        scan_y, sequential_y, rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        scan_final, sequential_state, rtol=1e-5, atol=1e-6
    )

    state = initial.clone()
    output = torch.empty(
        batch, sequence, d_model, dtype=torch.float32
    )
    status = lib.vn97_recurrent_prefill_f32(
        ptr(decay), ptr(drive), ptr(readout), ptr(gate),
        ptr(state), ptr(output),
        batch, sequence, d_model, d_state, 1,
    )
    assert status == 0
    torch.testing.assert_close(
        output, scan_y, rtol=2e-5, atol=2e-6
    )
    torch.testing.assert_close(
        state, scan_final, rtol=2e-5, atol=2e-6
    )

    step_state = initial.clone()
    step_outputs = []
    for t in range(sequence):
        decay_t = decay[:, t].contiguous()
        drive_t = drive[:, t].contiguous()
        readout_t = readout[:, t].contiguous()
        gate_t = gate[:, t].contiguous()
        out_t = torch.empty(
            batch, d_model, dtype=torch.float32
        )
        status = lib.vn97_recurrent_step_f32(
            ptr(decay_t), ptr(drive_t), ptr(readout_t), ptr(gate_t),
            ptr(step_state), ptr(out_t),
            batch, d_model, d_state, 1,
        )
        assert status == 0
        step_outputs.append(out_t)

    step_y = torch.stack(step_outputs, dim=1)
    torch.testing.assert_close(
        step_y, output, rtol=2e-5, atol=2e-6
    )
    torch.testing.assert_close(
        step_state, state, rtol=2e-5, atol=2e-6
    )

    auto_state = initial.clone()
    auto_output = torch.empty_like(output)
    status = lib.vn97_recurrent_prefill_f32(
        ptr(decay), ptr(drive), ptr(readout), ptr(gate),
        ptr(auto_state), ptr(auto_output),
        batch, sequence, d_model, d_state, 0,
    )
    assert status == 0
    torch.testing.assert_close(
        auto_output, output, rtol=2e-5, atol=2e-6
    )
    torch.testing.assert_close(
        auto_state, state, rtol=2e-5, atol=2e-6
    )

    assert lib.vn97_recurrent_backend_available(1) == 1
    print(
        "native/PyTorch equivalence PASS:",
        f"B={batch} L={sequence} D={d_model} N={d_state}",
        f"scan_rounds={math.ceil(math.log2(sequence))}",
    )


if __name__ == "__main__":
    main()
