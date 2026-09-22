from __future__ import annotations

import ctypes
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


def configure(lib: ctypes.CDLL) -> None:
    lib.vn97_selective_prefill_f32.argtypes = [
        FLOAT_P,
        FLOAT_P,
        FLOAT_P,
        FLOAT_P,
        FLOAT_P,
        FLOAT_P,
        FLOAT_P,
        FLOAT_P,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
    ]
    lib.vn97_selective_prefill_f32.restype = ctypes.c_int
    lib.vn97_selective_step_f32.argtypes = [
        FLOAT_P,
        FLOAT_P,
        FLOAT_P,
        FLOAT_P,
        FLOAT_P,
        FLOAT_P,
        FLOAT_P,
        FLOAT_P,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
    ]
    lib.vn97_selective_step_f32.restype = ctypes.c_int
    lib.vn97_recurrent_prefill_f32.argtypes = [
        FLOAT_P,
        FLOAT_P,
        FLOAT_P,
        FLOAT_P,
        FLOAT_P,
        FLOAT_P,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.vn97_recurrent_prefill_f32.restype = ctypes.c_int


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: selective_python_equivalence.py <shared-library>"
        )

    lib = ctypes.CDLL(str(Path(sys.argv[1]).resolve()))
    configure(lib)

    torch.manual_seed(197)
    batch, sequence, d_model, d_state = 2, 11, 7, 9
    dt_min, dt_max = 1e-4, 1.0

    signal = torch.randn(batch, sequence, d_model).contiguous()
    dt_logits = torch.randn(batch, sequence, d_model).contiguous()
    input_b = torch.randn(batch, sequence, d_state).contiguous()
    readout_c = torch.randn(batch, sequence, d_state).contiguous()
    gate = torch.randn(batch, sequence, d_model).contiguous()
    a_log = torch.linspace(
        -4.0, 2.0, d_model * d_state, dtype=torch.float32
    ).reshape(d_model, d_state)
    a = (-torch.exp(a_log)).contiguous()
    initial = torch.randn(batch, d_model, d_state).contiguous()

    dt = F.softplus(dt_logits).clamp(min=dt_min, max=dt_max)
    z = a.unsqueeze(0).unsqueeze(0) * dt.unsqueeze(-1)
    decay = torch.exp(z).contiguous()
    zoh = torch.expm1(z) / a.unsqueeze(0).unsqueeze(0)
    drive = (
        zoh * input_b.unsqueeze(2) * signal.unsqueeze(-1)
    ).contiguous()

    states, scan_final = affine_prefix_scan(decay, drive, initial)
    expected = (
        (states * readout_c.unsqueeze(2)).sum(dim=-1)
        * F.silu(gate)
    ).contiguous()

    selective_state = initial.clone()
    selective_output = torch.empty(batch, sequence, d_model)
    status = lib.vn97_selective_prefill_f32(
        ptr(signal),
        ptr(dt_logits),
        ptr(input_b),
        ptr(readout_c),
        ptr(gate),
        ptr(a),
        ptr(selective_state),
        ptr(selective_output),
        batch,
        sequence,
        d_model,
        d_state,
        dt_min,
        dt_max,
        1,
    )
    assert status == 0
    torch.testing.assert_close(
        selective_output, expected, rtol=3e-5, atol=3e-6
    )
    torch.testing.assert_close(
        selective_state, scan_final, rtol=3e-5, atol=3e-6
    )

    recurrent_state = initial.clone()
    recurrent_output = torch.empty_like(selective_output)
    status = lib.vn97_recurrent_prefill_f32(
        ptr(decay),
        ptr(drive),
        ptr(readout_c),
        ptr(gate),
        ptr(recurrent_state),
        ptr(recurrent_output),
        batch,
        sequence,
        d_model,
        d_state,
        1,
    )
    assert status == 0
    torch.testing.assert_close(
        selective_output, recurrent_output, rtol=3e-5, atol=3e-6
    )
    torch.testing.assert_close(
        selective_state, recurrent_state, rtol=3e-5, atol=3e-6
    )

    step_state = initial.clone()
    step_outputs = []
    for t in range(sequence):
        signal_t = signal[:, t].contiguous()
        dt_t = dt_logits[:, t].contiguous()
        b_t = input_b[:, t].contiguous()
        c_t = readout_c[:, t].contiguous()
        gate_t = gate[:, t].contiguous()
        output_t = torch.empty(batch, d_model)
        status = lib.vn97_selective_step_f32(
            ptr(signal_t),
            ptr(dt_t),
            ptr(b_t),
            ptr(c_t),
            ptr(gate_t),
            ptr(a),
            ptr(step_state),
            ptr(output_t),
            batch,
            d_model,
            d_state,
            dt_min,
            dt_max,
            1,
        )
        assert status == 0
        step_outputs.append(output_t)

    step_output = torch.stack(step_outputs, dim=1)
    torch.testing.assert_close(
        step_output, selective_output, rtol=3e-5, atol=3e-6
    )
    torch.testing.assert_close(
        step_state, selective_state, rtol=3e-5, atol=3e-6
    )

    compact_values = (
        signal.numel()
        + dt_logits.numel()
        + input_b.numel()
        + readout_c.numel()
        + gate.numel()
        + a.numel()
    )
    expanded_values = decay.numel() + drive.numel()
    assert compact_values < expanded_values
    print(
        "M2C native selective equivalence PASS:",
        f"compact_inputs={compact_values}",
        f"avoided_expanded_values={expanded_values}",
    )


if __name__ == "__main__":
    main()
