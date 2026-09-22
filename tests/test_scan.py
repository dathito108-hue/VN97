import torch

from vn97 import (
    SelectiveSSM,
    affine_prefix_scan,
    affine_scan_rounds,
)


def test_affine_scan_matches_sequential_with_nonzero_initial_state():
    torch.manual_seed(1)
    decay = torch.sigmoid(
        torch.randn(2, 19, 3, 5)
    )
    drive = torch.randn(
        2, 19, 3, 5
    )
    initial = torch.randn(
        2, 3, 5
    )

    states, final_state = (
        affine_prefix_scan(
            decay,
            drive,
            initial,
        )
    )

    sequential = initial
    expected_states = []
    for t in range(decay.shape[1]):
        sequential = (
            decay[:, t] * sequential
            + drive[:, t]
        )
        expected_states.append(
            sequential
        )

    expected = torch.stack(
        expected_states,
        dim=1,
    )

    torch.testing.assert_close(
        states,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        final_state,
        expected[:, -1],
        rtol=1e-5,
        atol=1e-6,
    )


def test_scan_round_count_is_logarithmic():
    assert affine_scan_rounds(1) == 0
    assert affine_scan_rounds(2) == 1
    assert affine_scan_rounds(4096) == 12
    assert affine_scan_rounds(4097) == 13


def test_selective_ssm_parallel_matches_sequential_reference():
    torch.manual_seed(2)

    layer = SelectiveSSM(
        12,
        6,
        ternary_threshold=0.5,
        dt_min=1e-4,
        dt_max=1.0,
        min_decay=0.01,
        max_decay=16.0,
    ).eval()

    x = torch.randn(
        2, 17, 12
    )
    initial = torch.randn(
        2, 12, 6
    )

    with torch.no_grad():
        parallel_y, parallel_state = (
            layer(x, initial)
        )
        sequential_y, sequential_state = (
            layer.forward_sequential_reference(
                x, initial
            )
        )

    torch.testing.assert_close(
        parallel_y,
        sequential_y,
        rtol=1e-4,
        atol=1e-5,
    )
    torch.testing.assert_close(
        parallel_state,
        sequential_state,
        rtol=1e-4,
        atol=1e-5,
    )


def test_parallel_scan_backward_has_finite_gradients():
    torch.manual_seed(3)

    decay_logits = torch.randn(
        2,
        13,
        4,
        3,
        requires_grad=True,
    )
    drive = torch.randn(
        2,
        13,
        4,
        3,
        requires_grad=True,
    )

    decay = torch.sigmoid(
        decay_logits
    )
    states, final_state = (
        affine_prefix_scan(
            decay, drive
        )
    )

    loss = (
        states.square().mean()
        + final_state.abs().mean()
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(
        decay_logits.grad
    ).all()
    assert torch.isfinite(
        drive.grad
    ).all()
