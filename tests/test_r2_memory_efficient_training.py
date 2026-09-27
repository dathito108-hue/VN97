from __future__ import annotations

import copy

import torch

from vn97.r2 import VN97R2Model, r2_smoke_config
from vn97.scan import (
    affine_prefix_scan,
    affine_prefix_scan_memory_efficient,
)


def test_memory_efficient_scan_matches_parallel_scan() -> None:
    torch.manual_seed(9701)
    decay = torch.sigmoid(torch.randn(2, 13, 3, 4))
    drive = torch.randn(2, 13, 3, 4)
    initial = torch.randn(2, 3, 4)

    expected_states, expected_final = affine_prefix_scan(
        decay,
        drive,
        initial,
    )
    actual_states, actual_final = affine_prefix_scan_memory_efficient(
        decay,
        drive,
        initial,
        chunk_size=4,
    )

    torch.testing.assert_close(
        actual_states,
        expected_states,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        actual_final,
        expected_final,
        rtol=1e-5,
        atol=1e-6,
    )


def test_memory_efficient_scan_gradients_match_parallel_scan() -> None:
    torch.manual_seed(9702)
    base_decay = torch.sigmoid(torch.randn(1, 9, 2, 3))
    base_drive = torch.randn(1, 9, 2, 3)
    base_initial = torch.randn(1, 2, 3)

    decay_a = base_decay.clone().requires_grad_(True)
    drive_a = base_drive.clone().requires_grad_(True)
    initial_a = base_initial.clone().requires_grad_(True)
    states_a, final_a = affine_prefix_scan(
        decay_a,
        drive_a,
        initial_a,
    )
    loss_a = states_a.square().mean() + final_a.square().mean()
    loss_a.backward()

    decay_b = base_decay.clone().requires_grad_(True)
    drive_b = base_drive.clone().requires_grad_(True)
    initial_b = base_initial.clone().requires_grad_(True)
    states_b, final_b = affine_prefix_scan_memory_efficient(
        decay_b,
        drive_b,
        initial_b,
        chunk_size=3,
    )
    loss_b = states_b.square().mean() + final_b.square().mean()
    loss_b.backward()

    torch.testing.assert_close(
        decay_b.grad,
        decay_a.grad,
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(
        drive_b.grad,
        drive_a.grad,
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(
        initial_b.grad,
        initial_a.grad,
        rtol=2e-5,
        atol=2e-6,
    )


def test_activation_checkpointed_training_forward_preserves_logits() -> None:
    torch.manual_seed(9703)
    config = r2_smoke_config(vocab_size=96)
    model = VN97R2Model(config)
    tokens = torch.randint(0, config.vocab_size, (2, 10))

    normal_logits, _ = model(tokens, profile="deep")
    checkpointed_logits, _ = model.forward_training(
        tokens,
        profile="deep",
        activation_checkpointing=True,
    )

    torch.testing.assert_close(
        checkpointed_logits,
        normal_logits,
        rtol=2e-5,
        atol=2e-6,
    )


def test_activation_checkpointed_training_gradients_match() -> None:
    torch.manual_seed(9704)
    config = r2_smoke_config(vocab_size=80)
    base = VN97R2Model(config)
    reference = copy.deepcopy(base)
    checkpointed = copy.deepcopy(base)
    tokens = torch.randint(0, config.vocab_size, (1, 7))
    targets = torch.randint(0, config.vocab_size, (1, 7))

    reference_logits, _ = reference.forward_training(
        tokens,
        profile="deep",
        activation_checkpointing=False,
    )
    reference_loss = torch.nn.functional.cross_entropy(
        reference_logits.reshape(-1, config.vocab_size),
        targets.reshape(-1),
    )
    reference_loss.backward()

    checkpointed_logits, _ = checkpointed.forward_training(
        tokens,
        profile="deep",
        activation_checkpointing=True,
    )
    checkpointed_loss = torch.nn.functional.cross_entropy(
        checkpointed_logits.reshape(-1, config.vocab_size),
        targets.reshape(-1),
    )
    checkpointed_loss.backward()

    torch.testing.assert_close(
        checkpointed_loss,
        reference_loss,
        rtol=1e-6,
        atol=1e-7,
    )

    ref_params = dict(reference.named_parameters())
    chk_params = dict(checkpointed.named_parameters())
    assert ref_params.keys() == chk_params.keys()
    for name in ref_params:
        assert ref_params[name].grad is not None, name
        assert chk_params[name].grad is not None, name
        torch.testing.assert_close(
            chk_params[name].grad,
            ref_params[name].grad,
            rtol=3e-4,
            atol=3e-5,
            msg=lambda message, n=name: f"{n}: {message}",
        )
