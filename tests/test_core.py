import torch

from vn97 import (
    VN97Config,
    VN97LanguageCore,
    quantize_ternary_per_channel,
)


def tiny_config() -> VN97Config:
    return VN97Config(
        vocab_size=97,
        d_model=24,
        n_layers=3,
        d_state=8,
    )


def test_quantizer_uses_only_zero_or_signed_channel_scale():
    weight = torch.tensor(
        [
            [0.0, 0.25, -2.0, 1.0],
            [3.0, -0.1, 0.0, -3.0],
        ]
    )
    q = quantize_ternary_per_channel(
        weight, threshold=0.5
    )
    scales = weight.abs().mean(
        dim=1, keepdim=True
    )
    normalized = q / scales
    allowed = torch.tensor(
        [-1.0, 0.0, 1.0]
    )
    assert torch.isin(
        normalized.flatten(), allowed
    ).all()


def test_full_sequence_matches_recurrent_token_steps():
    torch.manual_seed(7)
    model = VN97LanguageCore(
        tiny_config()
    ).eval()
    tokens = torch.randint(
        0,
        model.config.vocab_size,
        (2, 9),
    )

    with torch.no_grad():
        full_logits, full_states = model(
            tokens
        )
        step_states = None
        step_logits = []
        for t in range(tokens.shape[1]):
            logits, step_states = model(
                tokens[:, t : t + 1],
                step_states,
            )
            step_logits.append(logits)
        recurrent_logits = torch.cat(
            step_logits, dim=1
        )

    torch.testing.assert_close(
        full_logits,
        recurrent_logits,
        rtol=1e-4,
        atol=1e-5,
    )
    assert step_states is not None
    for a, b in zip(
        full_states, step_states
    ):
        torch.testing.assert_close(
            a, b, rtol=1e-4, atol=1e-5
        )


def test_state_shape_is_independent_of_context_length():
    model = VN97LanguageCore(
        tiny_config()
    ).eval()
    with torch.no_grad():
        _, short_states = model(
            torch.randint(
                0,
                model.config.vocab_size,
                (1, 2),
            )
        )
        _, long_states = model(
            torch.randint(
                0,
                model.config.vocab_size,
                (1, 17),
            )
        )

    expected = (
        1,
        model.config.d_model,
        model.config.d_state,
    )
    assert all(
        tuple(s.shape) == expected
        for s in short_states
    )
    assert all(
        tuple(s.shape) == expected
        for s in long_states
    )


def test_training_step_has_finite_gradients():
    torch.manual_seed(11)
    model = VN97LanguageCore(
        tiny_config()
    ).train()
    tokens = torch.randint(
        0,
        model.config.vocab_size,
        (2, 6),
    )
    targets = torch.roll(
        tokens, shifts=-1, dims=1
    )
    logits, _ = model(tokens)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(
            -1, model.config.vocab_size
        ),
        targets.reshape(-1),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert all(
        p.grad is None
        or torch.isfinite(p.grad).all()
        for p in model.parameters()
    )
