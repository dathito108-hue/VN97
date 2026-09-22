import pytest
import torch
import torch.nn.functional as F

from vn97 import (
    FactorizedEmbedding,
    FactorizedLMHead,
    VN97Config,
    VN97LanguageCore,
)


def _tiny_config(
    rank: int | None = None,
) -> VN97Config:
    return VN97Config(
        vocab_size=97,
        d_model=24,
        n_layers=3,
        d_state=8,
        embedding_rank=rank,
    )


def _assert_full_equals_recurrent(
    model: VN97LanguageCore,
) -> None:
    model.eval()
    tokens = torch.randint(
        0,
        model.config.vocab_size,
        (2, 9),
    )

    with torch.no_grad():
        full_logits, full_states = (
            model(tokens)
        )
        step_states = None
        step_logits = []
        for t in range(
            tokens.shape[1]
        ):
            logits, step_states = (
                model(
                    tokens[
                        :, t : t + 1
                    ],
                    step_states,
                )
            )
            step_logits.append(
                logits
            )

    recurrent_logits = torch.cat(
        step_logits,
        dim=1,
    )
    torch.testing.assert_close(
        full_logits,
        recurrent_logits,
        rtol=1e-4,
        atol=1e-5,
    )
    assert step_states is not None
    for full, step in zip(
        full_states,
        step_states,
    ):
        torch.testing.assert_close(
            full,
            step,
            rtol=1e-4,
            atol=1e-5,
        )


def test_factorized_embedding_and_head_are_exactly_tied():
    torch.manual_seed(3)
    embedding = FactorizedEmbedding(
        vocab_size=101,
        d_model=32,
        rank=8,
    )
    head = FactorizedLMHead(
        embedding
    )

    assert (
        head.token_factors
        is embedding.token_factors
    )
    assert (
        head.projection
        is embedding.projection
    )

    ids = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    hidden = torch.randn(
        2, 3, 32
    )
    effective = (
        embedding.effective_weight()
    )
    torch.testing.assert_close(
        embedding(ids),
        F.embedding(
            ids,
            effective,
        ),
    )
    torch.testing.assert_close(
        head(hidden),
        F.linear(
            hidden,
            effective,
        ),
    )


def test_factorized_parameter_count():
    config = VN97Config(
        vocab_size=32000,
        d_model=512,
        embedding_rank=64,
    )
    assert (
        config.tied_embedding_parameters()
        == 32000 * 64
        + 64 * 512
    )
    assert (
        config.embedding_compression_ratio()
        < 0.14
    )

    with pytest.raises(ValueError):
        VN97Config(
            vocab_size=97,
            d_model=24,
            embedding_rank=24,
        )


def test_legacy_embedding_path_remains_default():
    torch.manual_seed(7)
    model = VN97LanguageCore(
        _tiny_config()
    )
    assert isinstance(
        model.embedding,
        torch.nn.Embedding,
    )
    assert (
        model.lm_head.weight
        is model.embedding.weight
    )
    assert (
        model.config.embedding_compression_ratio()
        == 1.0
    )
    _assert_full_equals_recurrent(
        model
    )


def test_factorized_model_preserves_recurrent_invariant_and_gradients():
    torch.manual_seed(17)
    model = VN97LanguageCore(
        _tiny_config(rank=8)
    )

    assert (
        model.embedding.token_factors
        is model.lm_head.token_factors
    )
    assert (
        model.embedding.projection
        is model.lm_head.projection
    )

    _assert_full_equals_recurrent(
        model
    )

    model.train()
    tokens = torch.randint(
        0,
        model.config.vocab_size,
        (2, 6),
    )
    logits, _ = model(tokens)
    loss = logits.square().mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None
        or torch.isfinite(
            parameter.grad
        ).all()
        for parameter
        in model.parameters()
    )
