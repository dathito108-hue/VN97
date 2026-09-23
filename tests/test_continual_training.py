import math

import pytest
import torch

from vn97 import (
    VN97Config,
    VN97LanguageCore,
    VN97ReleaseCriteria,
    VN97ReleaseQualityError,
    VN97Tokenizer,
    VN97TrainingConfig,
    build_training_windows,
    encode_causal_text,
    guarded_fine_tune_vn97,
)


def _windows(tokenizer, texts, config):
    return build_training_windows(
        [encode_causal_text(tokenizer, text) for text in texts],
        config,
        pad_token_id=tokenizer.pad_id,
    )


def test_guarded_finetune_updates_native_vn97_and_returns_validation():
    torch.manual_seed(97)
    tokenizer = VN97Tokenizer()
    model = VN97LanguageCore(
        VN97Config(
            vocab_size=tokenizer.vocab_size,
            d_model=8,
            n_layers=1,
            d_state=2,
        )
    )
    before = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    config = VN97TrainingConfig(
        sequence_length=16,
        batch_size=2,
        epochs=1,
        learning_rate=1e-3,
        seed=97,
        shuffle=False,
    )
    train = _windows(
        tokenizer,
        ["aaaa aaaa aaaa", "bbbb bbbb bbbb"],
        config,
    )
    validation = _windows(
        tokenizer,
        ["held out validation one", "held out validation two"],
        config,
    )
    result = guarded_fine_tune_vn97(
        model,
        train,
        validation,
        config,
        VN97ReleaseCriteria(
            max_validation_loss=100.0,
            min_target_tokens=1,
        ),
        max_validation_loss_increase=100.0,
        device="cpu",
    )
    assert result.training.steps > 0
    assert math.isfinite(result.baseline.mean_loss)
    assert math.isfinite(result.final.mean_loss)
    assert result.baseline.target_tokens == result.final.target_tokens
    assert any(
        not torch.equal(before[name], model.state_dict()[name])
        for name in before
    )
    assert not model.training


def test_guarded_finetune_fails_absolute_release_gate():
    torch.manual_seed(97)
    tokenizer = VN97Tokenizer()
    model = VN97LanguageCore(
        VN97Config(
            vocab_size=tokenizer.vocab_size,
            d_model=8,
            n_layers=1,
            d_state=2,
        )
    )
    config = VN97TrainingConfig(
        sequence_length=16,
        batch_size=1,
        epochs=1,
        learning_rate=1e-3,
        seed=97,
        shuffle=False,
    )
    windows = _windows(
        tokenizer,
        ["small sample for quality gate"],
        config,
    )
    with pytest.raises(VN97ReleaseQualityError, match="loss"):
        guarded_fine_tune_vn97(
            model,
            windows,
            windows,
            config,
            VN97ReleaseCriteria(
                max_validation_loss=0.000001,
                min_target_tokens=1,
            ),
            max_validation_loss_increase=100.0,
            device="cpu",
        )
