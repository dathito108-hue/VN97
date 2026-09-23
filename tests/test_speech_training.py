import math

import torch

from vn97 import (
    AudioAdapterConfig,
    AudioFrameAdapter,
    VN97Config,
    VN97LanguageCore,
    VN97SpeechExample,
    VN97SpeechReleaseCriteria,
    VN97SpeechTrainingConfig,
    VN97Tokenizer,
    VN97TokenizerPackage,
    evaluate_vn97_speech,
    require_speech_release_quality,
    train_vn97_speech_adapter,
)


def _fixture():
    torch.manual_seed(97)
    tokenizer = VN97Tokenizer(VN97TokenizerPackage())
    model = VN97LanguageCore(
        VN97Config(
            vocab_size=tokenizer.vocab_size,
            d_model=8,
            n_layers=1,
            d_state=2,
        )
    )
    adapter = AudioFrameAdapter(
        model.config.d_model,
        ternary_threshold=model.config.ternary_threshold,
        config=AudioAdapterConfig(),
        rms_eps=model.config.rms_eps,
    )
    t = torch.arange(640, dtype=torch.float32)
    waveform = 0.2 * torch.sin(t * 0.07)
    examples = (
        VN97SpeechExample(waveform, "a"),
        VN97SpeechExample(waveform.flip(0), "b"),
    )
    return model, adapter, tokenizer, examples


def test_speech_adapter_training_updates_only_adapter_and_evaluates():
    model, adapter, tokenizer, examples = _fixture()
    model_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    adapter_before = {
        name: value.detach().clone()
        for name, value in adapter.state_dict().items()
    }
    config = VN97SpeechTrainingConfig(
        epochs=1,
        learning_rate=1e-3,
        max_frames=8,
        max_target_tokens=16,
        seed=97,
        shuffle=False,
    )
    result = train_vn97_speech_adapter(
        model,
        adapter,
        tokenizer,
        examples,
        config,
        device="cpu",
    )
    assert result.steps == len(examples)
    assert result.target_tokens > 0
    assert math.isfinite(result.mean_loss)
    assert math.isfinite(result.final_loss)
    assert not model.training
    assert not adapter.training

    assert all(
        torch.equal(model_before[name], value)
        for name, value in model.state_dict().items()
    )
    assert any(
        not torch.equal(adapter_before[name], value)
        for name, value in adapter.state_dict().items()
    )

    evaluation = evaluate_vn97_speech(
        model,
        adapter,
        tokenizer,
        examples,
        config=config,
        device="cpu",
    )
    assert evaluation.examples == len(examples)
    assert evaluation.target_tokens > 0
    assert math.isfinite(evaluation.mean_loss)
    assert 0.0 <= evaluation.top1_accuracy <= 1.0

    require_speech_release_quality(
        evaluation,
        VN97SpeechReleaseCriteria(
            max_validation_loss=max(100.0, evaluation.mean_loss + 1.0),
            min_top1_accuracy=0.0,
            min_target_tokens=1,
            min_examples=1,
        ),
    )


def test_speech_example_and_bounds_fail_closed():
    model, adapter, tokenizer, examples = _fixture()
    import pytest

    with pytest.raises(ValueError, match="max_target_tokens"):
        evaluate_vn97_speech(
            model,
            adapter,
            tokenizer,
            examples,
            config=VN97SpeechTrainingConfig(
                max_frames=8,
                max_target_tokens=2,
                shuffle=False,
            ),
            device="cpu",
        )

    with pytest.raises(ValueError, match="frame count"):
        evaluate_vn97_speech(
            model,
            adapter,
            tokenizer,
            examples,
            config=VN97SpeechTrainingConfig(
                max_frames=1,
                max_target_tokens=16,
                shuffle=False,
            ),
            device="cpu",
        )
