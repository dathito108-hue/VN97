import math

import torch

from vn97 import (
    VN97Config,
    VN97LanguageCore,
    VN97Tokenizer,
    VN97TokenizerPackage,
    VN97VisionExample,
    VN97VisionReleaseCriteria,
    VN97VisionTrainingConfig,
    VisionAdapterConfig,
    VisionPatchAdapter,
    evaluate_vn97_vision,
    require_vision_release_quality,
    train_vn97_vision_adapter,
)


def test_vision_training_updates_only_adapter_and_evaluates():
    tokenizer = VN97Tokenizer(VN97TokenizerPackage())
    model = VN97LanguageCore(
        VN97Config(
            vocab_size=tokenizer.vocab_size,
            d_model=8,
            n_layers=1,
            d_state=2,
        )
    )
    adapter = VisionPatchAdapter(
        model.config.d_model,
        ternary_threshold=model.config.ternary_threshold,
        config=VisionAdapterConfig(),
        rms_eps=model.config.rms_eps,
    )
    image = torch.linspace(
        0.0,
        1.0,
        3 * 16 * 16,
        dtype=torch.float32,
    ).reshape(3, 16, 16)
    examples = (
        VN97VisionExample(image=image, description="a"),
        VN97VisionExample(image=image.flip(-1), description="b"),
    )
    before_model = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    before_adapter = {
        name: value.detach().clone()
        for name, value in adapter.state_dict().items()
    }
    config = VN97VisionTrainingConfig(
        epochs=1,
        learning_rate=1e-3,
        max_patches=4,
        max_target_tokens=16,
        shuffle=False,
    )
    result = train_vn97_vision_adapter(
        model,
        adapter,
        tokenizer,
        examples,
        config,
        device="cpu",
    )
    assert result.steps == 2
    assert math.isfinite(result.mean_loss)
    assert all(
        torch.equal(before_model[name], value)
        for name, value in model.state_dict().items()
    )
    assert any(
        not torch.equal(before_adapter[name], value)
        for name, value in adapter.state_dict().items()
    )

    evaluation = evaluate_vn97_vision(
        model,
        adapter,
        tokenizer,
        examples,
        config=config,
        device="cpu",
    )
    assert evaluation.examples == 2
    assert evaluation.target_tokens > 0
    assert 0.0 <= evaluation.top1_accuracy <= 1.0
    require_vision_release_quality(
        evaluation,
        VN97VisionReleaseCriteria(
            max_validation_loss=max(100.0, evaluation.mean_loss + 1.0),
            min_target_tokens=1,
            min_examples=1,
        ),
    )
