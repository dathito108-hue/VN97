import math

import torch

from vn97 import (
    VN97ChatMessage,
    VN97Config,
    VN97LanguageCore,
    VN97Tokenizer,
    VN97TokenizerPackage,
    VN97TrainingConfig,
    build_deployment_checkpoint,
    build_training_windows,
    encode_causal_text,
    encode_chat_messages,
    load_deployment_checkpoint,
    make_training_windows,
    train_vn97_language,
)


def test_chat_sft_masks_nonassistant_content_and_targets_assistant():
    tokenizer = VN97Tokenizer()
    example = encode_chat_messages(
        tokenizer,
        (
            VN97ChatMessage("system", "Be concise."),
            VN97ChatMessage("user", "2+2?"),
            VN97ChatMessage("assistant", "4"),
        ),
    )
    assert len(example.token_ids) == len(example.target_mask)
    assert any(example.target_mask)
    assert not all(example.target_mask)
    assert example.target_mask[-1]  # EOS is supervised after final assistant.

    windows = make_training_windows(
        example,
        sequence_length=64,
        pad_token_id=tokenizer.pad_id,
    )
    assert windows
    assert sum(window.target_tokens for window in windows) > 0
    assert any(label == -100 for label in windows[0].labels)


def test_causal_windows_are_deterministic_and_fixed_length():
    tokenizer = VN97Tokenizer()
    example = encode_causal_text(tokenizer, "abc" * 20)
    first = make_training_windows(
        example,
        sequence_length=16,
        stride=8,
        pad_token_id=tokenizer.pad_id,
    )
    second = make_training_windows(
        example,
        sequence_length=16,
        stride=8,
        pad_token_id=tokenizer.pad_id,
    )
    assert first == second
    assert all(len(window.input_ids) == 16 for window in first)
    assert all(len(window.labels) == 16 for window in first)


def test_native_training_loop_updates_model_and_exports_vn97ck1():
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
    before = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }

    examples = [
        encode_causal_text(tokenizer, "aaaa aaaa aaaa"),
        encode_causal_text(tokenizer, "aaaa aaaa"),
    ]
    config = VN97TrainingConfig(
        sequence_length=12,
        batch_size=2,
        epochs=1,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        seed=97,
    )
    windows = build_training_windows(
        examples,
        config,
        pad_token_id=tokenizer.pad_id,
    )
    result = train_vn97_language(
        model,
        windows,
        config,
        device="cpu",
    )
    assert result.steps > 0
    assert result.target_tokens > 0
    assert math.isfinite(result.mean_loss)
    assert math.isfinite(result.final_loss)
    assert not model.training

    after = model.state_dict()
    assert any(
        not torch.equal(before[name], after[name])
        for name in before
    )

    checkpoint = build_deployment_checkpoint(model)
    restored = load_deployment_checkpoint(checkpoint)
    assert restored.config == model.config
