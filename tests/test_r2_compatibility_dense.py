from __future__ import annotations

import math

import torch

from vn97.cognition_adapter import TorchVN97InferenceEngine
from vn97.r2 import (
    R2DenseTrainingConfig,
    R2TrainingStage,
    VN97R2InferenceView,
    VN97R2Model,
    assert_r2_data_compatible,
    build_completion_windows,
    load_r2_checkpoint,
    load_vn97tk1,
    r2_smoke_config,
    train_dense,
)
from vn97.r2.training_plan import R2_CANONICAL_TRAINING_SEQUENCE
from vn97.tokenizer import VN97Tokenizer, VN97TokenizerPackage
from vn97.training import (
    VN97ChatMessage,
    VN97TrainingConfig,
)


def _conversations():
    return (
        (
            VN97ChatMessage(role="user", content="2+2?"),
            VN97ChatMessage(role="assistant", content="4"),
        ),
        (
            VN97ChatMessage(role="user", content="Say yes."),
            VN97ChatMessage(role="assistant", content="yes"),
        ),
    )


def test_r2_vn97tk1_bridge_and_existing_embedding_engine(tmp_path) -> None:
    package = VN97TokenizerPackage()
    path = tmp_path / "tokenizer.vn97tk1"
    path.write_bytes(package.to_bytes())
    tokenizer = load_vn97tk1(path)

    config = r2_smoke_config(tokenizer.vocab_size)
    model = VN97R2Model(config).eval()
    view = VN97R2InferenceView(model, profile="deep")
    engine = TorchVN97InferenceEngine(view, tokenizer)

    vector = engine.embed_text(
        "VN97 R2",
        vector_dim=config.d_model,
    )
    assert len(vector) == config.d_model
    assert all(math.isfinite(value) for value in vector)


def test_r2_completion_windows_use_existing_chat_contract() -> None:
    tokenizer = VN97Tokenizer()
    model = VN97R2Model(
        r2_smoke_config(tokenizer.vocab_size)
    )
    config = VN97TrainingConfig(
        sequence_length=32,
        stride=16,
        batch_size=1,
        epochs=1,
        max_windows=100,
    )
    windows = build_completion_windows(
        tokenizer,
        _conversations(),
        config,
    )
    assert windows
    assert_r2_data_compatible(model, tokenizer, windows)
    assert sum(window.target_tokens for window in windows) > 0


def test_r2_dense_trainer_checkpoint_resume_contract(tmp_path) -> None:
    torch.manual_seed(12)
    tokenizer = VN97Tokenizer()
    model = VN97R2Model(
        r2_smoke_config(tokenizer.vocab_size)
    )
    window_config = VN97TrainingConfig(
        sequence_length=32,
        stride=16,
        batch_size=1,
        epochs=1,
        max_windows=100,
    )
    windows = build_completion_windows(
        tokenizer,
        _conversations(),
        window_config,
    )
    result = train_dense(
        model,
        windows,
        windows,
        R2DenseTrainingConfig(
            epochs=1,
            batch_size=1,
            learning_rate=5e-4,
            checkpoint_every_steps=1,
            seed=12,
        ),
        work_dir=tmp_path / "work",
        best_checkpoint_path=tmp_path / "best.r2.pt",
        dataset_identity="test-dataset",
        device="cpu",
    )
    assert result.steps > 0
    assert result.target_tokens > 0
    assert math.isfinite(result.best_validation_loss)
    assert len(result.best_checkpoint_sha256) == 64

    loaded, evidence = load_r2_checkpoint(
        tmp_path / "best.r2.pt"
    )
    assert loaded.config.fingerprint() == model.config.fingerprint()
    assert evidence["stage"] == "dense_pretrain"


def test_r2_training_order_contains_fast_alignment_before_validation() -> None:
    order = list(R2_CANONICAL_TRAINING_SEQUENCE)
    assert R2TrainingStage.FAST_PATH_ALIGNMENT.value in order
    assert order.index(
        R2TrainingStage.FAST_PATH_ALIGNMENT.value
    ) < order.index(R2TrainingStage.FRESH_VALIDATION.value)
