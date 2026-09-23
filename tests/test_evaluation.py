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
    evaluate_vn97_language,
    require_release_quality,
)


def test_evaluation_is_token_weighted_and_quality_gated():
    torch.manual_seed(97)
    tokenizer = VN97Tokenizer()
    model = VN97LanguageCore(
        VN97Config(
            vocab_size=tokenizer.vocab_size,
            d_model=8,
            n_layers=1,
            d_state=2,
        )
    ).eval()
    config = VN97TrainingConfig(
        sequence_length=16,
        batch_size=2,
        shuffle=False,
    )
    windows = build_training_windows(
        [
            encode_causal_text(tokenizer, "held out alpha"),
            encode_causal_text(tokenizer, "held out beta"),
        ],
        config,
        pad_token_id=tokenizer.pad_id,
    )
    result = evaluate_vn97_language(
        model,
        windows,
        batch_size=2,
        device="cpu",
    )
    assert result.windows == len(windows)
    assert result.target_tokens > 0
    assert math.isfinite(result.mean_loss)
    assert 0.0 <= result.top1_accuracy <= 1.0

    require_release_quality(
        result,
        VN97ReleaseCriteria(
            max_validation_loss=result.mean_loss + 1e-6,
            min_top1_accuracy=result.top1_accuracy,
            min_target_tokens=result.target_tokens,
        ),
    )
    with pytest.raises(VN97ReleaseQualityError, match="loss"):
        require_release_quality(
            result,
            VN97ReleaseCriteria(
                max_validation_loss=max(1e-9, result.mean_loss / 2),
                min_target_tokens=1,
            ),
        )
    with pytest.raises(VN97ReleaseQualityError, match="target-token"):
        require_release_quality(
            result,
            VN97ReleaseCriteria(
                max_validation_loss=result.mean_loss + 1,
                min_target_tokens=result.target_tokens + 1,
            ),
        )
