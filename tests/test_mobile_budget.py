import pytest

from vn97.config import VN97Config
from vn97.model import VN97LanguageCore
from vn97.model_image import build_model_image
from vn97.mobile_budget import (
    VN97MobileBudget,
    estimate_vn97_mobile_footprint,
)
from vn97.tokenizer import learn_byte_bpe


def test_mobile_footprint_matches_vn97mi1_full_embedding():
    config = VN97Config(
        vocab_size=300,
        d_model=16,
        n_layers=2,
        d_state=4,
    )
    model = VN97LanguageCore(config)
    image = build_model_image(
        model,
        tile_rows=16,
        tile_cols=16,
    )
    footprint = estimate_vn97_mobile_footprint(
        config,
        tile_rows=16,
        tile_cols=16,
    )

    assert footprint.model_image_bytes == len(image.data)
    assert footprint.recurrent_state_bytes == config.recurrent_state_bytes()
    assert footprint.tokenizer_bytes == 0
    assert footprint.packed_ternary_bytes > 0
    assert footprint.float_parameter_bytes > 0


def test_mobile_footprint_matches_factorized_image_with_tokenizer():
    package = learn_byte_bpe(
        ["hello mobile vn97", "native recurrent intelligence"],
        max_learned_tokens=8,
        min_pair_count=1,
    )
    tokenizer_bytes = package.to_bytes()
    config = VN97Config(
        vocab_size=package.vocab_size,
        d_model=24,
        n_layers=3,
        d_state=4,
        embedding_rank=8,
    )
    model = VN97LanguageCore(config)
    image = build_model_image(
        model,
        tokenizer=package,
        tile_rows=8,
        tile_cols=16,
    )
    footprint = estimate_vn97_mobile_footprint(
        config,
        tokenizer_nbytes=len(tokenizer_bytes),
        tile_rows=8,
        tile_cols=16,
    )

    assert footprint.model_image_bytes == len(image.data)
    assert footprint.tokenizer_bytes == len(tokenizer_bytes)


def test_mobile_budget_rejects_model_then_state_deterministically():
    footprint = estimate_vn97_mobile_footprint(
        VN97Config(
            vocab_size=300,
            d_model=16,
            n_layers=2,
            d_state=4,
        ),
    )

    model_budget = VN97MobileBudget(
        max_model_image_bytes=footprint.model_image_bytes - 1,
        max_recurrent_state_bytes=footprint.recurrent_state_bytes - 1,
    )
    assert (
        model_budget.rejection_status(footprint)
        == "REJECTED_MODEL_IMAGE_BUDGET"
    )

    state_budget = VN97MobileBudget(
        max_model_image_bytes=footprint.model_image_bytes,
        max_recurrent_state_bytes=footprint.recurrent_state_bytes - 1,
    )
    assert (
        state_budget.rejection_status(footprint)
        == "REJECTED_RECURRENT_STATE_BUDGET"
    )

    passing = VN97MobileBudget(
        max_model_image_bytes=footprint.model_image_bytes,
        max_recurrent_state_bytes=footprint.recurrent_state_bytes,
    )
    assert passing.rejection_status(footprint) is None


def test_mobile_footprint_rejects_invalid_tiles():
    config = VN97Config(
        vocab_size=300,
        d_model=16,
        n_layers=1,
        d_state=4,
    )
    with pytest.raises(ValueError, match="tile dimensions"):
        estimate_vn97_mobile_footprint(
            config,
            tile_rows=0,
            tile_cols=16,
        )
