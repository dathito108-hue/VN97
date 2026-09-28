import numpy as np

from vn97.r2.mamba2_kaggle_k2i_trajectory import (
    CHECKPOINT_STEPS,
    TOKEN_COUNT,
    _top1_margin,
)


def test_k2i_checkpoint_schedule_reaches_full_horizon() -> None:
    assert CHECKPOINT_STEPS == (1, 2, 4, 8, 16, 32)
    assert CHECKPOINT_STEPS[-1] == TOKEN_COUNT


def test_top1_margin() -> None:
    logits = np.asarray([[1.0, 4.5, 2.0, 4.0]], dtype=np.float32)
    assert _top1_margin(logits) == 0.5
