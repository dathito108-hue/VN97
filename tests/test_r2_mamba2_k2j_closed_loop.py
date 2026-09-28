import numpy as np

from vn97.r2.mamba2_kaggle_k2j_closed_loop import (
    _common_prefix_length,
    _topk_ids,
    _topk_overlap,
)


def test_common_prefix_length() -> None:
    assert _common_prefix_length([1, 2, 3], [1, 2, 4]) == 2
    assert _common_prefix_length([1, 2], [1, 2]) == 2
    assert _common_prefix_length([1], [2]) == 0


def test_topk_ids_are_ranked() -> None:
    logits = np.asarray([[0.1, 4.0, 2.0, 5.0]], dtype=np.float32)
    assert _topk_ids(logits, 3) == [3, 1, 2]


def test_topk_overlap() -> None:
    left = np.asarray([[5.0, 4.0, 3.0, 2.0]], dtype=np.float32)
    right = np.asarray([[5.0, 4.0, 1.0, 3.0]], dtype=np.float32)
    assert _topk_overlap(left, right, 2) == 2
