import numpy as np
import pytest

from vn97.r2.mamba2_kaggle_k2b_provider_split import compare_cases


def _case(logits, conv, ssm):
    return [{
        "case_index": 0,
        "token_ids": [7],
        "logits": [np.asarray(logits, dtype=np.float16)],
        "final_conv_state": np.asarray(conv, dtype=np.float16),
        "final_ssm_state": np.asarray(ssm, dtype=np.float16),
    }]


def test_compare_cases_exact() -> None:
    left = _case([[1.0, 2.0]], [1.0, 2.0], [3.0, 4.0])
    out = compare_cases(
        left,
        _case([[1.0, 2.0]], [1.0, 2.0], [3.0, 4.0]),
        epsilon=float(np.finfo(np.float16).eps),
    )
    assert out["argmax_exact"] is True
    assert out["steps_compared"] == 1
    assert out["metrics"]["logits"]["max_abs_error"] == 0.0
    assert out["metrics"]["conv_state"]["max_abs_error"] == 0.0
    assert out["metrics"]["ssm_state"]["max_abs_error"] == 0.0


def test_compare_cases_reports_numeric_drift_without_hiding_argmax() -> None:
    left = _case([[1.0, 2.0]], [1.0, 2.0], [3.0, 4.0])
    right = _case([[1.0, 2.125]], [1.0, 2.25], [3.0, 4.5])
    out = compare_cases(
        left,
        right,
        epsilon=float(np.finfo(np.float16).eps),
    )
    assert out["argmax_exact"] is True
    assert out["metrics"]["logits"]["max_abs_error"] > 0.0
    assert out["metrics"]["conv_state"]["max_abs_error"] > 0.0
    assert out["metrics"]["ssm_state"]["max_abs_error"] > 0.0


def test_compare_cases_rejects_token_mismatch() -> None:
    left = _case([[1.0, 2.0]], [1.0], [1.0])
    right = _case([[1.0, 2.0]], [1.0], [1.0])
    right[0]["token_ids"] = [8]
    with pytest.raises(ValueError, match="token sequence mismatch"):
        compare_cases(
            left,
            right,
            epsilon=float(np.finfo(np.float16).eps),
        )
