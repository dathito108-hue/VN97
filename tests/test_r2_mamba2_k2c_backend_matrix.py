import numpy as np
import pytest

from vn97.r2.mamba2_kaggle_k2c_backend_matrix import (
    _layerwise_state_metrics,
)


def _case(conv, ssm):
    return {
        "final_conv_state": np.asarray(conv, dtype=np.float16),
        "final_ssm_state": np.asarray(ssm, dtype=np.float16),
    }


def test_layerwise_state_metrics_exact() -> None:
    left = _case(
        [[[1.0, 2.0]], [[3.0, 4.0]]],
        [[[5.0, 6.0]], [[7.0, 8.0]]],
    )
    out = _layerwise_state_metrics(
        left,
        left,
        epsilon=float(np.finfo(np.float16).eps),
        diagnostic_reference_units=2.0,
    )
    assert out["first_layer_over_reference"] is None
    assert out["max_layer_epsilon_units"] == 0.0
    assert len(out["layers"]) == 2


def test_layerwise_state_metrics_finds_first_divergent_layer() -> None:
    left = _case(
        [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]],
        [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]],
    )
    right = _case(
        [[[1.0, 2.0]], [[3.0, 4.03125]], [[5.0, 6.0]]],
        [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.5]]],
    )
    out = _layerwise_state_metrics(
        left,
        right,
        epsilon=float(np.finfo(np.float16).eps),
        diagnostic_reference_units=2.0,
    )
    assert out["first_layer_over_reference"] == 1
    assert out["layers"][1]["conv_max_abs_error"] > 0.0
    assert out["layers"][2]["ssm_max_abs_error"] > 0.0


def test_layerwise_state_metrics_rejects_shape_mismatch() -> None:
    left = _case([[[1.0]]], [[[1.0]]])
    right = _case([[[1.0]], [[2.0]]], [[[1.0]], [[2.0]]])
    with pytest.raises(ValueError, match="shape mismatch"):
        _layerwise_state_metrics(
            left,
            right,
            epsilon=float(np.finfo(np.float16).eps),
            diagnostic_reference_units=2.0,
        )
