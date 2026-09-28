import numpy as np

from vn97.r2.mamba2_kaggle_k2m_tie_sensitivity import (
    _diagnose,
    _interpolate,
    _perturbation_summary,
)


def test_interpolate_endpoints() -> None:
    ort = np.asarray([1.0, 2.0], dtype=np.float16)
    py = np.asarray([3.0, 4.0], dtype=np.float16)
    assert np.array_equal(_interpolate(ort, py, alpha=0.0), ort)
    assert np.array_equal(_interpolate(ort, py, alpha=1.0), py)


def test_perturbation_summary_changed_fraction() -> None:
    left = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    right = np.asarray([1.0, 2.5, 3.0, 5.0], dtype=np.float16)
    out = _perturbation_summary(left, right)
    assert out["changed_fraction"] == 0.5
    assert out["max_abs_delta"] == 1.0


def test_diagnose_distributed_near_tie() -> None:
    candidates = [
        {
            "layer": 1,
            "mode": "conv",
            "minimal_alpha_to_reference": 0.125,
            "k2l_gap_influence": 0.01,
        },
        {
            "layer": 13,
            "mode": "both",
            "minimal_alpha_to_reference": 0.25,
            "k2l_gap_influence": 0.02,
        },
        {
            "layer": 15,
            "mode": "ssm",
            "minimal_alpha_to_reference": 0.25,
            "k2l_gap_influence": 0.03,
        },
    ]
    assert _diagnose(candidates) == "distributed_near_tie_sensitivity"


def test_diagnose_localized() -> None:
    candidates = [
        {
            "layer": 1,
            "mode": "conv",
            "minimal_alpha_to_reference": 0.125,
            "k2l_gap_influence": 0.02,
        },
        {
            "layer": 2,
            "mode": "conv",
            "minimal_alpha_to_reference": 1.0,
            "k2l_gap_influence": 0.01,
        },
    ]
    assert _diagnose(candidates) == "localized_state_sensitivity"
