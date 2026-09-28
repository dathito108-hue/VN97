import numpy as np

from vn97.r2.mamba2_kaggle_k2p_g05_bridge import (
    MAX_REFERENCE_GAP_ULP_FOR_MISMATCH,
    MIN_TOP5_OVERLAP_FOR_MISMATCH,
    _final_logits,
    _gate,
)


def test_final_logits_selects_valid_prefix_tail() -> None:
    values = np.arange(2 * 4 * 3, dtype=np.float16).reshape(2, 4, 3)
    try:
        _final_logits(values, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("batch other than one must fail")

    values = np.arange(1 * 4 * 3, dtype=np.float16).reshape(1, 4, 3)
    actual = _final_logits(values, 3)
    assert np.array_equal(actual, values[:, 2, :])


def test_gate_passes_exact() -> None:
    passed, failures = _gate(nonfinite_count=0, mismatches=[])
    assert passed is True
    assert failures == []


def test_gate_passes_locked_near_tie_boundary() -> None:
    passed, failures = _gate(
        nonfinite_count=0,
        mismatches=[
            {
                "reference_candidate_gap_ulp":
                    MAX_REFERENCE_GAP_ULP_FOR_MISMATCH,
                "top5_overlap": MIN_TOP5_OVERLAP_FOR_MISMATCH,
            }
        ],
    )
    assert passed is True
    assert failures == []


def test_gate_fails_material_decision_divergence() -> None:
    passed, failures = _gate(
        nonfinite_count=0,
        mismatches=[
            {
                "reference_candidate_gap_ulp": 2.0,
                "top5_overlap": 5,
            }
        ],
    )
    assert passed is False
    assert "mismatch_exceeds_one_reference_fp16_ulp" in failures
