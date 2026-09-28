from vn97.r2.mamba2_kaggle_k2o_holdout_gate import (
    MAX_REFERENCE_GAP_ULP_FOR_MISMATCH,
    MIN_TOP5_OVERLAP_FOR_MISMATCH,
    _gate,
)


def test_gate_passes_exact_parity() -> None:
    passed, failures = _gate(
        nonfinite_count=0,
        mismatch_records=[],
    )
    assert passed is True
    assert failures == []


def test_gate_passes_one_ulp_near_tie_with_top5_overlap() -> None:
    passed, failures = _gate(
        nonfinite_count=0,
        mismatch_records=[
            {
                "reference_candidate_gap_ulp":
                    MAX_REFERENCE_GAP_ULP_FOR_MISMATCH,
                "top5_overlap": MIN_TOP5_OVERLAP_FOR_MISMATCH,
            }
        ],
    )
    assert passed is True
    assert failures == []


def test_gate_fails_material_gap() -> None:
    passed, failures = _gate(
        nonfinite_count=0,
        mismatch_records=[
            {
                "reference_candidate_gap_ulp": 1.5,
                "top5_overlap": 5,
            }
        ],
    )
    assert passed is False
    assert "mismatch_exceeds_one_reference_fp16_ulp" in failures


def test_gate_fails_rank_neighborhood_change() -> None:
    passed, failures = _gate(
        nonfinite_count=0,
        mismatch_records=[
            {
                "reference_candidate_gap_ulp": 0.5,
                "top5_overlap": 3,
            }
        ],
    )
    assert passed is False
    assert "mismatch_top5_overlap_below_four" in failures


def test_gate_fails_nonfinite() -> None:
    passed, failures = _gate(
        nonfinite_count=1,
        mismatch_records=[],
    )
    assert passed is False
    assert "nonfinite_logits" in failures
