import numpy as np

from vn97.r2.mamba2_kaggle_k2n_ulp_audit import (
    _candidate_gap_ulp,
    _diagnose,
    _fp16_downward_ulp,
)


def test_fp16_downward_ulp_positive() -> None:
    assert _fp16_downward_ulp(-15.8) > 0.0


def test_candidate_gap_ulp_exact_tie() -> None:
    logits = np.asarray([[2.0, 2.0, 1.0]], dtype=np.float16)
    out = _candidate_gap_ulp(
        logits,
        reference_top1=0,
        candidate_token=1,
    )
    assert out["reference_candidate_gap_ulp"] == 0.0
    assert out["within_one_reference_ulp"] is True


def test_candidate_gap_ulp_one_step() -> None:
    top = np.float16(2.0)
    lower = np.nextafter(
        top,
        np.float16(-np.inf),
        dtype=np.float16,
    )
    logits = np.asarray([[top, lower]], dtype=np.float16)
    out = _candidate_gap_ulp(
        logits,
        reference_top1=0,
        candidate_token=1,
    )
    assert abs(float(out["reference_candidate_gap_ulp"]) - 1.0) < 1e-6
    assert out["within_one_reference_ulp"] is True


def test_diagnose_only_near_ties() -> None:
    assert _diagnose(
        mismatch_count=3,
        mismatches_outside_one_ulp=0,
        top5_overlap_min=4,
    ) == "only_fp16_near_tie_decision_divergences"


def test_diagnose_material_divergence() -> None:
    assert _diagnose(
        mismatch_count=2,
        mismatches_outside_one_ulp=1,
        top5_overlap_min=5,
    ) == "material_same_input_decision_divergence_present"
