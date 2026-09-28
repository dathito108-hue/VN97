from vn97.r2.mamba2_kaggle_k2g_b_path import (
    STAGES,
    _first_over,
)


def _block(units: float):
    return {"d_b_x": {"max_epsilon_units": units}}


def test_first_over_finds_earliest_sensitive_boundary() -> None:
    values = {
        "normalized": _block(0.0),
        "xbc_projected": _block(9.0),
        "next_conv": _block(9.5),
        "conv_affine": _block(9.5),
        "activated_xbc": _block(9.5),
        "b_value": _block(9.5),
    }
    assert _first_over(
        values,
        reference_units=2.0,
        order=STAGES,
    ) == "xbc_projected"


def test_first_over_returns_none_when_not_reproduced() -> None:
    values = {name: _block(1.0) for name in STAGES}
    assert _first_over(
        values,
        reference_units=2.0,
        order=STAGES,
    ) is None
