from vn97.r2.mamba2_kaggle_k2k_state_swap import (
    _classify_top1,
    _diagnose,
)


def _matrix(label_by_key: dict[str, str]):
    return {
        key: {"top1_class": value}
        for key, value in label_by_key.items()
    }


def test_classify_top1() -> None:
    assert _classify_top1(
        10,
        reference_token=10,
        ort_token=11,
    ) == "reference"
    assert _classify_top1(
        11,
        reference_token=10,
        ort_token=11,
    ) == "ort"
    assert _classify_top1(
        12,
        reference_token=10,
        ort_token=11,
    ) == "other"


def test_diagnose_runtime_step_dominant() -> None:
    labels = {}
    for state in ("pp", "po", "op", "oo"):
        labels[f"py_{state}"] = "reference"
        labels[f"ort_{state}"] = "ort"
    assert _diagnose(_matrix(labels)) == "runtime_step_dominant"


def test_diagnose_ssm_state_dominant() -> None:
    labels = {}
    for runtime in ("py", "ort"):
        labels[f"{runtime}_pp"] = "reference"
        labels[f"{runtime}_op"] = "reference"
        labels[f"{runtime}_po"] = "ort"
        labels[f"{runtime}_oo"] = "ort"
    assert _diagnose(_matrix(labels)) == "ssm_state_dominant"


def test_diagnose_conv_state_dominant() -> None:
    labels = {}
    for runtime in ("py", "ort"):
        labels[f"{runtime}_pp"] = "reference"
        labels[f"{runtime}_po"] = "reference"
        labels[f"{runtime}_op"] = "ort"
        labels[f"{runtime}_oo"] = "ort"
    assert _diagnose(_matrix(labels)) == "conv_state_dominant"
