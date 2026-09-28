from vn97.r2.mamba2_kaggle_k2l_layer_influence import (
    _diagnose,
    _rank_records,
    _selected_group_starts,
)


def _rec(
    *,
    mode: str,
    influence: float,
    top1_class: str = "ort",
    start: int = 0,
    layer: int | None = None,
):
    value = {
        "mode": mode,
        "gap_influence": influence,
        "top1_class": top1_class,
        "layer_start": start,
    }
    if layer is not None:
        value["layer"] = layer
    return value


def test_selected_groups_use_best_component_influence() -> None:
    records = [
        _rec(mode="conv", influence=0.1, start=0),
        _rec(mode="ssm", influence=0.5, start=0),
        _rec(mode="both", influence=0.2, start=8),
        _rec(mode="conv", influence=0.8, start=16),
        _rec(mode="both", influence=0.7, start=24),
    ]
    assert _selected_group_starts(records, top_n=2) == [16, 24]


def test_rank_records_descending() -> None:
    records = [
        _rec(mode="both", influence=0.1, start=0, layer=0),
        _rec(mode="both", influence=0.9, start=8, layer=8),
        _rec(mode="both", influence=0.5, start=16, layer=16),
    ]
    ranked = _rank_records(records, limit=2)
    assert [item["layer"] for item in ranked] == [8, 16]


def test_diagnose_single_layer_combined() -> None:
    groups = [_rec(mode="both", influence=0.2, top1_class="reference")]
    layers = [
        _rec(
            mode="both",
            influence=0.3,
            top1_class="reference",
            layer=7,
        )
    ]
    assert _diagnose(
        group_records=groups,
        layer_records=layers,
    ) == "single_layer_state_interaction_localized"


def test_diagnose_component_layer() -> None:
    groups = []
    layers = [
        _rec(
            mode="ssm",
            influence=0.3,
            top1_class="reference",
            layer=7,
        )
    ]
    assert _diagnose(
        group_records=groups,
        layer_records=layers,
    ) == "component_layer_interaction_localized"
