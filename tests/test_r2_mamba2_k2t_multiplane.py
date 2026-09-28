from __future__ import annotations

import torch

from vn97.r2.mamba2_kaggle_k2t_multiplane_ternary import (
    _multiplane_metrics,
    _packed_plane_bytes,
    _projection_keys,
)


def test_k2t_projection_inventory() -> None:
    keys = _projection_keys(64)
    assert len(keys) == 128
    assert keys[0] == (
        "vn97.core.layers.0.mixer.in_proj.weight",
        "in_proj",
        0,
    )
    assert keys[-1] == (
        "vn97.core.layers.63.mixer.out_proj.weight",
        "out_proj",
        63,
    )


def test_k2t_multiplane_residual_improves_controlled_tensor() -> None:
    weight = torch.tensor(
        [
            [-2.0, -1.0, 0.0, 1.0, 2.0],
            [3.0, 1.5, 0.0, -1.5, -3.0],
        ],
        dtype=torch.float32,
    )
    result = _multiplane_metrics(
        weight,
        plane_count=4,
        threshold=0.5,
    )
    rel = [
        item["relative_rmse"]
        for item in result["planes"]
    ]
    assert all(0.0 <= value < 1.0 for value in rel)
    assert rel[1] < rel[0]
    assert rel[2] <= rel[1]
    assert rel[3] <= rel[2]


def test_k2t_packed_plane_bytes_include_header_and_scales() -> None:
    rows = 16
    cols = 16
    expected = 40 + rows * 4 + (rows * cols) // 4
    assert _packed_plane_bytes(rows, cols) == expected
