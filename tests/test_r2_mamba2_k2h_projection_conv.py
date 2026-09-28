import torch

from vn97.r2.mamba2_kaggle_k2h_projection_conv import (
    VN97Mamba2ConvAffineMicro,
    _choose_candidate,
)
from vn97.r2.mamba2_onnx import (
    Mamba2OnnxConfig,
    VN97Mamba2OnnxLayer,
)


def _layer() -> VN97Mamba2OnnxLayer:
    cfg = Mamba2OnnxConfig(
        d_model=8,
        n_layers=1,
        vocab_size=16,
        d_state=4,
        d_conv=3,
        expand=2,
        head_dim=4,
        n_groups=1,
    )
    g = torch.Generator().manual_seed(97)

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g, dtype=torch.float16)

    return VN97Mamba2OnnxLayer(
        config=cfg,
        block_norm=rand(cfg.d_model),
        in_proj=rand(
            cfg.d_inner + cfg.conv_dim + cfg.n_heads,
            cfg.d_model,
        ),
        conv_weight=rand(cfg.conv_dim, 1, cfg.d_conv),
        conv_bias=rand(cfg.conv_dim),
        dt_bias=rand(cfg.n_heads),
        a_log=rand(cfg.n_heads),
        d_skip=rand(cfg.n_heads),
        mixer_norm=rand(cfg.d_inner),
        out_proj=rand(cfg.d_model, cfg.d_inner),
    ).eval()


def test_widened_conv_preserves_shape_and_dtype() -> None:
    layer = _layer()
    cfg = layer.config
    next_conv = torch.randn(
        1,
        cfg.conv_dim,
        cfg.d_conv,
        dtype=torch.float16,
    )
    out = VN97Mamba2ConvAffineMicro(
        layer,
        "fp32_widened",
    )(next_conv)
    assert out.shape == (1, cfg.conv_dim)
    assert out.dtype == torch.float16


def test_choose_candidate_prefers_lowest_passing_dbx() -> None:
    metrics = {
        "a": {"d_b_x": {"max_epsilon_units": 1.5}},
        "b": {"d_b_x": {"max_epsilon_units": 0.5}},
        "c": {"d_b_x": {"max_epsilon_units": 3.0}},
    }
    assert _choose_candidate(metrics, reference_units=2.0) == "b"


def test_choose_candidate_returns_none_without_passing_mode() -> None:
    metrics = {
        "a": {"d_b_x": {"max_epsilon_units": 3.0}},
        "b": {"d_b_x": {"max_epsilon_units": 4.0}},
    }
    assert _choose_candidate(metrics, reference_units=2.0) is None
