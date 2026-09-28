import torch

from vn97.r2.mamba2_kaggle_k2d_stage_trace import (
    VN97Mamba2OnnxLayerTrace,
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
        return torch.randn(*shape, generator=g)

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


def test_trace_wrapper_matches_original_layer() -> None:
    layer = _layer()
    cfg = layer.config
    hidden = torch.randn(1, cfg.d_model)
    residual = torch.randn(1, cfg.d_model)
    conv = torch.randn(1, cfg.conv_dim, cfg.d_conv)
    ssm = torch.randn(
        1,
        cfg.n_heads,
        cfg.head_dim,
        cfg.d_state,
    )

    original = layer(hidden, residual, conv, ssm)
    traced = VN97Mamba2OnnxLayerTrace(layer)(
        hidden,
        residual,
        conv,
        ssm,
    )

    torch.testing.assert_close(traced[-1], original[0], rtol=0, atol=0)
    torch.testing.assert_close(traced[0], original[1], rtol=0, atol=0)
    torch.testing.assert_close(traced[3], original[2], rtol=0, atol=0)
    torch.testing.assert_close(traced[9], original[3], rtol=0, atol=0)
