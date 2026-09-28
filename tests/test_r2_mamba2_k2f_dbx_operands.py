import torch

from vn97.r2.mamba2_kaggle_k2f_dbx_operands import (
    VN97Mamba2DbxMicro,
)


def test_dbx_micro_shapes_and_dtypes() -> None:
    generator = torch.Generator().manual_seed(97)
    dt = torch.randn(2, 5, generator=generator, dtype=torch.float16)
    b = torch.randn(2, 7, generator=generator, dtype=torch.float16)
    x = torch.randn(2, 5, 3, generator=generator, dtype=torch.float16)

    for mode in (
        "einsum",
        "mul_dt_x_b",
        "mul_b_x_dt",
        "fp32_widened",
    ):
        out = VN97Mamba2DbxMicro(mode)(dt, b, x)
        assert out.shape == (2, 5, 3, 7)
        assert out.dtype == torch.float16


def test_dbx_rewrites_are_bounded_against_source_einsum() -> None:
    generator = torch.Generator().manual_seed(9704)
    dt = torch.randn(1, 8, generator=generator, dtype=torch.float16)
    b = torch.randn(1, 9, generator=generator, dtype=torch.float16)
    x = torch.randn(1, 8, 4, generator=generator, dtype=torch.float16)

    source = VN97Mamba2DbxMicro("einsum")(dt, b, x).float()
    eps = torch.finfo(torch.float16).eps
    scale = torch.maximum(torch.ones_like(source), source.abs())

    for mode in ("mul_dt_x_b", "mul_b_x_dt", "fp32_widened"):
        candidate = VN97Mamba2DbxMicro(mode)(dt, b, x).float()
        units = (source - candidate).abs() / (eps * scale)
        assert float(units.max().item()) <= 1.0
