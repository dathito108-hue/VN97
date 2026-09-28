import torch

from vn97.r2.mamba2_onnx import _mamba2_d_b_x_activation


def test_explicit_dbx_stays_within_one_fp16_epsilon_of_einsum() -> None:
    for seed in (1, 7, 97, 9704):
        generator = torch.Generator().manual_seed(seed)
        dt = torch.randn(2, 5, generator=generator, dtype=torch.float16)
        b = torch.randn(2, 7, generator=generator, dtype=torch.float16)
        x = torch.randn(2, 5, 3, generator=generator, dtype=torch.float16)

        expected = torch.einsum("bh,bn,bhp->bhpn", dt, b, x)
        actual = _mamba2_d_b_x_activation(dt, b, x)

        assert actual.dtype == torch.float16
        left = expected.float()
        diff = (left - actual.float()).abs()
        scale = torch.maximum(torch.ones_like(left), left.abs())
        units = diff / (torch.finfo(torch.float16).eps * scale)
        assert float(units.max().item()) <= 1.0


def test_explicit_dbx_preserves_shape() -> None:
    dt = torch.ones(1, 4, dtype=torch.float16)
    b = torch.ones(1, 6, dtype=torch.float16)
    x = torch.ones(1, 4, 8, dtype=torch.float16)
    out = _mamba2_d_b_x_activation(dt, b, x)
    assert out.shape == (1, 4, 8, 6)
