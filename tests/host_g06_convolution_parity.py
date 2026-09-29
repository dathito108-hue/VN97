"""Regression for G06 chunk convolution's FP16 rounding boundaries."""
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import g06_deployment_preflight
import torch
from torch.nn import functional as F

Chunk = importlib.import_module('_g06_contracts.mamba2_parallel_onnx').VN97Mamba2ParallelChunkOnnx


class ConvolutionParity(unittest.TestCase):
    def test_chunk_equals_individual_step_convolutions(self):
        torch.set_num_threads(2)
        for dtype in (torch.float16, torch.float32):
            for chunk in (8, 16, 32):
                for width in (3, 4):
                    with self.subTest(dtype=dtype, chunk=chunk, width=width):
                        generator = torch.Generator().manual_seed(9721)
                        window = torch.randn(1, 24, chunk + width - 1, generator=generator).to(dtype)
                        layer = SimpleNamespace(
                            conv_weight=torch.randn(24, 1, width, generator=generator).to(dtype),
                            conv_bias=torch.randn(24, generator=generator).to(dtype),
                        )
                        context = SimpleNamespace(config=SimpleNamespace(d_conv=width), chunk_size=chunk)
                        actual = Chunk._causal_convolution_affine(context, layer, window)
                        expected = torch.stack([
                            torch.sum(window[..., i:i + width] * layer.conv_weight[:, 0, :], dim=-1)
                            + layer.conv_bias
                            for i in range(chunk)
                        ], dim=1)
                        self.assertEqual(actual.dtype, dtype)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        if dtype == torch.float16:
                            fused = F.conv1d(window, layer.conv_weight, layer.conv_bias, groups=24).transpose(1, 2)
                            self.assertGreater((fused - expected).abs().max().item(), 0,
                                               'fixture must detect the previous fused rounding mismatch')


if __name__ == '__main__':
    unittest.main()
