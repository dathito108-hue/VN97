import torch

from vn97 import (
    PackedTernaryMatrix,
    TernaryLinear,
    pack_ternary_symbols,
    pack_ternary_weight,
    packed_linear_reference,
    quantize_ternary_per_channel,
)


def test_known_two_bit_encoding_is_lsb_first_and_tile_major():
    symbols = torch.tensor(
        [
            [0, 1, -1, 1],
            [-1, 0, 1, -1],
        ],
        dtype=torch.int8,
    )
    scales = torch.tensor([2.0, 0.5])
    packed = pack_ternary_symbols(
        symbols,
        scales,
        tile_rows=2,
        tile_cols=4,
    )
    assert packed.data == bytes([0x64, 0x92])
    torch.testing.assert_close(packed.ternary_symbols(), symbols)


def test_pack_roundtrip_preserves_quantized_weight_with_padding():
    torch.manual_seed(4)
    weight = torch.randn(19, 37)
    packed = pack_ternary_weight(
        weight,
        threshold=0.5,
        tile_rows=8,
        tile_cols=16,
    )
    restored = packed.dequantize()
    expected = quantize_ternary_per_channel(weight, threshold=0.5)
    torch.testing.assert_close(restored, expected)
    assert packed.padded_rows == 24
    assert packed.padded_cols == 48


def test_serialized_format_roundtrip_and_rejects_reserved_code():
    weight = torch.tensor([[1.0, -1.0, 0.01, 3.0]])
    packed = pack_ternary_weight(weight, tile_rows=1, tile_cols=4)
    blob = packed.to_bytes()
    restored = PackedTernaryMatrix.from_bytes(blob)
    assert restored.rows == packed.rows
    assert restored.cols == packed.cols
    assert restored.data == packed.data
    torch.testing.assert_close(restored.scales, packed.scales)

    corrupted = bytearray(blob)
    corrupted[-1] |= 0b11
    try:
        PackedTernaryMatrix.from_bytes(bytes(corrupted))
    except ValueError as exc:
        assert "reserved" in str(exc)
    else:
        raise AssertionError("reserved ternary code must be rejected")


def test_packed_reference_linear_matches_training_quantized_linear():
    torch.manual_seed(21)
    layer = TernaryLinear(31, 17, bias=True, threshold=0.5).eval()
    x = torch.randn(3, 5, 31)
    packed = layer.export_packed(tile_rows=8, tile_cols=16)
    with torch.no_grad():
        expected = layer(x)
        actual = packed_linear_reference(x, packed, layer.bias)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_large_aligned_matrix_is_close_to_sixteen_x_smaller_than_fp32():
    torch.manual_seed(9)
    packed = pack_ternary_weight(
        torch.randn(128, 256),
        tile_rows=16,
        tile_cols=16,
    )
    assert packed.compression_ratio_vs_fp32 > 14.0
