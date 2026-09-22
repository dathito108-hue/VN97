# VN97T2 packed ternary format v1

VN97T2 is the first deployment representation for VN97 ternary projection weights.
It replaces the training-time FP32/FP16 weight tensor with physical ternary symbols and one
FP32 scale per output channel.

## Why 2 physical bits instead of calling it 1.58-bit

A ternary alphabet has an information lower bound of `log2(3) ~= 1.585` bits/symbol, but a
random-access, SIMD-friendly fixed-width representation needs 2 bits per weight in v1.
Calling an ordinary floating-point tensor "1.58-bit" does not reduce memory traffic. VN97T2
actually stores four ternary weights per byte.

Codes are LSB-first within each byte:

- `00` = zero
- `01` = +1
- `10` = -1
- `11` = reserved and rejected

The dequantized value for output row `r` is `symbol * scale[r]`.

## Tile-major layout

Matrices are padded to configurable `(tile_rows, tile_cols)` geometry. Tiles are stored in
row-major tile order, and values inside each tile are row-major. A tile is therefore a
contiguous unit for a future CPU/GPU/NPU kernel.

The format does **not** claim that 16x16 or 32x32 is universally optimal. The packer accepts
backend-selected tile geometry so Android/iOS runtime profiling can choose an implementation
that matches the actual device.

## Serialized blob

All integer fields are little-endian.

1. 8-byte magic: `VN97T2` followed by two zero bytes.
2. `u32 version` (currently 1).
3. `u32 rows`.
4. `u32 cols`.
5. `u32 tile_rows`.
6. `u32 tile_cols`.
7. `u32 padded_rows`.
8. `u32 padded_cols`.
9. `u32 packed_data_nbytes`.
10. `rows` FP32 little-endian scales.
11. packed 2-bit tile-major symbols.

Bias is intentionally not part of the weight blob. Runtime operators may retain a small bias
vector in FP16/FP32 as appropriate.

## Runtime contract

The portable scalar C++ kernel is a correctness baseline. Optimized backends must match it
within the accumulator tolerance while avoiding full dequantization of the matrix.
Future implementations may use ARM NEON, SME/SVE where available, Metal, Vulkan, NNAPI or
vendor delegates without changing the VN97 model semantics.
