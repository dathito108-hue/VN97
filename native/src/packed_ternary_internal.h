#pragma once

#include "vn97/packed_ternary.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace vn97::internal {

inline std::uint32_t ReadU32LE(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) |
           (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
}

inline float ReadF32LE(const std::uint8_t* p) {
    const std::uint32_t bits = ReadU32LE(p);
    float value = 0.0f;
    static_assert(sizeof(value) == sizeof(bits), "float32 required");
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline std::size_t SymbolIndex(
    const PackedTernaryView& m,
    std::uint32_t row,
    std::uint32_t col) {
    const std::uint32_t tile_row = row / m.tile_rows;
    const std::uint32_t tile_col = col / m.tile_cols;
    const std::uint32_t inner_row = row % m.tile_rows;
    const std::uint32_t inner_col = col % m.tile_cols;
    const std::uint32_t tile_cols_count = m.padded_cols / m.tile_cols;
    const std::size_t tile_elements =
        static_cast<std::size_t>(m.tile_rows) * m.tile_cols;
    const std::size_t tile_index =
        static_cast<std::size_t>(tile_row) * tile_cols_count + tile_col;
    return tile_index * tile_elements +
           static_cast<std::size_t>(inner_row) * m.tile_cols + inner_col;
}

inline std::uint8_t ReadCode(
    const PackedTernaryView& m,
    std::size_t symbol_index) {
    const std::uint8_t byte = m.packed_data[symbol_index / 4];
    const unsigned shift = static_cast<unsigned>((symbol_index % 4) * 2);
    return static_cast<std::uint8_t>((byte >> shift) & 0x03u);
}

PackedTernaryStatus PackedTernaryMatVecF32Scalar(
    const PackedTernaryView& matrix,
    const float* input,
    const float* bias,
    float* output);

#if defined(VN97_HAS_ARM64_NEON)
PackedTernaryStatus PackedTernaryMatVecF32Arm64Neon(
    const PackedTernaryView& matrix,
    const float* input,
    const float* bias,
    float* output);
#endif

}  // namespace vn97::internal
