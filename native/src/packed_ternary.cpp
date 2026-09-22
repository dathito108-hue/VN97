#include "vn97/packed_ternary.h"

#include <cstring>
#include <limits>

namespace vn97 {
namespace {

constexpr std::uint8_t kMagic[8] = {'V', 'N', '9', '7', 'T', '2', 0, 0};
constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kHeaderSize = 40;

std::uint32_t ReadU32LE(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) |
           (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
}

float ReadF32LE(const std::uint8_t* p) {
    const std::uint32_t bits = ReadU32LE(p);
    float value = 0.0f;
    static_assert(sizeof(value) == sizeof(bits), "float32 required");
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

bool MulOverflows(std::size_t a, std::size_t b) {
    return b != 0 && a > std::numeric_limits<std::size_t>::max() / b;
}

std::size_t SymbolIndex(
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

std::uint8_t ReadCode(const PackedTernaryView& m, std::size_t symbol_index) {
    const std::uint8_t byte = m.packed_data[symbol_index / 4];
    const unsigned shift = static_cast<unsigned>((symbol_index % 4) * 2);
    return static_cast<std::uint8_t>((byte >> shift) & 0x03u);
}

}  // namespace

PackedTernaryStatus ParsePackedTernary(
    const std::uint8_t* blob,
    std::size_t blob_size,
    PackedTernaryView* out) {
    if (blob == nullptr || out == nullptr) {
        return PackedTernaryStatus::kNullArgument;
    }
    if (blob_size < kHeaderSize) {
        return PackedTernaryStatus::kBlobTooShort;
    }
    if (std::memcmp(blob, kMagic, sizeof(kMagic)) != 0) {
        return PackedTernaryStatus::kBadMagic;
    }

    const std::uint32_t version = ReadU32LE(blob + 8);
    if (version != kVersion) {
        return PackedTernaryStatus::kUnsupportedVersion;
    }

    PackedTernaryView m;
    m.rows = ReadU32LE(blob + 12);
    m.cols = ReadU32LE(blob + 16);
    m.tile_rows = ReadU32LE(blob + 20);
    m.tile_cols = ReadU32LE(blob + 24);
    m.padded_rows = ReadU32LE(blob + 28);
    m.padded_cols = ReadU32LE(blob + 32);
    m.packed_data_size = ReadU32LE(blob + 36);

    if (m.rows == 0 || m.cols == 0 || m.tile_rows == 0 || m.tile_cols == 0 ||
        m.padded_rows < m.rows || m.padded_cols < m.cols ||
        m.padded_rows % m.tile_rows != 0 || m.padded_cols % m.tile_cols != 0) {
        return PackedTernaryStatus::kInvalidGeometry;
    }

    if (MulOverflows(m.padded_rows, m.padded_cols)) {
        return PackedTernaryStatus::kInvalidGeometry;
    }
    const std::size_t symbol_count =
        static_cast<std::size_t>(m.padded_rows) * m.padded_cols;
    const std::size_t expected_packed = (symbol_count + 3) / 4;
    if (m.packed_data_size != expected_packed) {
        return PackedTernaryStatus::kInvalidLength;
    }
    if (MulOverflows(m.rows, 4)) {
        return PackedTernaryStatus::kInvalidLength;
    }
    const std::size_t scale_bytes = static_cast<std::size_t>(m.rows) * 4;
    const std::size_t expected_size = kHeaderSize + scale_bytes + expected_packed;
    if (blob_size != expected_size) {
        return PackedTernaryStatus::kInvalidLength;
    }

    m.scale_bytes = blob + kHeaderSize;
    m.packed_data = m.scale_bytes + scale_bytes;

    for (std::size_t i = 0; i < symbol_count; ++i) {
        if (ReadCode(m, i) == 3u) {
            return PackedTernaryStatus::kReservedCode;
        }
    }

    *out = m;
    return PackedTernaryStatus::kOk;
}

PackedTernaryStatus PackedTernaryMatVecF32(
    const PackedTernaryView& matrix,
    const float* input,
    const float* bias,
    float* output) {
    if (input == nullptr || output == nullptr || matrix.scale_bytes == nullptr ||
        matrix.packed_data == nullptr) {
        return PackedTernaryStatus::kNullArgument;
    }

    for (std::uint32_t row = 0; row < matrix.rows; ++row) {
        float sum = 0.0f;
        for (std::uint32_t col = 0; col < matrix.cols; ++col) {
            const std::uint8_t code = ReadCode(matrix, SymbolIndex(matrix, row, col));
            if (code == 1u) {
                sum += input[col];
            } else if (code == 2u) {
                sum -= input[col];
            } else if (code == 3u) {
                return PackedTernaryStatus::kReservedCode;
            }
        }
        const float scale = ReadF32LE(
            matrix.scale_bytes + static_cast<std::size_t>(row) * 4);
        output[row] = sum * scale + (bias != nullptr ? bias[row] : 0.0f);
    }
    return PackedTernaryStatus::kOk;
}

}  // namespace vn97
