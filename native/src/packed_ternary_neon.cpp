#include "packed_ternary_internal.h"

#if !defined(__aarch64__)
#error "packed_ternary_neon.cpp must only be built for AArch64"
#endif

#include <arm_neon.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace vn97::internal {
namespace {

constexpr std::int8_t kDecodeCode[4] = {0, 1, -1, 0};

inline void Decode16Signs(
    const std::uint8_t* packed,
    std::int8_t* signs) {
    for (std::size_t byte_index = 0; byte_index < 4; ++byte_index) {
        const std::uint8_t byte = packed[byte_index];
        signs[byte_index * 4 + 0] = kDecodeCode[(byte >> 0) & 0x03u];
        signs[byte_index * 4 + 1] = kDecodeCode[(byte >> 2) & 0x03u];
        signs[byte_index * 4 + 2] = kDecodeCode[(byte >> 4) & 0x03u];
        signs[byte_index * 4 + 3] = kDecodeCode[(byte >> 6) & 0x03u];
    }
}

inline float Dot16Ternary(
    const float* input,
    const std::uint8_t* packed) {
    alignas(16) std::int8_t signs[16];
    Decode16Signs(packed, signs);

    const int8x16_t s8 = vld1q_s8(signs);
    const int16x8_t s16_lo = vmovl_s8(vget_low_s8(s8));
    const int16x8_t s16_hi = vmovl_s8(vget_high_s8(s8));

    const float32x4_t s0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16_lo)));
    const float32x4_t s1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16_lo)));
    const float32x4_t s2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(s16_hi)));
    const float32x4_t s3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(s16_hi)));

    float32x4_t acc = vmulq_f32(vld1q_f32(input + 0), s0);
    acc = vfmaq_f32(acc, vld1q_f32(input + 4), s1);
    acc = vfmaq_f32(acc, vld1q_f32(input + 8), s2);
    acc = vfmaq_f32(acc, vld1q_f32(input + 12), s3);
    return vaddvq_f32(acc);
}

inline float DotScalarRange(
    const PackedTernaryView& matrix,
    std::size_t symbol_base,
    const float* input,
    std::uint32_t count) {
    float sum = 0.0f;
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::uint8_t code = ReadCode(matrix, symbol_base + i);
        if (code == 1u) {
            sum += input[i];
        } else if (code == 2u) {
            sum -= input[i];
        }
    }
    return sum;
}

}  // namespace

PackedTernaryStatus PackedTernaryMatVecF32Arm64Neon(
    const PackedTernaryView& matrix,
    const float* input,
    const float* bias,
    float* output) {
    if (input == nullptr || output == nullptr || matrix.scale_bytes == nullptr ||
        matrix.packed_data == nullptr) {
        return PackedTernaryStatus::kNullArgument;
    }

    const std::uint32_t tile_cols_count = matrix.padded_cols / matrix.tile_cols;
    const std::size_t tile_elements =
        static_cast<std::size_t>(matrix.tile_rows) * matrix.tile_cols;

    for (std::uint32_t row = 0; row < matrix.rows; ++row) {
        const std::uint32_t tile_row = row / matrix.tile_rows;
        const std::uint32_t inner_row = row % matrix.tile_rows;
        float sum = 0.0f;

        for (std::uint32_t tile_col = 0; tile_col < tile_cols_count; ++tile_col) {
            const std::uint32_t col_base = tile_col * matrix.tile_cols;
            if (col_base >= matrix.cols) {
                break;
            }
            const std::uint32_t valid_cols = std::min(
                matrix.tile_cols, matrix.cols - col_base);
            const std::size_t tile_index =
                static_cast<std::size_t>(tile_row) * tile_cols_count + tile_col;
            const std::size_t symbol_base =
                tile_index * tile_elements +
                static_cast<std::size_t>(inner_row) * matrix.tile_cols;

            std::uint32_t consumed = 0;
            if ((symbol_base % 4u) == 0u) {
                for (; consumed + 16u <= valid_cols; consumed += 16u) {
                    const std::size_t symbol_index = symbol_base + consumed;
                    const std::uint8_t* packed =
                        matrix.packed_data + symbol_index / 4u;
                    sum += Dot16Ternary(input + col_base + consumed, packed);
                }
            }
            if (consumed < valid_cols) {
                sum += DotScalarRange(
                    matrix,
                    symbol_base + consumed,
                    input + col_base + consumed,
                    valid_cols - consumed);
            }
        }

        const float scale = ReadF32LE(
            matrix.scale_bytes + static_cast<std::size_t>(row) * 4u);
        output[row] = sum * scale + (bias != nullptr ? bias[row] : 0.0f);
    }
    return PackedTernaryStatus::kOk;
}

}  // namespace vn97::internal
