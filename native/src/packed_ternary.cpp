#include "vn97/packed_ternary.h"

#include "packed_ternary_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace vn97 {
namespace {

constexpr std::uint8_t kMagic[8] = {'V', 'N', '9', '7', 'T', '2', 0, 0};
constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kHeaderSize = 40;
constexpr std::uint32_t kMaxTileDimension = 256;
constexpr std::size_t kMaxAccumulatorFloats = kMaxTileDimension;

bool MulOverflows(std::size_t a, std::size_t b) {
    return b != 0 && a > std::numeric_limits<std::size_t>::max() / b;
}

bool ValidViewGeometry(const PackedTernaryView& matrix) {
    return matrix.rows != 0 &&
           matrix.cols != 0 &&
           matrix.tile_rows != 0 &&
           matrix.tile_cols != 0 &&
           matrix.tile_rows <= kMaxTileDimension &&
           matrix.tile_cols <= kMaxTileDimension &&
           matrix.padded_rows >= matrix.rows &&
           matrix.padded_cols >= matrix.cols &&
           matrix.padded_rows % matrix.tile_rows == 0 &&
           matrix.padded_cols % matrix.tile_cols == 0;
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

    const std::uint32_t version = internal::ReadU32LE(blob + 8);
    if (version != kVersion) {
        return PackedTernaryStatus::kUnsupportedVersion;
    }

    PackedTernaryView m;
    m.rows = internal::ReadU32LE(blob + 12);
    m.cols = internal::ReadU32LE(blob + 16);
    m.tile_rows = internal::ReadU32LE(blob + 20);
    m.tile_cols = internal::ReadU32LE(blob + 24);
    m.padded_rows = internal::ReadU32LE(blob + 28);
    m.padded_cols = internal::ReadU32LE(blob + 32);
    m.packed_data_size = internal::ReadU32LE(blob + 36);

    if (!ValidViewGeometry(m)) {
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

    for (std::uint32_t row = 0; row < m.rows; ++row) {
        const float scale = internal::ReadF32LE(
            m.scale_bytes + static_cast<std::size_t>(row) * 4);
        if (!std::isfinite(scale) || !(scale > 0.0f)) {
            return PackedTernaryStatus::kInvalidScale;
        }
    }

    for (std::size_t i = 0; i < symbol_count; ++i) {
        if (internal::ReadCode(m, i) == 3u) {
            return PackedTernaryStatus::kReservedCode;
        }
    }

    *out = m;
    return PackedTernaryStatus::kOk;
}

const char* PackedTernaryBackendName(PackedTernaryBackend backend) {
    switch (backend) {
        case PackedTernaryBackend::kAuto:
            return "auto";
        case PackedTernaryBackend::kScalar:
            return "scalar";
        case PackedTernaryBackend::kArm64Neon:
            return "arm64-neon";
    }
    return "unknown";
}

bool PackedTernaryBackendAvailable(PackedTernaryBackend backend) {
    switch (backend) {
        case PackedTernaryBackend::kAuto:
        case PackedTernaryBackend::kScalar:
            return true;
        case PackedTernaryBackend::kArm64Neon:
#if defined(VN97_HAS_ARM64_NEON)
            return true;
#else
            return false;
#endif
    }
    return false;
}

PackedTernaryBackend ResolvePackedTernaryBackend(PackedTernaryBackend requested) {
    if (requested != PackedTernaryBackend::kAuto) {
        return requested;
    }
#if defined(VN97_HAS_ARM64_NEON)
    return PackedTernaryBackend::kArm64Neon;
#else
    return PackedTernaryBackend::kScalar;
#endif
}

PackedTernaryStatus PlanPackedTernaryMatVecF32(
    const PackedTernaryView& matrix,
    PackedTernaryBackend requested,
    PackedTernaryExecutionPlan* out) {
    if (out == nullptr ||
        matrix.scale_bytes == nullptr ||
        matrix.packed_data == nullptr) {
        return PackedTernaryStatus::kNullArgument;
    }
    if (!ValidViewGeometry(matrix)) {
        return PackedTernaryStatus::kInvalidGeometry;
    }

    const PackedTernaryBackend backend =
        ResolvePackedTernaryBackend(requested);
    if (!PackedTernaryBackendAvailable(backend)) {
        return PackedTernaryStatus::kBackendUnavailable;
    }

    const std::size_t tile_rows_count =
        matrix.padded_rows / matrix.tile_rows;
    const std::size_t tile_cols_count =
        matrix.padded_cols / matrix.tile_cols;
    if (MulOverflows(tile_rows_count, tile_cols_count)) {
        return PackedTernaryStatus::kInvalidGeometry;
    }

    const std::size_t tile_elements =
        static_cast<std::size_t>(matrix.tile_rows) * matrix.tile_cols;
    const std::size_t packed_tile_bytes =
        (tile_elements + 3) / 4;
    const std::size_t input_tile_bytes =
        static_cast<std::size_t>(matrix.tile_cols) * sizeof(float);
    const std::size_t accumulator_floats =
        std::min<std::size_t>(matrix.tile_rows, matrix.rows);
    const std::size_t accumulator_bytes =
        accumulator_floats * sizeof(float);

    PackedTernaryExecutionPlan plan;
    plan.backend = backend;
    plan.rows = matrix.rows;
    plan.cols = matrix.cols;
    plan.row_block =
        static_cast<std::uint32_t>(accumulator_floats);
    plan.col_block = matrix.tile_cols;
    plan.vector_width =
        backend == PackedTernaryBackend::kArm64Neon &&
                matrix.tile_cols >= 16u &&
                matrix.tile_cols % 4u == 0u
            ? 16u
            : 1u;
    plan.accumulator_floats = accumulator_floats;
    plan.tile_count = tile_rows_count * tile_cols_count;
    plan.tile_working_set_bytes =
        packed_tile_bytes + input_tile_bytes + accumulator_bytes;

    *out = plan;
    return PackedTernaryStatus::kOk;
}

namespace internal {

PackedTernaryStatus PackedTernaryMatVecF32TiledScalar(
    const PackedTernaryView& matrix,
    const float* input,
    const float* bias,
    float* output,
    float* accumulator) {
    if (input == nullptr ||
        output == nullptr ||
        accumulator == nullptr ||
        matrix.scale_bytes == nullptr ||
        matrix.packed_data == nullptr) {
        return PackedTernaryStatus::kNullArgument;
    }

    const std::uint32_t tile_rows_count =
        matrix.padded_rows / matrix.tile_rows;
    const std::uint32_t tile_cols_count =
        matrix.padded_cols / matrix.tile_cols;
    const std::size_t tile_elements =
        static_cast<std::size_t>(matrix.tile_rows) * matrix.tile_cols;

    for (std::uint32_t tile_row = 0;
         tile_row < tile_rows_count;
         ++tile_row) {
        const std::uint32_t row_base =
            tile_row * matrix.tile_rows;
        if (row_base >= matrix.rows) break;

        const std::uint32_t valid_rows =
            std::min(matrix.tile_rows, matrix.rows - row_base);
        std::fill(
            accumulator,
            accumulator + valid_rows,
            0.0f);

        for (std::uint32_t tile_col = 0;
             tile_col < tile_cols_count;
             ++tile_col) {
            const std::uint32_t col_base =
                tile_col * matrix.tile_cols;
            if (col_base >= matrix.cols) break;

            const std::uint32_t valid_cols =
                std::min(matrix.tile_cols, matrix.cols - col_base);
            const std::size_t tile_index =
                static_cast<std::size_t>(tile_row) *
                    tile_cols_count +
                tile_col;
            const std::size_t tile_symbol_base =
                tile_index * tile_elements;

            PrefetchRead(input + col_base);
            PrefetchRead(
                matrix.packed_data +
                tile_symbol_base / 4u);

            for (std::uint32_t inner_row = 0;
                 inner_row < valid_rows;
                 ++inner_row) {
                const std::size_t symbol_base =
                    tile_symbol_base +
                    static_cast<std::size_t>(inner_row) *
                        matrix.tile_cols;
                float sum = accumulator[inner_row];

                for (std::uint32_t inner_col = 0;
                     inner_col < valid_cols;
                     ++inner_col) {
                    const std::uint8_t code =
                        ReadCode(
                            matrix,
                            symbol_base + inner_col);
                    if (code == 1u) {
                        sum += input[col_base + inner_col];
                    } else if (code == 2u) {
                        sum -= input[col_base + inner_col];
                    } else if (code == 3u) {
                        return PackedTernaryStatus::kReservedCode;
                    }
                }
                accumulator[inner_row] = sum;
            }
        }

        for (std::uint32_t inner_row = 0;
             inner_row < valid_rows;
             ++inner_row) {
            const std::uint32_t row =
                row_base + inner_row;
            const float scale = ReadF32LE(
                matrix.scale_bytes +
                static_cast<std::size_t>(row) * 4);
            output[row] =
                accumulator[inner_row] * scale +
                (bias != nullptr ? bias[row] : 0.0f);
        }
    }

    return PackedTernaryStatus::kOk;
}

}  // namespace internal

PackedTernaryStatus PackedTernaryMatVecF32WithPlan(
    const PackedTernaryView& matrix,
    const float* input,
    const float* bias,
    float* output,
    const PackedTernaryExecutionPlan& plan) {
    if (input == nullptr ||
        output == nullptr ||
        matrix.scale_bytes == nullptr ||
        matrix.packed_data == nullptr) {
        return PackedTernaryStatus::kNullArgument;
    }
    if (!ValidViewGeometry(matrix)) {
        return PackedTernaryStatus::kInvalidGeometry;
    }
    if (plan.rows != matrix.rows ||
        plan.cols != matrix.cols ||
        plan.row_block !=
            std::min(matrix.tile_rows, matrix.rows) ||
        plan.col_block != matrix.tile_cols ||
        plan.accumulator_floats != plan.row_block ||
        plan.accumulator_floats == 0 ||
        plan.accumulator_floats > kMaxAccumulatorFloats ||
        !PackedTernaryBackendAvailable(plan.backend)) {
        return PackedTernaryStatus::kPlanMismatch;
    }

    std::array<float, kMaxAccumulatorFloats> accumulator{};

    switch (plan.backend) {
        case PackedTernaryBackend::kScalar:
            return internal::PackedTernaryMatVecF32TiledScalar(
                matrix,
                input,
                bias,
                output,
                accumulator.data());
        case PackedTernaryBackend::kArm64Neon:
#if defined(VN97_HAS_ARM64_NEON)
            return internal::PackedTernaryMatVecF32TiledArm64Neon(
                matrix,
                input,
                bias,
                output,
                accumulator.data());
#else
            return PackedTernaryStatus::kBackendUnavailable;
#endif
        case PackedTernaryBackend::kAuto:
            return PackedTernaryStatus::kPlanMismatch;
    }
    return PackedTernaryStatus::kPlanMismatch;
}

PackedTernaryStatus PackedTernaryMatVecF32WithBackend(
    const PackedTernaryView& matrix,
    const float* input,
    const float* bias,
    float* output,
    PackedTernaryBackend backend) {
    PackedTernaryExecutionPlan plan;
    const auto status =
        PlanPackedTernaryMatVecF32(
            matrix,
            backend,
            &plan);
    if (status != PackedTernaryStatus::kOk) {
        return status;
    }
    return PackedTernaryMatVecF32WithPlan(
        matrix,
        input,
        bias,
        output,
        plan);
}

PackedTernaryStatus PackedTernaryMatVecF32(
    const PackedTernaryView& matrix,
    const float* input,
    const float* bias,
    float* output) {
    return PackedTernaryMatVecF32WithBackend(
        matrix,
        input,
        bias,
        output,
        PackedTernaryBackend::kAuto);
}

}  // namespace vn97
