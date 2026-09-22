#pragma once

#include <cstddef>
#include <cstdint>

namespace vn97 {

struct PackedTernaryView {
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    std::uint32_t tile_rows = 0;
    std::uint32_t tile_cols = 0;
    std::uint32_t padded_rows = 0;
    std::uint32_t padded_cols = 0;
    const std::uint8_t* scale_bytes = nullptr;
    const std::uint8_t* packed_data = nullptr;
    std::size_t packed_data_size = 0;
};

enum class PackedTernaryStatus {
    kOk = 0,
    kNullArgument,
    kBlobTooShort,
    kBadMagic,
    kUnsupportedVersion,
    kInvalidGeometry,
    kInvalidLength,
    kReservedCode,
    kBackendUnavailable,
    kInvalidScale,
    kPlanMismatch,
};

enum class PackedTernaryBackend {
    kAuto = 0,
    kScalar,
    kArm64Neon,
};

struct PackedTernaryExecutionPlan {
    PackedTernaryBackend backend = PackedTernaryBackend::kScalar;
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    std::uint32_t row_block = 0;
    std::uint32_t col_block = 0;
    std::uint32_t vector_width = 1;
    std::size_t accumulator_floats = 0;
    std::size_t tile_count = 0;
    std::size_t tile_working_set_bytes = 0;
};

PackedTernaryStatus ParsePackedTernary(
    const std::uint8_t* blob,
    std::size_t blob_size,
    PackedTernaryView* out);

const char* PackedTernaryBackendName(PackedTernaryBackend backend);

bool PackedTernaryBackendAvailable(PackedTernaryBackend backend);

PackedTernaryBackend ResolvePackedTernaryBackend(PackedTernaryBackend requested);

PackedTernaryStatus PlanPackedTernaryMatVecF32(
    const PackedTernaryView& matrix,
    PackedTernaryBackend requested,
    PackedTernaryExecutionPlan* out);

PackedTernaryStatus PackedTernaryMatVecF32WithBackend(
    const PackedTernaryView& matrix,
    const float* input,
    const float* bias,
    float* output,
    PackedTernaryBackend backend);

PackedTernaryStatus PackedTernaryMatVecF32(
    const PackedTernaryView& matrix,
    const float* input,
    const float* bias,
    float* output);

}  // namespace vn97
