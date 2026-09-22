#include "vn97/packed_ternary.h"

#include "packed_ternary_internal.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace vn97 {
namespace {

constexpr std::uint8_t kMagic[8] = {'V', 'N', '9', '7', 'T', '2', 0, 0};
constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kHeaderSize = 40;

bool MulOverflows(std::size_t a, std::size_t b) {
    return b != 0 && a > std::numeric_limits<std::size_t>::max() / b;
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

namespace internal {

PackedTernaryStatus PackedTernaryMatVecF32Scalar(
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

}  // namespace internal

PackedTernaryStatus PackedTernaryMatVecF32WithBackend(
    const PackedTernaryView& matrix,
    const float* input,
    const float* bias,
    float* output,
    PackedTernaryBackend backend) {
    const PackedTernaryBackend resolved = ResolvePackedTernaryBackend(backend);
    if (!PackedTernaryBackendAvailable(resolved)) {
        return PackedTernaryStatus::kBackendUnavailable;
    }

    switch (resolved) {
        case PackedTernaryBackend::kScalar:
            return internal::PackedTernaryMatVecF32Scalar(
                matrix, input, bias, output);
        case PackedTernaryBackend::kArm64Neon:
#if defined(VN97_HAS_ARM64_NEON)
            return internal::PackedTernaryMatVecF32Arm64Neon(
                matrix, input, bias, output);
#else
            return PackedTernaryStatus::kBackendUnavailable;
#endif
        case PackedTernaryBackend::kAuto:
            break;
    }
    return PackedTernaryStatus::kBackendUnavailable;
}

PackedTernaryStatus PackedTernaryMatVecF32(
    const PackedTernaryView& matrix,
    const float* input,
    const float* bias,
    float* output) {
    return PackedTernaryMatVecF32WithBackend(
        matrix, input, bias, output, PackedTernaryBackend::kAuto);
}

}  // namespace vn97
