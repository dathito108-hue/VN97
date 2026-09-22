#include "vn97/packed_ternary.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

void PushU32(std::vector<std::uint8_t>& v, std::uint32_t x) {
    v.push_back(static_cast<std::uint8_t>(x));
    v.push_back(static_cast<std::uint8_t>(x >> 8));
    v.push_back(static_cast<std::uint8_t>(x >> 16));
    v.push_back(static_cast<std::uint8_t>(x >> 24));
}

void SetU32(
    std::vector<std::uint8_t>& v,
    std::size_t offset,
    std::uint32_t x) {
    for (int i = 0; i < 4; ++i) {
        v[offset + static_cast<std::size_t>(i)] =
            static_cast<std::uint8_t>(x >> (8 * i));
    }
}

void PushF32(std::vector<std::uint8_t>& v, float x) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    PushU32(v, bits);
}

std::uint8_t EncodeSymbol(int symbol) {
    if (symbol > 0) return 1u;
    if (symbol < 0) return 2u;
    return 0u;
}

void PushFourSymbols(
    std::vector<std::uint8_t>& v,
    int s0,
    int s1,
    int s2,
    int s3) {
    const std::uint8_t packed =
        EncodeSymbol(s0) |
        static_cast<std::uint8_t>(EncodeSymbol(s1) << 2) |
        static_cast<std::uint8_t>(EncodeSymbol(s2) << 4) |
        static_cast<std::uint8_t>(EncodeSymbol(s3) << 6);
    v.push_back(packed);
}

std::vector<std::uint8_t> BuildTwoBySixteenBlob() {
    std::vector<std::uint8_t> blob =
        {'V', 'N', '9', '7', 'T', '2', 0, 0};
    PushU32(blob, 1);
    PushU32(blob, 2);
    PushU32(blob, 16);
    PushU32(blob, 2);
    PushU32(blob, 16);
    PushU32(blob, 2);
    PushU32(blob, 16);
    PushU32(blob, 8);
    PushF32(blob, 1.0f);
    PushF32(blob, 0.25f);
    for (int i = 0; i < 4; ++i) {
        PushFourSymbols(blob, 1, -1, 1, -1);
    }
    for (int i = 0; i < 4; ++i) {
        PushFourSymbols(blob, 1, 1, 0, -1);
    }
    return blob;
}

int DenseSymbol(std::uint32_t row, std::uint32_t col) {
    return static_cast<int>((row * 7u + col * 3u + 1u) % 3u) - 1;
}

std::vector<std::uint8_t> BuildTiledBlob(
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t tile_rows,
    std::uint32_t tile_cols,
    std::vector<float>* scales_out) {
    const auto round_up = [](std::uint32_t value, std::uint32_t multiple) {
        return ((value + multiple - 1u) / multiple) * multiple;
    };
    const std::uint32_t padded_rows =
        round_up(rows, tile_rows);
    const std::uint32_t padded_cols =
        round_up(cols, tile_cols);
    const std::size_t symbol_count =
        static_cast<std::size_t>(padded_rows) * padded_cols;
    const std::size_t packed_bytes =
        (symbol_count + 3u) / 4u;

    std::vector<std::uint8_t> blob =
        {'V', 'N', '9', '7', 'T', '2', 0, 0};
    PushU32(blob, 1);
    PushU32(blob, rows);
    PushU32(blob, cols);
    PushU32(blob, tile_rows);
    PushU32(blob, tile_cols);
    PushU32(blob, padded_rows);
    PushU32(blob, padded_cols);
    PushU32(
        blob,
        static_cast<std::uint32_t>(packed_bytes));

    scales_out->clear();
    for (std::uint32_t row = 0; row < rows; ++row) {
        const float scale =
            0.125f * static_cast<float>(row + 1u);
        scales_out->push_back(scale);
        PushF32(blob, scale);
    }

    std::vector<std::uint8_t> codes;
    codes.reserve(symbol_count);
    const std::uint32_t tile_row_count =
        padded_rows / tile_rows;
    const std::uint32_t tile_col_count =
        padded_cols / tile_cols;
    for (std::uint32_t tr = 0; tr < tile_row_count; ++tr) {
        for (std::uint32_t tc = 0; tc < tile_col_count; ++tc) {
            for (std::uint32_t ir = 0; ir < tile_rows; ++ir) {
                const std::uint32_t row = tr * tile_rows + ir;
                for (std::uint32_t ic = 0; ic < tile_cols; ++ic) {
                    const std::uint32_t col = tc * tile_cols + ic;
                    const int symbol =
                        row < rows && col < cols
                            ? DenseSymbol(row, col)
                            : 0;
                    codes.push_back(EncodeSymbol(symbol));
                }
            }
        }
    }

    for (std::size_t index = 0; index < codes.size(); index += 4u) {
        std::uint8_t byte = 0;
        for (std::size_t lane = 0; lane < 4u; ++lane) {
            const std::size_t at = index + lane;
            const std::uint8_t code =
                at < codes.size() ? codes[at] : 0u;
            byte |= static_cast<std::uint8_t>(
                code << (2u * lane));
        }
        blob.push_back(byte);
    }

    return blob;
}

void AssertNear(
    float actual,
    float expected,
    float tolerance = 1e-5f) {
    assert(std::fabs(actual - expected) <= tolerance);
}

}  // namespace

int main() {
    std::vector<std::uint8_t> blob =
        {'V', 'N', '9', '7', 'T', '2', 0, 0};
    PushU32(blob, 1);
    PushU32(blob, 2);
    PushU32(blob, 4);
    PushU32(blob, 2);
    PushU32(blob, 4);
    PushU32(blob, 2);
    PushU32(blob, 4);
    PushU32(blob, 2);
    PushF32(blob, 2.0f);
    PushF32(blob, 0.5f);
    blob.push_back(0x64);
    blob.push_back(0x92);

    vn97::PackedTernaryView view;
    assert(
        vn97::ParsePackedTernary(
            blob.data(),
            blob.size(),
            &view) ==
        vn97::PackedTernaryStatus::kOk);

    auto bad_scale = blob;
    const float nan =
        std::numeric_limits<float>::quiet_NaN();
    std::uint32_t nan_bits = 0;
    std::memcpy(&nan_bits, &nan, sizeof(nan_bits));
    for (int i = 0; i < 4; ++i) {
        bad_scale[40 + i] =
            static_cast<std::uint8_t>(
                (nan_bits >> (8 * i)) & 0xffu);
    }
    assert(
        vn97::ParsePackedTernary(
            bad_scale.data(),
            bad_scale.size(),
            &view) ==
        vn97::PackedTernaryStatus::kInvalidScale);

    bad_scale = blob;
    std::fill(
        bad_scale.begin() + 40,
        bad_scale.begin() + 44,
        0);
    assert(
        vn97::ParsePackedTernary(
            bad_scale.data(),
            bad_scale.size(),
            &view) ==
        vn97::PackedTernaryStatus::kInvalidScale);

    auto bad_tile = blob;
    SetU32(bad_tile, 20, 257u);
    assert(
        vn97::ParsePackedTernary(
            bad_tile.data(),
            bad_tile.size(),
            &view) ==
        vn97::PackedTernaryStatus::kInvalidGeometry);

    auto extra_padding = blob;
    SetU32(extra_padding, 28, 4u);
    assert(
        vn97::ParsePackedTernary(
            extra_padding.data(),
            extra_padding.size(),
            &view) ==
        vn97::PackedTernaryStatus::kInvalidGeometry);

    const float input[4] =
        {1.0f, 2.0f, 3.0f, 4.0f};
    float scalar_output[2] = {};
    float auto_output[2] = {};
    assert(
        vn97::PackedTernaryMatVecF32WithBackend(
            view,
            input,
            nullptr,
            scalar_output,
            vn97::PackedTernaryBackend::kScalar) ==
        vn97::PackedTernaryStatus::kOk);
    assert(
        vn97::PackedTernaryMatVecF32(
            view,
            input,
            nullptr,
            auto_output) ==
        vn97::PackedTernaryStatus::kOk);
    AssertNear(scalar_output[0], 6.0f);
    AssertNear(scalar_output[1], -1.0f);
    AssertNear(auto_output[0], scalar_output[0]);
    AssertNear(auto_output[1], scalar_output[1]);

    assert(
        std::strcmp(
            vn97::PackedTernaryBackendName(
                vn97::PackedTernaryBackend::kScalar),
            "scalar") == 0);
    assert(
        vn97::PackedTernaryBackendAvailable(
            vn97::PackedTernaryBackend::kScalar));

    const auto wide_blob =
        BuildTwoBySixteenBlob();
    vn97::PackedTernaryView wide_view;
    assert(
        vn97::ParsePackedTernary(
            wide_blob.data(),
            wide_blob.size(),
            &wide_view) ==
        vn97::PackedTernaryStatus::kOk);
    float wide_input[16] = {};
    for (int i = 0; i < 16; ++i) {
        wide_input[i] = static_cast<float>(i + 1);
    }
    const float bias[2] = {0.5f, -1.0f};
    float reference[2] = {};
    assert(
        vn97::PackedTernaryMatVecF32WithBackend(
            wide_view,
            wide_input,
            bias,
            reference,
            vn97::PackedTernaryBackend::kScalar) ==
        vn97::PackedTernaryStatus::kOk);

    vn97::PackedTernaryExecutionPlan wide_plan;
    assert(
        vn97::PlanPackedTernaryMatVecF32(
            wide_view,
            vn97::PackedTernaryBackend::kScalar,
            &wide_plan) ==
        vn97::PackedTernaryStatus::kOk);
    assert(wide_plan.row_block == 2u);
    assert(wide_plan.col_block == 16u);
    assert(wide_plan.accumulator_floats == 2u);
    assert(wide_plan.tile_count == 1u);
    assert(wide_plan.vector_width == 1u);

    auto short_view = wide_view;
    --short_view.packed_data_size;
    vn97::PackedTernaryExecutionPlan rejected_plan;
    assert(
        vn97::PlanPackedTernaryMatVecF32(
            short_view,
            vn97::PackedTernaryBackend::kScalar,
            &rejected_plan) ==
        vn97::PackedTernaryStatus::kInvalidLength);

    auto mismatched_plan = wide_plan;
    ++mismatched_plan.rows;
    float rejected_output[2] = {};
    assert(
        vn97::PackedTernaryMatVecF32WithPlan(
            wide_view,
            wide_input,
            bias,
            rejected_output,
            mismatched_plan) ==
        vn97::PackedTernaryStatus::kPlanMismatch);

    std::vector<float> tiled_scales;
    const auto tiled_blob =
        BuildTiledBlob(
            17,
            19,
            16,
            16,
            &tiled_scales);
    vn97::PackedTernaryView tiled_view;
    assert(
        vn97::ParsePackedTernary(
            tiled_blob.data(),
            tiled_blob.size(),
            &tiled_view) ==
        vn97::PackedTernaryStatus::kOk);

    vn97::PackedTernaryExecutionPlan tiled_plan;
    assert(
        vn97::PlanPackedTernaryMatVecF32(
            tiled_view,
            vn97::PackedTernaryBackend::kScalar,
            &tiled_plan) ==
        vn97::PackedTernaryStatus::kOk);
    assert(tiled_plan.row_block == 16u);
    assert(tiled_plan.col_block == 16u);
    assert(tiled_plan.accumulator_floats == 16u);
    assert(tiled_plan.tile_count == 4u);
    assert(tiled_plan.tile_working_set_bytes == 192u);

    std::vector<float> tiled_input(19);
    for (std::size_t col = 0;
         col < tiled_input.size();
         ++col) {
        tiled_input[col] =
            static_cast<float>(
                static_cast<int>(col % 7u) - 3) *
            0.25f;
    }
    std::vector<float> tiled_bias(17);
    for (std::size_t row = 0;
         row < tiled_bias.size();
         ++row) {
        tiled_bias[row] =
            static_cast<float>(row) * 0.01f - 0.07f;
    }

    std::vector<float> expected(17, 0.0f);
    for (std::uint32_t row = 0; row < 17u; ++row) {
        float sum = 0.0f;
        for (std::uint32_t col = 0; col < 19u; ++col) {
            sum +=
                static_cast<float>(DenseSymbol(row, col)) *
                tiled_input[col];
        }
        expected[row] =
            sum * tiled_scales[row] +
            tiled_bias[row];
    }

    std::vector<float> tiled_scalar(17, 0.0f);
    assert(
        vn97::PackedTernaryMatVecF32WithPlan(
            tiled_view,
            tiled_input.data(),
            tiled_bias.data(),
            tiled_scalar.data(),
            tiled_plan) ==
        vn97::PackedTernaryStatus::kOk);
    for (std::size_t row = 0; row < expected.size(); ++row) {
        AssertNear(tiled_scalar[row], expected[row]);
    }

    if (vn97::PackedTernaryBackendAvailable(
            vn97::PackedTernaryBackend::kArm64Neon)) {
        assert(
            vn97::ResolvePackedTernaryBackend(
                vn97::PackedTernaryBackend::kAuto) ==
            vn97::PackedTernaryBackend::kArm64Neon);

        float neon_output[2] = {};
        assert(
            vn97::PackedTernaryMatVecF32WithBackend(
                wide_view,
                wide_input,
                bias,
                neon_output,
                vn97::PackedTernaryBackend::kArm64Neon) ==
            vn97::PackedTernaryStatus::kOk);
        AssertNear(neon_output[0], reference[0]);
        AssertNear(neon_output[1], reference[1]);

        vn97::PackedTernaryExecutionPlan neon_plan;
        assert(
            vn97::PlanPackedTernaryMatVecF32(
                tiled_view,
                vn97::PackedTernaryBackend::kArm64Neon,
                &neon_plan) ==
            vn97::PackedTernaryStatus::kOk);
        assert(neon_plan.vector_width == 16u);

        std::vector<float> tiled_neon(17, 0.0f);
        assert(
            vn97::PackedTernaryMatVecF32WithPlan(
                tiled_view,
                tiled_input.data(),
                tiled_bias.data(),
                tiled_neon.data(),
                neon_plan) ==
            vn97::PackedTernaryStatus::kOk);
        for (std::size_t row = 0; row < expected.size(); ++row) {
            AssertNear(
                tiled_neon[row],
                tiled_scalar[row]);
        }
    } else {
        assert(
            vn97::ResolvePackedTernaryBackend(
                vn97::PackedTernaryBackend::kAuto) ==
            vn97::PackedTernaryBackend::kScalar);
        float unavailable_output[2] = {};
        assert(
            vn97::PackedTernaryMatVecF32WithBackend(
                wide_view,
                wide_input,
                bias,
                unavailable_output,
                vn97::PackedTernaryBackend::kArm64Neon) ==
            vn97::PackedTernaryStatus::kBackendUnavailable);
    }

    return 0;
}
