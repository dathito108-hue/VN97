#pragma once

#include "vn97/language.h"
#include "vn97/packed_ternary.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace vn97_test {

inline void WriteU32(std::uint8_t* out, std::uint32_t value) {
    for (int i = 0; i < 4; ++i) {
        out[i] =
            static_cast<std::uint8_t>((value >> (8 * i)) & 0xffu);
    }
}

inline void WriteF32(std::uint8_t* out, float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    WriteU32(out, bits);
}

struct PackedFixture {
    std::vector<std::uint8_t> blob;
    vn97::PackedTernaryView view;
};

inline PackedFixture PackDenseTernary(
    std::uint32_t rows,
    std::uint32_t cols,
    const std::vector<float>& dense) {
    assert(dense.size() ==
           static_cast<std::size_t>(rows) * cols);

    constexpr std::size_t header = 40;
    const std::size_t symbol_count =
        static_cast<std::size_t>(rows) * cols;
    const std::size_t packed_bytes =
        (symbol_count + 3) / 4;

    PackedFixture result;
    result.blob.assign(
        header +
            static_cast<std::size_t>(rows) * 4 +
            packed_bytes,
        0);

    const std::uint8_t magic[8] =
        {'V','N','9','7','T','2',0,0};
    std::memcpy(result.blob.data(), magic, 8);
    WriteU32(result.blob.data() + 8, 1);
    WriteU32(result.blob.data() + 12, rows);
    WriteU32(result.blob.data() + 16, cols);
    WriteU32(result.blob.data() + 20, 1);
    WriteU32(result.blob.data() + 24, 1);
    WriteU32(result.blob.data() + 28, rows);
    WriteU32(result.blob.data() + 32, cols);
    WriteU32(
        result.blob.data() + 36,
        static_cast<std::uint32_t>(packed_bytes));

    std::vector<float> scales(rows, 1.0f);
    for (std::uint32_t row = 0; row < rows; ++row) {
        float scale = 0.0f;
        for (std::uint32_t col = 0; col < cols; ++col) {
            scale = std::max(
                scale,
                std::fabs(
                    dense[
                        static_cast<std::size_t>(row) *
                            cols +
                        col]));
        }
        if (scale == 0.0f) scale = 1.0f;
        scales[row] = scale;
        WriteF32(
            result.blob.data() + header +
                static_cast<std::size_t>(row) * 4,
            scale);
    }

    std::uint8_t* packed =
        result.blob.data() + header +
        static_cast<std::size_t>(rows) * 4;

    for (std::size_t index = 0;
         index < symbol_count;
         ++index) {
        const std::uint32_t row =
            static_cast<std::uint32_t>(
                index / cols);
        const float value = dense[index];
        const float scale = scales[row];
        std::uint8_t code = 0;
        if (value == scale) {
            code = 1;
        } else if (value == -scale) {
            code = 2;
        } else {
            assert(value == 0.0f);
        }
        packed[index / 4] |=
            static_cast<std::uint8_t>(
                code << (2 * (index % 4)));
    }

    assert(
        vn97::ParsePackedTernary(
            result.blob.data(),
            result.blob.size(),
            &result.view) ==
        vn97::PackedTernaryStatus::kOk);
    return result;
}

struct TinyLanguageFixture {
    static constexpr std::uint32_t kVocab = 5;
    static constexpr std::uint32_t kModel = 3;
    static constexpr std::uint32_t kState = 2;
    static constexpr std::uint32_t kLayers = 1;
    static constexpr std::uint32_t kRank = 2;

    std::vector<float> embedding = {
        0.2f, -0.1f, 0.4f,
        -0.3f, 0.5f, 0.2f,
        0.1f, 0.2f, -0.2f,
        0.7f, -0.4f, 0.1f,
        -0.2f, -0.3f, 0.6f,
    };
    std::vector<float> norm_weight =
        {1.1f, 0.9f, 1.2f};
    std::vector<float> final_norm_weight =
        {0.8f, 1.05f, 0.95f};
    std::vector<float> dt_bias =
        {0.05f, -0.03f, 0.02f};
    std::vector<float> a = {
        -0.5f, -1.2f,
        -0.7f, -1.4f,
        -0.9f, -1.6f,
    };

    std::vector<float> in_dense = {
        1,0,-1,
        0,1,1,
        -1,1,0,
        1,1,0,
        0,-1,1,
        1,0,1,
    };
    std::vector<float> dt_dense = {
        1,0,-1,
        0,1,1,
        -1,0,1,
    };
    std::vector<float> b_dense = {
        1,-1,0,
        0,1,1,
    };
    std::vector<float> c_dense = {
        0,1,1,
        -1,1,0,
    };
    std::vector<float> out_dense = {
        1,-1,0,
        0,1,-1,
        -1,0,1,
    };

    PackedFixture in_proj =
        PackDenseTernary(
            kModel * 2,
            kModel,
            in_dense);
    PackedFixture dt_proj =
        PackDenseTernary(
            kModel,
            kModel,
            dt_dense);
    PackedFixture b_proj =
        PackDenseTernary(
            kState,
            kModel,
            b_dense);
    PackedFixture c_proj =
        PackDenseTernary(
            kState,
            kModel,
            c_dense);
    PackedFixture out_proj =
        PackDenseTernary(
            kModel,
            kModel,
            out_dense);

    vn97::LanguageLayerView layer;
    vn97::LanguageModelView model;

    TinyLanguageFixture() {
        layer.norm_weight = norm_weight.data();
        layer.in_proj = in_proj.view;
        layer.dt_proj = dt_proj.view;
        layer.dt_bias = dt_bias.data();
        layer.b_proj = b_proj.view;
        layer.c_proj = c_proj.view;
        layer.out_proj = out_proj.view;
        layer.a = a.data();

        model.model_id[0] = 0x97;
        model.model_id[31] = 0x01;
        model.vocab_size = kVocab;
        model.d_model = kModel;
        model.n_layers = kLayers;
        model.d_state = kState;
        model.embedding_kind =
            vn97::LanguageEmbeddingKind::kFullF32;
        model.dt_min = 1e-4f;
        model.dt_max = 1.0f;
        model.rms_eps = 1e-5f;
        model.embedding = embedding.data();
        model.layers = &layer;
        model.final_norm_weight =
            final_norm_weight.data();
    }
};

inline void DenseMatVec(
    const std::vector<float>& matrix,
    std::size_t rows,
    std::size_t cols,
    const float* input,
    const float* bias,
    float* output) {
    for (std::size_t row = 0; row < rows; ++row) {
        float sum = bias != nullptr ? bias[row] : 0.0f;
        for (std::size_t col = 0; col < cols; ++col) {
            sum +=
                matrix[row * cols + col] *
                input[col];
        }
        output[row] = sum;
    }
}

inline void DenseRmsNorm(
    const float* input,
    const float* weight,
    std::size_t count,
    float eps,
    float* output) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < count; ++i) {
        sum += input[i] * input[i];
    }
    const float inv =
        1.0f /
        std::sqrt(
            sum / static_cast<float>(count) +
            eps);
    for (std::size_t i = 0; i < count; ++i) {
        output[i] =
            input[i] * weight[i] * inv;
    }
}

}  // namespace vn97_test
