#include "vn97/language.h"

#include "vn97/selective.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace vn97 {
namespace {

bool MulOverflow(std::size_t a, std::size_t b, std::size_t* out) {
    if (out == nullptr) return true;
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) return true;
    *out = a * b;
    return false;
}

bool AddOverflow(std::size_t a, std::size_t b, std::size_t* out) {
    if (out == nullptr) return true;
    if (b > std::numeric_limits<std::size_t>::max() - a) return true;
    *out = a + b;
    return false;
}

bool ModelIdPresent(const LanguageModelView& model) {
    for (std::uint8_t value : model.model_id) {
        if (value != 0) return true;
    }
    return false;
}

bool MatrixShape(
    const PackedTernaryView& matrix,
    std::uint32_t rows,
    std::uint32_t cols) {
    return matrix.rows == rows && matrix.cols == cols &&
           matrix.rows != 0 && matrix.cols != 0 &&
           matrix.scale_bytes != nullptr && matrix.packed_data != nullptr &&
           matrix.tile_rows != 0 && matrix.tile_cols != 0 &&
           matrix.padded_rows >= matrix.rows &&
           matrix.padded_cols >= matrix.cols &&
           matrix.padded_rows % matrix.tile_rows == 0 &&
           matrix.padded_cols % matrix.tile_cols == 0;
}

bool FiniteArray(const float* values, std::size_t count) {
    if (values == nullptr) return false;
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i])) return false;
    }
    return true;
}

bool ValidStructure(const LanguageModelView& model) {
    if (!ModelIdPresent(model) || model.vocab_size <= 1 ||
        model.d_model == 0 || model.n_layers == 0 || model.d_state == 0 ||
        !std::isfinite(model.dt_min) || !std::isfinite(model.dt_max) ||
        !std::isfinite(model.rms_eps) || model.dt_min <= 0.0f ||
        model.dt_max < model.dt_min || model.rms_eps <= 0.0f ||
        model.layers == nullptr || model.final_norm_weight == nullptr) {
        return false;
    }

    if (model.embedding_kind == LanguageEmbeddingKind::kFullF32) {
        if (model.embedding == nullptr || model.embedding_rank != 0 ||
            model.token_factors != nullptr || model.projection != nullptr) {
            return false;
        }
    } else if (model.embedding_kind == LanguageEmbeddingKind::kFactorizedF32) {
        if (model.embedding != nullptr || model.token_factors == nullptr ||
            model.projection == nullptr || model.embedding_rank == 0 ||
            model.embedding_rank >= model.d_model) {
            return false;
        }
    } else {
        return false;
    }

    for (std::uint32_t i = 0; i < model.n_layers; ++i) {
        const auto& layer = model.layers[i];
        if (layer.norm_weight == nullptr || layer.dt_bias == nullptr ||
            layer.a == nullptr ||
            !MatrixShape(layer.in_proj, model.d_model * 2u, model.d_model) ||
            !MatrixShape(layer.dt_proj, model.d_model, model.d_model) ||
            !MatrixShape(layer.b_proj, model.d_state, model.d_model) ||
            !MatrixShape(layer.c_proj, model.d_state, model.d_model) ||
            !MatrixShape(layer.out_proj, model.d_model, model.d_model)) {
            return false;
        }
    }
    return true;
}

LanguageStatus MatVec(
    const PackedTernaryView& matrix,
    const float* input,
    const float* bias,
    float* output,
    PackedTernaryBackend backend) {
    const auto status = PackedTernaryMatVecF32WithBackend(
        matrix, input, bias, output, backend);
    if (status == PackedTernaryStatus::kOk) return LanguageStatus::kOk;
    if (status == PackedTernaryStatus::kBackendUnavailable) {
        return LanguageStatus::kBackendUnavailable;
    }
    return LanguageStatus::kInvalidModel;
}

void RmsNorm(
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
        1.0f / std::sqrt(sum / static_cast<float>(count) + eps);
    for (std::size_t i = 0; i < count; ++i) {
        output[i] = weight[i] * input[i] * inv;
    }
}

LanguageStatus EmbedToken(
    const LanguageModelView& model,
    std::uint32_t token,
    float* out) {
    if (token >= model.vocab_size) return LanguageStatus::kInvalidToken;

    if (model.embedding_kind == LanguageEmbeddingKind::kFullF32) {
        std::copy(
            model.embedding +
                static_cast<std::size_t>(token) * model.d_model,
            model.embedding +
                (static_cast<std::size_t>(token) + 1) * model.d_model,
            out);
        return LanguageStatus::kOk;
    }

    const float* factors =
        model.token_factors +
        static_cast<std::size_t>(token) * model.embedding_rank;
    for (std::uint32_t m = 0; m < model.d_model; ++m) {
        float sum = 0.0f;
        for (std::uint32_t r = 0; r < model.embedding_rank; ++r) {
            sum += factors[r] *
                model.projection[
                    static_cast<std::size_t>(r) * model.d_model + m];
        }
        out[m] = sum;
    }
    return LanguageStatus::kOk;
}

void ProjectLogits(
    const LanguageModelView& model,
    const float* hidden,
    float* logits,
    float* reduced) {
    if (model.embedding_kind == LanguageEmbeddingKind::kFullF32) {
        for (std::uint32_t v = 0; v < model.vocab_size; ++v) {
            const float* row =
                model.embedding +
                static_cast<std::size_t>(v) * model.d_model;
            float sum = 0.0f;
            for (std::uint32_t m = 0; m < model.d_model; ++m) {
                sum += hidden[m] * row[m];
            }
            logits[v] = sum;
        }
        return;
    }

    for (std::uint32_t r = 0; r < model.embedding_rank; ++r) {
        const float* row =
            model.projection +
            static_cast<std::size_t>(r) * model.d_model;
        float sum = 0.0f;
        for (std::uint32_t m = 0; m < model.d_model; ++m) {
            sum += hidden[m] * row[m];
        }
        reduced[r] = sum;
    }

    for (std::uint32_t v = 0; v < model.vocab_size; ++v) {
        const float* factors =
            model.token_factors +
            static_cast<std::size_t>(v) * model.embedding_rank;
        float sum = 0.0f;
        for (std::uint32_t r = 0; r < model.embedding_rank; ++r) {
            sum += reduced[r] * factors[r];
        }
        logits[v] = sum;
    }
}

}  // namespace

LanguageStatus ValidateLanguageModel(const LanguageModelView& model) {
    if (!ValidStructure(model)) return LanguageStatus::kInvalidModel;

    std::size_t count = 0;
    if (model.embedding_kind == LanguageEmbeddingKind::kFullF32) {
        if (MulOverflow(model.vocab_size, model.d_model, &count)) {
            return LanguageStatus::kSizeOverflow;
        }
        if (!FiniteArray(model.embedding, count)) {
            return LanguageStatus::kNonFinite;
        }
    } else {
        if (MulOverflow(model.vocab_size, model.embedding_rank, &count)) {
            return LanguageStatus::kSizeOverflow;
        }
        if (!FiniteArray(model.token_factors, count)) {
            return LanguageStatus::kNonFinite;
        }
        if (MulOverflow(model.embedding_rank, model.d_model, &count)) {
            return LanguageStatus::kSizeOverflow;
        }
        if (!FiniteArray(model.projection, count)) {
            return LanguageStatus::kNonFinite;
        }
    }

    if (!FiniteArray(model.final_norm_weight, model.d_model)) {
        return LanguageStatus::kNonFinite;
    }

    if (MulOverflow(model.d_model, model.d_state, &count)) {
        return LanguageStatus::kSizeOverflow;
    }

    for (std::uint32_t i = 0; i < model.n_layers; ++i) {
        const auto& layer = model.layers[i];
        if (!FiniteArray(layer.norm_weight, model.d_model) ||
            !FiniteArray(layer.dt_bias, model.d_model) ||
            !FiniteArray(layer.a, count)) {
            return LanguageStatus::kNonFinite;
        }
        for (std::size_t j = 0; j < count; ++j) {
            if (!(layer.a[j] < 0.0f)) {
                return LanguageStatus::kInvalidModel;
            }
        }
    }
    return LanguageStatus::kOk;
}

LanguageStatus LanguageWorkspaceFloats(
    const LanguageModelView& model,
    std::size_t batch,
    std::size_t* out) {
    if (out == nullptr) return LanguageStatus::kNullArgument;
    *out = 0;
    if (!ValidStructure(model) || batch == 0) {
        return LanguageStatus::kInvalidConfig;
    }

    std::size_t batch_x = 0;
    if (MulOverflow(batch, model.d_model, &batch_x)) {
        return LanguageStatus::kSizeOverflow;
    }

    const std::size_t scratch =
        static_cast<std::size_t>(6) * model.d_model +
        static_cast<std::size_t>(2) * model.d_state +
        model.embedding_rank;

    if (AddOverflow(batch_x, scratch, out)) {
        return LanguageStatus::kSizeOverflow;
    }
    return LanguageStatus::kOk;
}

LanguageStatus LanguageStepF32WithBackends(
    const LanguageModelView& model,
    const std::uint32_t* input_ids,
    std::size_t batch,
    float* state_io,
    float* logits,
    std::size_t logits_count,
    float* workspace,
    std::size_t workspace_count,
    RecurrentBackend recurrent_backend,
    PackedTernaryBackend packed_backend) {
    if (input_ids == nullptr || state_io == nullptr ||
        logits == nullptr || workspace == nullptr) {
        return LanguageStatus::kNullArgument;
    }
    if (!ValidStructure(model) || batch == 0) {
        return LanguageStatus::kInvalidConfig;
    }

    const auto rr = ResolveRecurrentBackend(recurrent_backend);
    const auto rp = ResolvePackedTernaryBackend(packed_backend);
    if (!RecurrentBackendAvailable(rr) ||
        !PackedTernaryBackendAvailable(rp)) {
        return LanguageStatus::kBackendUnavailable;
    }

    std::size_t required_logits = 0;
    if (MulOverflow(batch, model.vocab_size, &required_logits)) {
        return LanguageStatus::kSizeOverflow;
    }
    if (logits_count < required_logits) {
        return LanguageStatus::kOutputTooSmall;
    }

    std::size_t required_workspace = 0;
    auto status =
        LanguageWorkspaceFloats(model, batch, &required_workspace);
    if (status != LanguageStatus::kOk) return status;
    if (workspace_count < required_workspace) {
        return LanguageStatus::kOutputTooSmall;
    }

    for (std::size_t b = 0; b < batch; ++b) {
        if (input_ids[b] >= model.vocab_size) {
            return LanguageStatus::kInvalidToken;
        }
    }

    float* x = workspace;
    float* scratch = x + batch * model.d_model;
    float* norm = scratch;
    float* in_proj = norm + model.d_model;
    float* dt =
        in_proj + static_cast<std::size_t>(2) * model.d_model;
    float* b_vec = dt + model.d_model;
    float* c_vec = b_vec + model.d_state;
    float* selective = c_vec + model.d_state;
    float* out_proj = selective + model.d_model;
    float* reduced = out_proj + model.d_model;

    for (std::size_t b = 0; b < batch; ++b) {
        status = EmbedToken(
            model, input_ids[b], x + b * model.d_model);
        if (status != LanguageStatus::kOk) return status;
    }

    const std::size_t layer_state_stride =
        batch * static_cast<std::size_t>(model.d_model) *
        model.d_state;
    const std::size_t batch_state_stride =
        static_cast<std::size_t>(model.d_model) * model.d_state;

    for (std::uint32_t layer_index = 0;
         layer_index < model.n_layers;
         ++layer_index) {
        const auto& layer = model.layers[layer_index];

        for (std::size_t batch_index = 0;
             batch_index < batch;
             ++batch_index) {
            float* x_row =
                x + batch_index * model.d_model;

            RmsNorm(
                x_row,
                layer.norm_weight,
                model.d_model,
                model.rms_eps,
                norm);

            status = MatVec(
                layer.in_proj,
                norm,
                nullptr,
                in_proj,
                rp);
            if (status != LanguageStatus::kOk) return status;

            const float* signal = in_proj;
            const float* gate = in_proj + model.d_model;

            status = MatVec(
                layer.dt_proj,
                signal,
                layer.dt_bias,
                dt,
                rp);
            if (status != LanguageStatus::kOk) return status;

            status = MatVec(
                layer.b_proj,
                signal,
                nullptr,
                b_vec,
                rp);
            if (status != LanguageStatus::kOk) return status;

            status = MatVec(
                layer.c_proj,
                signal,
                nullptr,
                c_vec,
                rp);
            if (status != LanguageStatus::kOk) return status;

            float* state =
                state_io +
                static_cast<std::size_t>(layer_index) *
                    layer_state_stride +
                batch_index * batch_state_stride;

            const auto recurrent_status =
                FusedSelectiveStepF32WithBackend(
                    signal,
                    dt,
                    b_vec,
                    c_vec,
                    gate,
                    layer.a,
                    state,
                    selective,
                    1,
                    model.d_model,
                    model.d_state,
                    model.dt_min,
                    model.dt_max,
                    rr);

            if (recurrent_status ==
                RecurrentStatus::kBackendUnavailable) {
                return LanguageStatus::kBackendUnavailable;
            }
            if (recurrent_status != RecurrentStatus::kOk) {
                return LanguageStatus::kInvalidModel;
            }

            status = MatVec(
                layer.out_proj,
                selective,
                nullptr,
                out_proj,
                rp);
            if (status != LanguageStatus::kOk) return status;

            for (std::uint32_t m = 0;
                 m < model.d_model;
                 ++m) {
                x_row[m] += out_proj[m];
            }
        }
    }

    for (std::size_t batch_index = 0;
         batch_index < batch;
         ++batch_index) {
        float* x_row =
            x + batch_index * model.d_model;

        RmsNorm(
            x_row,
            model.final_norm_weight,
            model.d_model,
            model.rms_eps,
            norm);

        ProjectLogits(
            model,
            norm,
            logits + batch_index * model.vocab_size,
            reduced);
    }

    return LanguageStatus::kOk;
}

LanguageStatus LanguageStepF32(
    const LanguageModelView& model,
    const std::uint32_t* input_ids,
    std::size_t batch,
    float* state_io,
    float* logits,
    std::size_t logits_count,
    float* workspace,
    std::size_t workspace_count) {
    return LanguageStepF32WithBackends(
        model,
        input_ids,
        batch,
        state_io,
        logits,
        logits_count,
        workspace,
        workspace_count,
        RecurrentBackend::kAuto,
        PackedTernaryBackend::kAuto);
}

}  // namespace vn97
