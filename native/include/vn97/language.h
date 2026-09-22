#pragma once

#include "vn97/packed_ternary.h"
#include "vn97/recurrent.h"

#include <cstddef>
#include <cstdint>

namespace vn97 {

enum class LanguageStatus {
    kOk = 0,
    kNullArgument,
    kInvalidConfig,
    kInvalidModel,
    kInvalidToken,
    kOutputTooSmall,
    kSizeOverflow,
    kBackendUnavailable,
    kNonFinite,
};

enum class LanguageEmbeddingKind : std::uint32_t {
    kFullF32 = 0,
    kFactorizedF32 = 1,
};

struct LanguageLayerView {
    const float* norm_weight = nullptr;
    PackedTernaryView in_proj;
    PackedTernaryView dt_proj;
    const float* dt_bias = nullptr;
    PackedTernaryView b_proj;
    PackedTernaryView c_proj;
    PackedTernaryView out_proj;
    const float* a = nullptr;
};

struct LanguageModelView {
    std::uint8_t model_id[32] = {};
    std::uint32_t vocab_size = 0;
    std::uint32_t d_model = 0;
    std::uint32_t n_layers = 0;
    std::uint32_t d_state = 0;
    LanguageEmbeddingKind embedding_kind = LanguageEmbeddingKind::kFullF32;
    std::uint32_t embedding_rank = 0;
    float dt_min = 0.0f;
    float dt_max = 0.0f;
    float rms_eps = 0.0f;
    const float* embedding = nullptr;
    const float* token_factors = nullptr;
    const float* projection = nullptr;
    const LanguageLayerView* layers = nullptr;
    const float* final_norm_weight = nullptr;
};

LanguageStatus ValidateLanguageModel(const LanguageModelView& model);

LanguageStatus LanguageWorkspaceFloats(
    const LanguageModelView& model,
    std::size_t batch,
    std::size_t* out);

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
    PackedTernaryBackend packed_backend);

LanguageStatus LanguageStepF32(
    const LanguageModelView& model,
    const std::uint32_t* input_ids,
    std::size_t batch,
    float* state_io,
    float* logits,
    std::size_t logits_count,
    float* workspace,
    std::size_t workspace_count);

}  // namespace vn97
