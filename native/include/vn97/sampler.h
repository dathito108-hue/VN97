#pragma once

#include <cstddef>
#include <cstdint>

namespace vn97 {

enum class SamplerStatus {
    kOk = 0,
    kNullArgument,
    kInvalidConfig,
    kEmptyLogits,
    kNonFinite,
    kOutOfMemory,
    kWorkspaceTooSmall,
};

struct SamplerConfig {
    float temperature = 0.8f;
    std::uint32_t top_k = 40;
    float top_p = 0.95f;
    std::uint64_t seed = 0;
};

struct SamplerCandidate {
    float logit = 0.0f;
    std::uint32_t token = 0;
    double weight = 0.0;
};

SamplerStatus SampleLogitsF32WithWorkspace(
    const float* logits,
    std::size_t logits_count,
    const SamplerConfig& config,
    std::uint64_t step,
    SamplerCandidate* workspace,
    std::size_t workspace_count,
    std::uint32_t* token_out);

SamplerStatus SampleLogitsF32(
    const float* logits,
    std::size_t logits_count,
    const SamplerConfig& config,
    std::uint64_t step,
    std::uint32_t* token_out);

}  // namespace vn97

extern "C" {

struct vn97_sampler_config {
    float temperature;
    std::uint32_t top_k;
    float top_p;
    std::uint64_t seed;
};

int vn97_sample_logits_f32(
    const float* logits,
    std::size_t logits_count,
    const vn97_sampler_config* config,
    std::uint64_t step,
    std::uint32_t* token_out);

}
