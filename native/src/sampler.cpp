#include "vn97/sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace vn97 {
namespace {

std::uint64_t SplitMix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

double Uniform01(std::uint64_t seed, std::uint64_t step) {
    const std::uint64_t mixed = SplitMix64(seed ^ SplitMix64(step));
    return static_cast<double>(mixed >> 11) * 0x1.0p-53;
}

bool ValidConfig(const SamplerConfig& config) {
    return std::isfinite(config.temperature) &&
           config.temperature >= 0.0f &&
           std::isfinite(config.top_p) &&
           config.top_p > 0.0f &&
           config.top_p <= 1.0f;
}

SamplerStatus ValidateAndGreedy(
    const float* logits,
    std::size_t logits_count,
    const SamplerConfig& config,
    std::uint32_t* greedy_token,
    float* greedy_logit) {
    if (logits == nullptr || greedy_token == nullptr || greedy_logit == nullptr) {
        return SamplerStatus::kNullArgument;
    }
    if (logits_count == 0 || logits_count > std::numeric_limits<std::uint32_t>::max()) {
        return SamplerStatus::kEmptyLogits;
    }
    if (!ValidConfig(config)) return SamplerStatus::kInvalidConfig;

    *greedy_token = 0;
    *greedy_logit = logits[0];
    if (!std::isfinite(*greedy_logit)) return SamplerStatus::kNonFinite;
    for (std::size_t i = 1; i < logits_count; ++i) {
        const float value = logits[i];
        if (!std::isfinite(value)) return SamplerStatus::kNonFinite;
        if (value > *greedy_logit) {
            *greedy_logit = value;
            *greedy_token = static_cast<std::uint32_t>(i);
        }
    }
    return SamplerStatus::kOk;
}

}  // namespace

SamplerStatus SampleLogitsF32WithWorkspace(
    const float* logits,
    std::size_t logits_count,
    const SamplerConfig& config,
    std::uint64_t step,
    SamplerCandidate* workspace,
    std::size_t workspace_count,
    std::uint32_t* token_out) {
    if (token_out == nullptr) return SamplerStatus::kNullArgument;
    std::uint32_t greedy_token = 0;
    float greedy_logit = 0.0f;
    const auto validation = ValidateAndGreedy(
        logits, logits_count, config, &greedy_token, &greedy_logit);
    if (validation != SamplerStatus::kOk) return validation;
    if (config.temperature == 0.0f) {
        *token_out = greedy_token;
        return SamplerStatus::kOk;
    }
    if (workspace == nullptr) return SamplerStatus::kNullArgument;
    if (workspace_count < logits_count) return SamplerStatus::kWorkspaceTooSmall;

    for (std::size_t i = 0; i < logits_count; ++i) {
        workspace[i].logit = logits[i];
        workspace[i].token = static_cast<std::uint32_t>(i);
        workspace[i].weight = 0.0;
    }
    std::sort(
        workspace,
        workspace + logits_count,
        [](const SamplerCandidate& a, const SamplerCandidate& b) {
            if (a.logit != b.logit) return a.logit > b.logit;
            return a.token < b.token;
        });

    std::size_t kept = logits_count;
    if (config.top_k != 0) {
        kept = std::min<std::size_t>(kept, config.top_k);
    }

    const double max_logit = static_cast<double>(workspace[0].logit);
    double total = 0.0;
    for (std::size_t i = 0; i < kept; ++i) {
        const double exponent =
            (static_cast<double>(workspace[i].logit) - max_logit) /
            static_cast<double>(config.temperature);
        workspace[i].weight = std::exp(exponent);
        if (!std::isfinite(workspace[i].weight) || workspace[i].weight < 0.0) {
            return SamplerStatus::kNonFinite;
        }
        total += workspace[i].weight;
    }
    if (!std::isfinite(total) || !(total > 0.0)) return SamplerStatus::kNonFinite;

    if (config.top_p < 1.0f) {
        const double cutoff = static_cast<double>(config.top_p) * total;
        double cumulative = 0.0;
        std::size_t nucleus = 0;
        while (nucleus < kept) {
            cumulative += workspace[nucleus].weight;
            ++nucleus;
            if (cumulative >= cutoff) break;
        }
        kept = std::max<std::size_t>(1, nucleus);
        total = 0.0;
        for (std::size_t i = 0; i < kept; ++i) total += workspace[i].weight;
    }

    const double target = Uniform01(config.seed, step) * total;
    double cumulative = 0.0;
    for (std::size_t i = 0; i < kept; ++i) {
        cumulative += workspace[i].weight;
        if (target < cumulative) {
            *token_out = workspace[i].token;
            return SamplerStatus::kOk;
        }
    }
    *token_out = workspace[kept - 1].token;
    return SamplerStatus::kOk;
}

SamplerStatus SampleLogitsF32(
    const float* logits,
    std::size_t logits_count,
    const SamplerConfig& config,
    std::uint64_t step,
    std::uint32_t* token_out) {
    if (logits == nullptr || token_out == nullptr) return SamplerStatus::kNullArgument;
    if (logits_count == 0 || logits_count > std::numeric_limits<std::uint32_t>::max()) {
        return SamplerStatus::kEmptyLogits;
    }
    if (!ValidConfig(config)) return SamplerStatus::kInvalidConfig;
    if (config.temperature == 0.0f) {
        return SampleLogitsF32WithWorkspace(
            logits, logits_count, config, step, nullptr, 0, token_out);
    }
    std::vector<SamplerCandidate> workspace;
    try {
        workspace.resize(logits_count);
    } catch (...) {
        return SamplerStatus::kOutOfMemory;
    }
    return SampleLogitsF32WithWorkspace(
        logits,
        logits_count,
        config,
        step,
        workspace.data(),
        workspace.size(),
        token_out);
}

}  // namespace vn97

extern "C" int vn97_sample_logits_f32(
    const float* logits,
    std::size_t logits_count,
    const vn97_sampler_config* config,
    std::uint64_t step,
    std::uint32_t* token_out) {
    if (config == nullptr) return static_cast<int>(vn97::SamplerStatus::kNullArgument);
    vn97::SamplerConfig cpp;
    cpp.temperature = config->temperature;
    cpp.top_k = config->top_k;
    cpp.top_p = config->top_p;
    cpp.seed = config->seed;
    return static_cast<int>(vn97::SampleLogitsF32(logits, logits_count, cpp, step, token_out));
}
