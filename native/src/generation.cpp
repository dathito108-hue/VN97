#include "vn97/generation.h"

#include "vn97/model_image.h"
#include "vn97/runtime.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>

namespace {
constexpr std::size_t kMaxGenerationWorkingBytes = 64ull * 1024ull * 1024ull;

struct GenerationCursor {
    std::uint64_t model_handle = 0;
    std::uint64_t runtime_handle = 0;
    vn97::GenerationConfig config;
    std::uint64_t expected_sequence = 0;
    std::vector<float> logits;
    std::vector<vn97::SamplerCandidate> sampler_workspace;
    bool ready = false;
    bool finished = false;
    std::mutex mutex;
};

std::mutex g_generation_registry_mutex;
std::unordered_map<std::uint64_t, std::shared_ptr<GenerationCursor>> g_generations;
std::uint64_t g_next_generation_handle = 1;

std::shared_ptr<GenerationCursor> LookupGeneration(std::uint64_t handle) {
    std::lock_guard<std::mutex> lock(g_generation_registry_mutex);
    const auto it = g_generations.find(handle);
    return it == g_generations.end() ? nullptr : it->second;
}

vn97::GenerationConfig ToCpp(const vn97_generation_config& config) {
    vn97::GenerationConfig cpp;
    cpp.sampler.temperature = config.sampler.temperature;
    cpp.sampler.top_k = config.sampler.top_k;
    cpp.sampler.top_p = config.sampler.top_p;
    cpp.sampler.seed = config.sampler.seed;
    cpp.eos_token = config.eos_token;
    return cpp;
}

bool ValidSamplerConfig(const vn97::SamplerConfig& config) {
    if (!(config.temperature >= 0.0f) ||
        !(config.top_p > 0.0f) ||
        !(config.top_p <= 1.0f)) {
        return false;
    }
    return std::isfinite(config.temperature) && std::isfinite(config.top_p);
}
}  // namespace

extern "C" int vn97_generation_open(
    std::uint64_t model_handle,
    std::uint64_t runtime_handle,
    const std::uint32_t* prompt_ids,
    std::size_t prompt_count,
    const vn97_generation_config* config,
    std::uint64_t* generation_handle_out) {
    if (prompt_ids == nullptr || config == nullptr || generation_handle_out == nullptr) {
        return static_cast<int>(vn97::GenerationStatus::kNullArgument);
    }
    *generation_handle_out = 0;
    if (model_handle == 0 || runtime_handle == 0 || prompt_count == 0) {
        return static_cast<int>(vn97::GenerationStatus::kInvalidConfig);
    }

    const vn97::GenerationConfig cpp_config = ToCpp(*config);
    if (!ValidSamplerConfig(cpp_config.sampler)) {
        return static_cast<int>(vn97::GenerationStatus::kInvalidConfig);
    }

    vn97_model_info model_info{};
    if (vn97_model_info_get(model_handle, &model_info) != 0 || model_info.vocab_size <= 1) {
        return static_cast<int>(vn97::GenerationStatus::kModelError);
    }
    constexpr std::size_t bytes_per_token =
        sizeof(float) + sizeof(vn97::SamplerCandidate);
    if (model_info.vocab_size >
        kMaxGenerationWorkingBytes / bytes_per_token) {
        return static_cast<int>(vn97::GenerationStatus::kInvalidConfig);
    }

    vn97_runtime_info runtime_info{};
    if (vn97_runtime_info_get(runtime_handle, &runtime_info) != 0) {
        return static_cast<int>(vn97::GenerationStatus::kRuntimeError);
    }
    if (runtime_info.batch != 1) {
        return static_cast<int>(vn97::GenerationStatus::kInvalidConfig);
    }

    std::shared_ptr<GenerationCursor> cursor;
    try {
        cursor = std::make_shared<GenerationCursor>();
        cursor->logits.assign(model_info.vocab_size, 0.0f);
        cursor->sampler_workspace.resize(model_info.vocab_size);
    } catch (...) {
        return static_cast<int>(vn97::GenerationStatus::kOutOfMemory);
    }
    cursor->model_handle = model_handle;
    cursor->runtime_handle = runtime_handle;
    cursor->config = cpp_config;

    std::uint64_t handle = 0;
    {
        std::lock_guard<std::mutex> lock(g_generation_registry_mutex);
        if (g_next_generation_handle == 0) {
            return static_cast<int>(vn97::GenerationStatus::kCounterOverflow);
        }
        handle = g_next_generation_handle++;
        try {
            g_generations.emplace(handle, cursor);
        } catch (...) {
            return static_cast<int>(vn97::GenerationStatus::kOutOfMemory);
        }
    }

    const int prefill_status = vn97_model_runtime_prefill(
        model_handle,
        runtime_handle,
        prompt_ids,
        prompt_count,
        prompt_count,
        cursor->logits.data(),
        cursor->logits.size());
    if (prefill_status != 0 ||
        vn97_runtime_info_get(runtime_handle, &runtime_info) != 0) {
        std::lock_guard<std::mutex> lock(g_generation_registry_mutex);
        g_generations.erase(handle);
        return static_cast<int>(vn97::GenerationStatus::kRuntimeError);
    }
    {
        std::lock_guard<std::mutex> cursor_lock(cursor->mutex);
        cursor->expected_sequence = runtime_info.sequence_position;
        cursor->ready = true;
    }
    *generation_handle_out = handle;
    return static_cast<int>(vn97::GenerationStatus::kOk);
}

extern "C" int vn97_generation_next(
    std::uint64_t generation_handle,
    std::uint32_t* token_out,
    int* eos_out) {
    if (token_out == nullptr || eos_out == nullptr) {
        return static_cast<int>(vn97::GenerationStatus::kNullArgument);
    }
    const auto cursor = LookupGeneration(generation_handle);
    if (!cursor) return static_cast<int>(vn97::GenerationStatus::kInvalidHandle);

    std::lock_guard<std::mutex> lock(cursor->mutex);
    if (!cursor->ready) return static_cast<int>(vn97::GenerationStatus::kInvalidHandle);
    if (cursor->finished) return static_cast<int>(vn97::GenerationStatus::kFinished);
    if (cursor->expected_sequence == std::numeric_limits<std::uint64_t>::max()) {
        return static_cast<int>(vn97::GenerationStatus::kCounterOverflow);
    }

    vn97_runtime_info runtime_info{};
    if (vn97_runtime_info_get(cursor->runtime_handle, &runtime_info) != 0) {
        return static_cast<int>(vn97::GenerationStatus::kRuntimeError);
    }
    if (runtime_info.sequence_position != cursor->expected_sequence) {
        return static_cast<int>(vn97::GenerationStatus::kSequenceMismatch);
    }

    std::uint32_t token = 0;
    const auto sample_status = vn97::SampleLogitsF32WithWorkspace(
        cursor->logits.data(),
        cursor->logits.size(),
        cursor->config.sampler,
        cursor->expected_sequence,
        cursor->sampler_workspace.data(),
        cursor->sampler_workspace.size(),
        &token);
    if (sample_status != vn97::SamplerStatus::kOk) {
        return static_cast<int>(vn97::GenerationStatus::kSamplerError);
    }

    const int infer_status = vn97_model_runtime_infer_step(
        cursor->model_handle,
        cursor->runtime_handle,
        &token,
        1,
        cursor->logits.data(),
        cursor->logits.size());
    if (infer_status != 0) {
        return static_cast<int>(vn97::GenerationStatus::kRuntimeError);
    }
    if (vn97_runtime_info_get(cursor->runtime_handle, &runtime_info) != 0) {
        return static_cast<int>(vn97::GenerationStatus::kRuntimeError);
    }
    if (runtime_info.sequence_position != cursor->expected_sequence + 1) {
        return static_cast<int>(vn97::GenerationStatus::kSequenceMismatch);
    }
    cursor->expected_sequence = runtime_info.sequence_position;

    const bool eos =
        cursor->config.eos_token != std::numeric_limits<std::uint32_t>::max() &&
        token == cursor->config.eos_token;
    cursor->finished = eos;
    *token_out = token;
    *eos_out = eos ? 1 : 0;
    return static_cast<int>(vn97::GenerationStatus::kOk);
}

extern "C" int vn97_generation_destroy(std::uint64_t generation_handle) {
    std::lock_guard<std::mutex> lock(g_generation_registry_mutex);
    return g_generations.erase(generation_handle) == 1
        ? static_cast<int>(vn97::GenerationStatus::kOk)
        : static_cast<int>(vn97::GenerationStatus::kInvalidHandle);
}
