#pragma once

#include "vn97/sampler.h"

#include <cstddef>
#include <cstdint>

namespace vn97 {

enum class GenerationStatus {
    kOk = 0,
    kNullArgument,
    kInvalidConfig,
    kInvalidHandle,
    kModelError,
    kRuntimeError,
    kSamplerError,
    kSequenceMismatch,
    kCounterOverflow,
    kOutOfMemory,
    kFinished,
};

struct GenerationConfig {
    SamplerConfig sampler;
    std::uint32_t eos_token = 2;
};

}  // namespace vn97

extern "C" {

struct vn97_generation_config {
    vn97_sampler_config sampler;
    std::uint32_t eos_token;
};

int vn97_generation_open(
    std::uint64_t model_handle,
    std::uint64_t runtime_handle,
    const std::uint32_t* prompt_ids,
    std::size_t prompt_count,
    const vn97_generation_config* config,
    std::uint64_t* generation_handle_out);

int vn97_generation_next(
    std::uint64_t generation_handle,
    std::uint32_t* token_out,
    int* eos_out);

int vn97_generation_destroy(std::uint64_t generation_handle);

}
