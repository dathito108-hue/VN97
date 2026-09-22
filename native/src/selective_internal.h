#pragma once

#include "vn97/selective.h"

#include <cstddef>

namespace vn97::internal {

RecurrentStatus FusedSelectivePrefillF32Scalar(
    const float* signal,
    const float* dt_logits,
    const float* input_b,
    const float* readout_c,
    const float* gate,
    const float* a,
    float* state_io,
    float* output,
    std::size_t batch,
    std::size_t sequence,
    std::size_t d_model,
    std::size_t d_state,
    float dt_min,
    float dt_max);

#if defined(VN97_HAS_ARM64_NEON)
RecurrentStatus FusedSelectivePrefillF32Arm64Neon(
    const float* signal,
    const float* dt_logits,
    const float* input_b,
    const float* readout_c,
    const float* gate,
    const float* a,
    float* state_io,
    float* output,
    std::size_t batch,
    std::size_t sequence,
    std::size_t d_model,
    std::size_t d_state,
    float dt_min,
    float dt_max);
#endif

}  // namespace vn97::internal
