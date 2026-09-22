#pragma once

#include "vn97/recurrent.h"

#include <cstddef>

namespace vn97::internal {

RecurrentStatus FusedRecurrentPrefillF32Scalar(
    const float* decay,
    const float* drive,
    const float* readout,
    const float* gate,
    float* state_io,
    float* output,
    std::size_t batch,
    std::size_t sequence,
    std::size_t d_model,
    std::size_t d_state);

#if defined(VN97_HAS_ARM64_NEON)
RecurrentStatus FusedRecurrentPrefillF32Arm64Neon(
    const float* decay,
    const float* drive,
    const float* readout,
    const float* gate,
    float* state_io,
    float* output,
    std::size_t batch,
    std::size_t sequence,
    std::size_t d_model,
    std::size_t d_state);
#endif

}  // namespace vn97::internal
