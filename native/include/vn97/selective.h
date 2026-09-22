#pragma once

#include "vn97/recurrent.h"

#include <cstddef>

namespace vn97 {

RecurrentStatus FusedSelectivePrefillF32WithBackend(
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
    float dt_max,
    RecurrentBackend backend);

RecurrentStatus FusedSelectivePrefillF32(
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

RecurrentStatus FusedSelectiveStepF32WithBackend(
    const float* signal,
    const float* dt_logits,
    const float* input_b,
    const float* readout_c,
    const float* gate,
    const float* a,
    float* state_io,
    float* output,
    std::size_t batch,
    std::size_t d_model,
    std::size_t d_state,
    float dt_min,
    float dt_max,
    RecurrentBackend backend);

RecurrentStatus FusedSelectiveStepF32(
    const float* signal,
    const float* dt_logits,
    const float* input_b,
    const float* readout_c,
    const float* gate,
    const float* a,
    float* state_io,
    float* output,
    std::size_t batch,
    std::size_t d_model,
    std::size_t d_state,
    float dt_min,
    float dt_max);

}  // namespace vn97

extern "C" {

int vn97_selective_prefill_f32(
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
    float dt_max,
    int backend);

int vn97_selective_step_f32(
    const float* signal,
    const float* dt_logits,
    const float* input_b,
    const float* readout_c,
    const float* gate,
    const float* a,
    float* state_io,
    float* output,
    std::size_t batch,
    std::size_t d_model,
    std::size_t d_state,
    float dt_min,
    float dt_max,
    int backend);

}
