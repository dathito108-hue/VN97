#pragma once

#include <cstddef>

namespace vn97 {

enum class RecurrentStatus {
    kOk = 0,
    kNullArgument,
    kInvalidShape,
    kSizeOverflow,
    kBackendUnavailable,
    kInvalidParameter,
};

enum class RecurrentBackend {
    kAuto = 0,
    kScalar,
    kArm64Neon,
};

const char* RecurrentBackendName(RecurrentBackend backend);
bool RecurrentBackendAvailable(RecurrentBackend backend);
RecurrentBackend ResolveRecurrentBackend(RecurrentBackend requested);

RecurrentStatus FusedRecurrentPrefillF32WithBackend(
    const float* decay,
    const float* drive,
    const float* readout,
    const float* gate,
    float* state_io,
    float* output,
    std::size_t batch,
    std::size_t sequence,
    std::size_t d_model,
    std::size_t d_state,
    RecurrentBackend backend);

RecurrentStatus FusedRecurrentPrefillF32(
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

RecurrentStatus FusedRecurrentStepF32WithBackend(
    const float* decay,
    const float* drive,
    const float* readout,
    const float* gate,
    float* state_io,
    float* output,
    std::size_t batch,
    std::size_t d_model,
    std::size_t d_state,
    RecurrentBackend backend);

RecurrentStatus FusedRecurrentStepF32(
    const float* decay,
    const float* drive,
    const float* readout,
    const float* gate,
    float* state_io,
    float* output,
    std::size_t batch,
    std::size_t d_model,
    std::size_t d_state);

}  // namespace vn97

extern "C" {

const char* vn97_recurrent_backend_name(int backend);
int vn97_recurrent_backend_available(int backend);

int vn97_recurrent_prefill_f32(
    const float* decay,
    const float* drive,
    const float* readout,
    const float* gate,
    float* state_io,
    float* output,
    std::size_t batch,
    std::size_t sequence,
    std::size_t d_model,
    std::size_t d_state,
    int backend);

int vn97_recurrent_step_f32(
    const float* decay,
    const float* drive,
    const float* readout,
    const float* gate,
    float* state_io,
    float* output,
    std::size_t batch,
    std::size_t d_model,
    std::size_t d_state,
    int backend);

}
