#include "vn97/recurrent.h"

#include "recurrent_internal.h"

#include <cmath>
#include <limits>

namespace vn97 {
namespace {

bool MulOverflows(std::size_t a, std::size_t b) {
    return b != 0 && a > std::numeric_limits<std::size_t>::max() / b;
}

RecurrentStatus ValidateShape(
    std::size_t batch,
    std::size_t sequence,
    std::size_t d_model,
    std::size_t d_state) {
    if (batch == 0 || sequence == 0 || d_model == 0 || d_state == 0) {
        return RecurrentStatus::kInvalidShape;
    }
    if (MulOverflows(batch, sequence) ||
        MulOverflows(batch * sequence, d_model) ||
        MulOverflows(batch * sequence * d_model, d_state) ||
        MulOverflows(batch, d_model) ||
        MulOverflows(batch * d_model, d_state)) {
        return RecurrentStatus::kSizeOverflow;
    }
    return RecurrentStatus::kOk;
}

bool DecodeBackend(int backend, RecurrentBackend* out) {
    if (out == nullptr) return false;
    switch (backend) {
        case 0:
            *out = RecurrentBackend::kAuto;
            return true;
        case 1:
            *out = RecurrentBackend::kScalar;
            return true;
        case 2:
            *out = RecurrentBackend::kArm64Neon;
            return true;
        default:
            return false;
    }
}

}  // namespace

const char* RecurrentBackendName(RecurrentBackend backend) {
    switch (backend) {
        case RecurrentBackend::kAuto:
            return "auto";
        case RecurrentBackend::kScalar:
            return "scalar";
        case RecurrentBackend::kArm64Neon:
            return "arm64-neon";
    }
    return "unknown";
}

bool RecurrentBackendAvailable(RecurrentBackend backend) {
    switch (backend) {
        case RecurrentBackend::kAuto:
        case RecurrentBackend::kScalar:
            return true;
        case RecurrentBackend::kArm64Neon:
#if defined(VN97_HAS_ARM64_NEON)
            return true;
#else
            return false;
#endif
    }
    return false;
}

RecurrentBackend ResolveRecurrentBackend(RecurrentBackend requested) {
    if (requested != RecurrentBackend::kAuto) return requested;
#if defined(VN97_HAS_ARM64_NEON)
    return RecurrentBackend::kArm64Neon;
#else
    return RecurrentBackend::kScalar;
#endif
}

namespace internal {

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
    std::size_t d_state) {
    for (std::size_t b = 0; b < batch; ++b) {
        float* batch_state = state_io + b * d_model * d_state;
        for (std::size_t t = 0; t < sequence; ++t) {
            const std::size_t bt = b * sequence + t;
            const float* bt_readout = readout + bt * d_state;
            const float* bt_gate = gate + bt * d_model;
            float* bt_output = output + bt * d_model;
            const std::size_t dynamics_base = bt * d_model * d_state;

            for (std::size_t m = 0; m < d_model; ++m) {
                float* model_state = batch_state + m * d_state;
                const float* model_decay =
                    decay + dynamics_base + m * d_state;
                const float* model_drive =
                    drive + dynamics_base + m * d_state;
                float sum = 0.0f;

                for (std::size_t n = 0; n < d_state; ++n) {
                    const float h =
                        model_decay[n] * model_state[n] + model_drive[n];
                    model_state[n] = h;
                    sum += h * bt_readout[n];
                }

                const float g = bt_gate[m];
                const float silu = g / (1.0f + std::exp(-g));
                bt_output[m] = sum * silu;
            }
        }
    }
    return RecurrentStatus::kOk;
}

}  // namespace internal

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
    RecurrentBackend backend) {
    if (decay == nullptr || drive == nullptr || readout == nullptr ||
        gate == nullptr || state_io == nullptr || output == nullptr) {
        return RecurrentStatus::kNullArgument;
    }

    const RecurrentStatus shape_status =
        ValidateShape(batch, sequence, d_model, d_state);
    if (shape_status != RecurrentStatus::kOk) return shape_status;

    const RecurrentBackend resolved = ResolveRecurrentBackend(backend);
    if (!RecurrentBackendAvailable(resolved)) {
        return RecurrentStatus::kBackendUnavailable;
    }

    switch (resolved) {
        case RecurrentBackend::kScalar:
            return internal::FusedRecurrentPrefillF32Scalar(
                decay, drive, readout, gate, state_io, output,
                batch, sequence, d_model, d_state);
        case RecurrentBackend::kArm64Neon:
#if defined(VN97_HAS_ARM64_NEON)
            return internal::FusedRecurrentPrefillF32Arm64Neon(
                decay, drive, readout, gate, state_io, output,
                batch, sequence, d_model, d_state);
#else
            return RecurrentStatus::kBackendUnavailable;
#endif
        case RecurrentBackend::kAuto:
            break;
    }
    return RecurrentStatus::kBackendUnavailable;
}

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
    std::size_t d_state) {
    return FusedRecurrentPrefillF32WithBackend(
        decay, drive, readout, gate, state_io, output,
        batch, sequence, d_model, d_state, RecurrentBackend::kAuto);
}

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
    RecurrentBackend backend) {
    return FusedRecurrentPrefillF32WithBackend(
        decay, drive, readout, gate, state_io, output,
        batch, 1, d_model, d_state, backend);
}

RecurrentStatus FusedRecurrentStepF32(
    const float* decay,
    const float* drive,
    const float* readout,
    const float* gate,
    float* state_io,
    float* output,
    std::size_t batch,
    std::size_t d_model,
    std::size_t d_state) {
    return FusedRecurrentStepF32WithBackend(
        decay, drive, readout, gate, state_io, output,
        batch, d_model, d_state, RecurrentBackend::kAuto);
}

}  // namespace vn97

extern "C" {

const char* vn97_recurrent_backend_name(int backend) {
    vn97::RecurrentBackend decoded;
    if (!vn97::DecodeBackend(backend, &decoded)) return "unknown";
    return vn97::RecurrentBackendName(decoded);
}

int vn97_recurrent_backend_available(int backend) {
    vn97::RecurrentBackend decoded;
    if (!vn97::DecodeBackend(backend, &decoded)) return 0;
    return vn97::RecurrentBackendAvailable(decoded) ? 1 : 0;
}

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
    int backend) {
    vn97::RecurrentBackend decoded;
    if (!vn97::DecodeBackend(backend, &decoded)) {
        return static_cast<int>(vn97::RecurrentStatus::kBackendUnavailable);
    }
    return static_cast<int>(vn97::FusedRecurrentPrefillF32WithBackend(
        decay, drive, readout, gate, state_io, output,
        batch, sequence, d_model, d_state, decoded));
}

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
    int backend) {
    vn97::RecurrentBackend decoded;
    if (!vn97::DecodeBackend(backend, &decoded)) {
        return static_cast<int>(vn97::RecurrentStatus::kBackendUnavailable);
    }
    return static_cast<int>(vn97::FusedRecurrentStepF32WithBackend(
        decay, drive, readout, gate, state_io, output,
        batch, d_model, d_state, decoded));
}

}
