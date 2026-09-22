#include "vn97/selective.h"

#include "selective_internal.h"

#include <algorithm>
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
        MulOverflows(batch * sequence, d_state) ||
        MulOverflows(batch, d_model) ||
        MulOverflows(batch * d_model, d_state) ||
        MulOverflows(d_model, d_state)) {
        return RecurrentStatus::kSizeOverflow;
    }
    return RecurrentStatus::kOk;
}

bool ValidDtBounds(float dt_min, float dt_max) {
    return std::isfinite(dt_min) && std::isfinite(dt_max) &&
           dt_min > 0.0f && dt_max >= dt_min;
}

bool ValidStableA(
    const float* a,
    std::size_t d_model,
    std::size_t d_state) {
    const std::size_t count = d_model * d_state;
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(a[i]) || !(a[i] < 0.0f)) {
            return false;
        }
    }
    return true;
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

inline float SoftplusDefault(float x) {
    if (x > 20.0f) return x;
    return std::log1p(std::exp(x));
}

inline float Silu(float x) {
    return x / (1.0f + std::exp(-x));
}

}  // namespace

namespace internal {

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
    float dt_max) {
    for (std::size_t batch_index = 0; batch_index < batch; ++batch_index) {
        float* batch_state = state_io + batch_index * d_model * d_state;
        for (std::size_t t = 0; t < sequence; ++t) {
            const std::size_t bt = batch_index * sequence + t;
            const float* bt_signal = signal + bt * d_model;
            const float* bt_dt = dt_logits + bt * d_model;
            const float* bt_b = input_b + bt * d_state;
            const float* bt_c = readout_c + bt * d_state;
            const float* bt_gate = gate + bt * d_model;
            float* bt_output = output + bt * d_model;

            for (std::size_t m = 0; m < d_model; ++m) {
                const float dt = std::clamp(
                    SoftplusDefault(bt_dt[m]), dt_min, dt_max);
                const float signal_value = bt_signal[m];
                const float* model_a = a + m * d_state;
                float* model_state = batch_state + m * d_state;
                float sum = 0.0f;

                for (std::size_t n = 0; n < d_state; ++n) {
                    const float a_value = model_a[n];
                    const float z = a_value * dt;
                    const float decay = std::exp(z);
                    const float zoh = std::expm1(z) / a_value;
                    const float drive = zoh * bt_b[n] * signal_value;
                    const float h = decay * model_state[n] + drive;
                    model_state[n] = h;
                    sum += h * bt_c[n];
                }

                bt_output[m] = sum * Silu(bt_gate[m]);
            }
        }
    }
    return RecurrentStatus::kOk;
}

}  // namespace internal

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
    RecurrentBackend backend) {
    if (signal == nullptr || dt_logits == nullptr || input_b == nullptr ||
        readout_c == nullptr || gate == nullptr || a == nullptr ||
        state_io == nullptr || output == nullptr) {
        return RecurrentStatus::kNullArgument;
    }

    const RecurrentStatus shape_status =
        ValidateShape(batch, sequence, d_model, d_state);
    if (shape_status != RecurrentStatus::kOk) return shape_status;
    if (!ValidDtBounds(dt_min, dt_max) || !ValidStableA(a, d_model, d_state)) {
        return RecurrentStatus::kInvalidParameter;
    }

    const RecurrentBackend resolved = ResolveRecurrentBackend(backend);
    if (!RecurrentBackendAvailable(resolved)) {
        return RecurrentStatus::kBackendUnavailable;
    }

    switch (resolved) {
        case RecurrentBackend::kScalar:
            return internal::FusedSelectivePrefillF32Scalar(
                signal, dt_logits, input_b, readout_c, gate, a,
                state_io, output, batch, sequence, d_model, d_state,
                dt_min, dt_max);
        case RecurrentBackend::kArm64Neon:
#if defined(VN97_HAS_ARM64_NEON)
            return internal::FusedSelectivePrefillF32Arm64Neon(
                signal, dt_logits, input_b, readout_c, gate, a,
                state_io, output, batch, sequence, d_model, d_state,
                dt_min, dt_max);
#else
            return RecurrentStatus::kBackendUnavailable;
#endif
        case RecurrentBackend::kAuto:
            break;
    }
    return RecurrentStatus::kBackendUnavailable;
}

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
    float dt_max) {
    return FusedSelectivePrefillF32WithBackend(
        signal, dt_logits, input_b, readout_c, gate, a,
        state_io, output, batch, sequence, d_model, d_state,
        dt_min, dt_max, RecurrentBackend::kAuto);
}

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
    RecurrentBackend backend) {
    return FusedSelectivePrefillF32WithBackend(
        signal, dt_logits, input_b, readout_c, gate, a,
        state_io, output, batch, 1, d_model, d_state,
        dt_min, dt_max, backend);
}

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
    float dt_max) {
    return FusedSelectiveStepF32WithBackend(
        signal, dt_logits, input_b, readout_c, gate, a,
        state_io, output, batch, d_model, d_state,
        dt_min, dt_max, RecurrentBackend::kAuto);
}

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
    int backend) {
    vn97::RecurrentBackend decoded;
    if (!vn97::DecodeBackend(backend, &decoded)) {
        return static_cast<int>(vn97::RecurrentStatus::kBackendUnavailable);
    }
    return static_cast<int>(vn97::FusedSelectivePrefillF32WithBackend(
        signal, dt_logits, input_b, readout_c, gate, a,
        state_io, output, batch, sequence, d_model, d_state,
        dt_min, dt_max, decoded));
}

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
    int backend) {
    vn97::RecurrentBackend decoded;
    if (!vn97::DecodeBackend(backend, &decoded)) {
        return static_cast<int>(vn97::RecurrentStatus::kBackendUnavailable);
    }
    return static_cast<int>(vn97::FusedSelectiveStepF32WithBackend(
        signal, dt_logits, input_b, readout_c, gate, a,
        state_io, output, batch, d_model, d_state,
        dt_min, dt_max, decoded));
}

}
