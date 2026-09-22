#include "selective_internal.h"

#if !defined(__aarch64__)
#error "selective_neon.cpp must only be built for AArch64"
#endif

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace vn97::internal {
namespace {

inline float SoftplusDefault(float x) {
    if (x > 20.0f) return x;
    return std::log1p(std::exp(x));
}

inline float Silu(float x) {
    return x / (1.0f + std::exp(-x));
}

}  // namespace

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

                float32x4_t vector_sum = vdupq_n_f32(0.0f);
                std::size_t n = 0;
                for (; n + 4 <= d_state; n += 4) {
                    alignas(16) float decay_lanes[4];
                    alignas(16) float zoh_lanes[4];
                    for (std::size_t lane = 0; lane < 4; ++lane) {
                        const float a_value = model_a[n + lane];
                        const float z = a_value * dt;
                        decay_lanes[lane] = std::exp(z);
                        zoh_lanes[lane] = std::expm1(z) / a_value;
                    }

                    const float32x4_t decay_v = vld1q_f32(decay_lanes);
                    const float32x4_t zoh_v = vld1q_f32(zoh_lanes);
                    const float32x4_t b_v = vld1q_f32(bt_b + n);
                    const float32x4_t drive_v = vmulq_n_f32(
                        vmulq_f32(zoh_v, b_v), signal_value);
                    const float32x4_t state_v = vld1q_f32(model_state + n);
                    const float32x4_t h_v =
                        vfmaq_f32(drive_v, decay_v, state_v);
                    vst1q_f32(model_state + n, h_v);
                    vector_sum = vfmaq_f32(
                        vector_sum, h_v, vld1q_f32(bt_c + n));
                }

                float sum = vaddvq_f32(vector_sum);
                for (; n < d_state; ++n) {
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

}  // namespace vn97::internal
