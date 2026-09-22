#include "recurrent_internal.h"

#if !defined(__aarch64__)
#error "recurrent_neon.cpp must only be built for AArch64"
#endif

#include <arm_neon.h>

#include <cmath>
#include <cstddef>

namespace vn97::internal {

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

                float32x4_t vector_sum = vdupq_n_f32(0.0f);
                std::size_t n = 0;
                for (; n + 4 <= d_state; n += 4) {
                    const float32x4_t state_v =
                        vld1q_f32(model_state + n);
                    const float32x4_t decay_v =
                        vld1q_f32(model_decay + n);
                    const float32x4_t drive_v =
                        vld1q_f32(model_drive + n);
                    const float32x4_t h_v =
                        vfmaq_f32(drive_v, decay_v, state_v);
                    vst1q_f32(model_state + n, h_v);
                    vector_sum = vfmaq_f32(
                        vector_sum, h_v, vld1q_f32(bt_readout + n));
                }

                float sum = vaddvq_f32(vector_sum);
                for (; n < d_state; ++n) {
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

}  // namespace vn97::internal
