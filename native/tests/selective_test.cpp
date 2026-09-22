#include "vn97/recurrent.h"
#include "vn97/selective.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

void AssertNear(float actual, float expected, float tolerance = 2e-5f) {
    assert(std::fabs(actual - expected) <= tolerance);
}

void AssertVectorNear(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float tolerance = 2e-5f) {
    assert(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        AssertNear(actual[i], expected[i], tolerance);
    }
}

float SoftplusDefault(float x) {
    if (x > 20.0f) return x;
    return std::log1p(std::exp(x));
}

float Silu(float x) {
    return x / (1.0f + std::exp(-x));
}

}  // namespace

int main() {
    constexpr std::size_t batch = 1;
    constexpr std::size_t sequence = 3;
    constexpr std::size_t d_model = 2;
    constexpr std::size_t d_state = 3;
    constexpr float dt_min = 0.05f;
    constexpr float dt_max = 0.8f;

    const float signal[6] = {
        1.0f, -0.5f,
        0.25f, 1.25f,
        -0.75f, 0.4f,
    };
    const float dt_logits[6] = {
        -4.0f, 0.0f,
        0.5f, 3.0f,
        1.0f, -1.0f,
    };
    const float input_b[9] = {
        0.5f, -0.25f, 1.0f,
        -0.4f, 0.8f, 0.2f,
        0.3f, 0.6f, -0.5f,
    };
    const float readout_c[9] = {
        1.0f, 0.5f, -0.25f,
        0.2f, -0.4f, 0.8f,
        -0.5f, 0.3f, 0.7f,
    };
    const float gate[6] = {
        0.0f, 1.0f,
        -0.5f, 0.25f,
        0.75f, -1.0f,
    };
    const float a[6] = {
        -0.1f, -0.5f, -2.0f,
        -0.2f, -1.0f, -4.0f,
    };
    const float initial[6] = {
        0.5f, -0.25f, 1.0f,
        -0.5f, 0.75f, 0.25f,
    };

    std::vector<float> expected_state(initial, initial + 6);
    std::vector<float> expected_output(6, 0.0f);
    std::vector<float> decay(18, 0.0f);
    std::vector<float> drive(18, 0.0f);

    for (std::size_t t = 0; t < sequence; ++t) {
        for (std::size_t m = 0; m < d_model; ++m) {
            const std::size_t tm = t * d_model + m;
            const float dt = std::clamp(
                SoftplusDefault(dt_logits[tm]), dt_min, dt_max);
            float sum = 0.0f;
            for (std::size_t n = 0; n < d_state; ++n) {
                const std::size_t dyn = tm * d_state + n;
                const std::size_t state_index = m * d_state + n;
                const float a_value = a[state_index];
                const float z = a_value * dt;
                decay[dyn] = std::exp(z);
                const float zoh = std::expm1(z) / a_value;
                drive[dyn] = zoh * input_b[t * d_state + n] * signal[tm];
                expected_state[state_index] =
                    decay[dyn] * expected_state[state_index] + drive[dyn];
                sum += expected_state[state_index] * readout_c[t * d_state + n];
            }
            expected_output[tm] = sum * Silu(gate[tm]);
        }
    }

    std::vector<float> selective_state(initial, initial + 6);
    std::vector<float> selective_output(6, 0.0f);
    assert(vn97::FusedSelectivePrefillF32WithBackend(
        signal, dt_logits, input_b, readout_c, gate, a,
        selective_state.data(), selective_output.data(),
        batch, sequence, d_model, d_state, dt_min, dt_max,
        vn97::RecurrentBackend::kScalar) == vn97::RecurrentStatus::kOk);
    AssertVectorNear(selective_state, expected_state);
    AssertVectorNear(selective_output, expected_output);

    std::vector<float> recurrent_state(initial, initial + 6);
    std::vector<float> recurrent_output(6, 0.0f);
    assert(vn97::FusedRecurrentPrefillF32WithBackend(
        decay.data(), drive.data(), readout_c, gate,
        recurrent_state.data(), recurrent_output.data(),
        batch, sequence, d_model, d_state,
        vn97::RecurrentBackend::kScalar) == vn97::RecurrentStatus::kOk);
    AssertVectorNear(selective_state, recurrent_state);
    AssertVectorNear(selective_output, recurrent_output);

    std::vector<float> step_state(initial, initial + 6);
    std::vector<float> step_output(6, 0.0f);
    for (std::size_t t = 0; t < sequence; ++t) {
        assert(vn97::FusedSelectiveStepF32WithBackend(
            signal + t * d_model,
            dt_logits + t * d_model,
            input_b + t * d_state,
            readout_c + t * d_state,
            gate + t * d_model,
            a,
            step_state.data(),
            step_output.data() + t * d_model,
            batch, d_model, d_state, dt_min, dt_max,
            vn97::RecurrentBackend::kScalar) == vn97::RecurrentStatus::kOk);
    }
    AssertVectorNear(step_state, selective_state);
    AssertVectorNear(step_output, selective_output);

    std::vector<float> invalid_state(initial, initial + 6);
    std::vector<float> invalid_output(6, 0.0f);
    assert(vn97::FusedSelectivePrefillF32WithBackend(
        signal, dt_logits, input_b, readout_c, gate, a,
        invalid_state.data(), invalid_output.data(),
        batch, sequence, d_model, d_state, 0.0f, dt_max,
        vn97::RecurrentBackend::kScalar) ==
        vn97::RecurrentStatus::kInvalidParameter);

    const float bad_a[6] = {
        -0.1f, -0.5f, 0.0f, -0.2f, -1.0f, -4.0f,
    };
    invalid_state.assign(initial, initial + 6);
    const auto invalid_before = invalid_state;
    assert(vn97::FusedSelectivePrefillF32WithBackend(
        signal, dt_logits, input_b, readout_c, gate, bad_a,
        invalid_state.data(), invalid_output.data(),
        batch, sequence, d_model, d_state, dt_min, dt_max,
        vn97::RecurrentBackend::kScalar) ==
        vn97::RecurrentStatus::kInvalidParameter);
    assert(std::memcmp(
        invalid_state.data(), invalid_before.data(),
        invalid_before.size() * sizeof(float)) == 0);

    if (vn97::RecurrentBackendAvailable(
            vn97::RecurrentBackend::kArm64Neon)) {
        std::vector<float> neon_state(initial, initial + 6);
        std::vector<float> neon_output(6, 0.0f);
        assert(vn97::FusedSelectivePrefillF32WithBackend(
            signal, dt_logits, input_b, readout_c, gate, a,
            neon_state.data(), neon_output.data(),
            batch, sequence, d_model, d_state, dt_min, dt_max,
            vn97::RecurrentBackend::kArm64Neon) ==
            vn97::RecurrentStatus::kOk);
        AssertVectorNear(neon_state, selective_state, 3e-5f);
        AssertVectorNear(neon_output, selective_output, 3e-5f);
    } else {
        std::vector<float> untouched(initial, initial + 6);
        const auto before = untouched;
        std::vector<float> out(6, 0.0f);
        assert(vn97::FusedSelectivePrefillF32WithBackend(
            signal, dt_logits, input_b, readout_c, gate, a,
            untouched.data(), out.data(),
            batch, sequence, d_model, d_state, dt_min, dt_max,
            vn97::RecurrentBackend::kArm64Neon) ==
            vn97::RecurrentStatus::kBackendUnavailable);
        assert(std::memcmp(
            untouched.data(), before.data(), before.size() * sizeof(float)) == 0);
    }

    return 0;
}
