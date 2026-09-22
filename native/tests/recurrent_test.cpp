#include "vn97/recurrent.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

void AssertNear(float actual, float expected, float tolerance = 1e-5f) {
    assert(std::fabs(actual - expected) <= tolerance);
}

float Silu(float x) {
    return x / (1.0f + std::exp(-x));
}

void AssertVectorNear(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float tolerance = 1e-5f) {
    assert(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        AssertNear(actual[i], expected[i], tolerance);
    }
}

}  // namespace

int main() {
    constexpr std::size_t batch = 1;
    constexpr std::size_t sequence = 3;
    constexpr std::size_t d_model = 2;
    constexpr std::size_t d_state = 3;

    const float decay[18] = {
        0.5f, 0.8f, 0.9f, 0.7f, 0.6f, 0.5f,
        0.4f, 0.9f, 0.8f, 0.5f, 0.7f, 0.6f,
        0.8f, 0.7f, 0.6f, 0.9f, 0.4f, 0.8f,
    };
    const float drive[18] = {
        1.0f, -0.5f, 0.25f, 0.2f, 0.4f, -0.3f,
        0.1f, 0.2f, -0.1f, -0.2f, 0.5f, 0.3f,
        0.3f, -0.4f, 0.2f, 0.6f, -0.1f, 0.5f,
    };
    const float readout[9] = {
        1.0f, 0.5f, -0.25f,
        0.2f, -0.4f, 0.8f,
        -0.5f, 0.3f, 0.7f,
    };
    const float gate[6] = {
        0.0f, 1.0f, -0.5f, 0.25f, 0.75f, -1.0f,
    };
    const float initial[6] = {
        0.5f, -0.25f, 1.0f, -0.5f, 0.75f, 0.25f,
    };

    std::vector<float> scalar_state(initial, initial + 6);
    std::vector<float> scalar_output(6, 0.0f);
    assert(vn97::FusedRecurrentPrefillF32WithBackend(
        decay, drive, readout, gate,
        scalar_state.data(), scalar_output.data(),
        batch, sequence, d_model, d_state,
        vn97::RecurrentBackend::kScalar) == vn97::RecurrentStatus::kOk);

    std::vector<float> expected_state(initial, initial + 6);
    std::vector<float> expected_output(6, 0.0f);
    for (std::size_t t = 0; t < sequence; ++t) {
        for (std::size_t m = 0; m < d_model; ++m) {
            float sum = 0.0f;
            for (std::size_t n = 0; n < d_state; ++n) {
                const std::size_t dyn = (t * d_model + m) * d_state + n;
                const std::size_t s = m * d_state + n;
                expected_state[s] =
                    decay[dyn] * expected_state[s] + drive[dyn];
                sum += expected_state[s] * readout[t * d_state + n];
            }
            expected_output[t * d_model + m] =
                sum * Silu(gate[t * d_model + m]);
        }
    }
    AssertVectorNear(scalar_state, expected_state);
    AssertVectorNear(scalar_output, expected_output);

    std::vector<float> step_state(initial, initial + 6);
    std::vector<float> step_output(6, 0.0f);
    for (std::size_t t = 0; t < sequence; ++t) {
        assert(vn97::FusedRecurrentStepF32WithBackend(
            decay + t * d_model * d_state,
            drive + t * d_model * d_state,
            readout + t * d_state,
            gate + t * d_model,
            step_state.data(),
            step_output.data() + t * d_model,
            batch, d_model, d_state,
            vn97::RecurrentBackend::kScalar) == vn97::RecurrentStatus::kOk);
    }
    AssertVectorNear(step_state, scalar_state);
    AssertVectorNear(step_output, scalar_output);

    if (vn97::RecurrentBackendAvailable(
            vn97::RecurrentBackend::kArm64Neon)) {
        std::vector<float> neon_state(initial, initial + 6);
        std::vector<float> neon_output(6, 0.0f);
        assert(vn97::FusedRecurrentPrefillF32WithBackend(
            decay, drive, readout, gate,
            neon_state.data(), neon_output.data(),
            batch, sequence, d_model, d_state,
            vn97::RecurrentBackend::kArm64Neon) == vn97::RecurrentStatus::kOk);
        AssertVectorNear(neon_state, scalar_state, 2e-5f);
        AssertVectorNear(neon_output, scalar_output, 2e-5f);
    } else {
        std::vector<float> untouched(initial, initial + 6);
        std::vector<float> out(6, 0.0f);
        const auto before = untouched;
        assert(vn97::FusedRecurrentPrefillF32WithBackend(
            decay, drive, readout, gate,
            untouched.data(), out.data(),
            batch, sequence, d_model, d_state,
            vn97::RecurrentBackend::kArm64Neon) ==
            vn97::RecurrentStatus::kBackendUnavailable);
        assert(std::memcmp(
            untouched.data(), before.data(), before.size() * sizeof(float)) == 0);
    }

    return 0;
}
