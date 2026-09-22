#include "vn97/sampler.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

int main() {
    const std::array<float, 5> logits = {0.0f, 2.0f, 2.0f, -1.0f, 1.0f};

    vn97::SamplerConfig greedy;
    greedy.temperature = 0.0f;
    greedy.top_k = 0;
    greedy.top_p = 1.0f;
    std::uint32_t token = 99;
    assert(vn97::SampleLogitsF32(logits.data(), logits.size(), greedy, 0, &token) ==
           vn97::SamplerStatus::kOk);
    assert(token == 1);  // stable lowest-ID tie break.

    vn97::SamplerConfig top_one;
    top_one.temperature = 1.0f;
    top_one.top_k = 1;
    top_one.top_p = 1.0f;
    top_one.seed = 97;
    std::array<vn97::SamplerCandidate, 5> workspace{};
    for (std::uint64_t step = 0; step < 32; ++step) {
        assert(vn97::SampleLogitsF32WithWorkspace(
                   logits.data(),
                   logits.size(),
                   top_one,
                   step,
                   workspace.data(),
                   workspace.size(),
                   &token) ==
               vn97::SamplerStatus::kOk);
        assert(token == 1);
    }
    assert(vn97::SampleLogitsF32WithWorkspace(
               logits.data(),
               logits.size(),
               top_one,
               0,
               workspace.data(),
               1,
               &token) ==
           vn97::SamplerStatus::kWorkspaceTooSmall);

    const std::array<float, 3> peaked = {8.0f, 1.0f, 0.0f};
    vn97::SamplerConfig nucleus;
    nucleus.temperature = 1.0f;
    nucleus.top_k = 0;
    nucleus.top_p = 0.5f;
    nucleus.seed = 1234;
    for (std::uint64_t step = 0; step < 16; ++step) {
        assert(vn97::SampleLogitsF32(peaked.data(), peaked.size(), nucleus, step, &token) ==
               vn97::SamplerStatus::kOk);
        assert(token == 0);
    }

    vn97::SamplerConfig deterministic;
    deterministic.temperature = 0.8f;
    deterministic.top_k = 4;
    deterministic.top_p = 0.9f;
    deterministic.seed = 0x123456789abcdef0ull;
    std::uint32_t first = 0;
    std::uint32_t second = 0;
    assert(vn97::SampleLogitsF32(logits.data(), logits.size(), deterministic, 11, &first) ==
           vn97::SamplerStatus::kOk);
    assert(vn97::SampleLogitsF32(logits.data(), logits.size(), deterministic, 11, &second) ==
           vn97::SamplerStatus::kOk);
    assert(first == second);

    auto invalid = deterministic;
    invalid.top_p = 0.0f;
    assert(vn97::SampleLogitsF32(logits.data(), logits.size(), invalid, 0, &token) ==
           vn97::SamplerStatus::kInvalidConfig);
    invalid = deterministic;
    invalid.temperature = -0.1f;
    assert(vn97::SampleLogitsF32(logits.data(), logits.size(), invalid, 0, &token) ==
           vn97::SamplerStatus::kInvalidConfig);

    auto bad_logits = logits;
    bad_logits[3] = std::numeric_limits<float>::quiet_NaN();
    assert(vn97::SampleLogitsF32(bad_logits.data(), bad_logits.size(), deterministic, 0, &token) ==
           vn97::SamplerStatus::kNonFinite);

    assert(vn97::SampleLogitsF32(nullptr, logits.size(), deterministic, 0, &token) ==
           vn97::SamplerStatus::kNullArgument);
    return 0;
}
