#include "vn97/packed_ternary.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

void PushU32(std::vector<std::uint8_t>& v, std::uint32_t x) {
    v.push_back(static_cast<std::uint8_t>(x));
    v.push_back(static_cast<std::uint8_t>(x >> 8));
    v.push_back(static_cast<std::uint8_t>(x >> 16));
    v.push_back(static_cast<std::uint8_t>(x >> 24));
}

void PushF32(std::vector<std::uint8_t>& v, float x) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    PushU32(v, bits);
}

std::uint8_t EncodeSymbol(int symbol) {
    if (symbol > 0) return 1u;
    if (symbol < 0) return 2u;
    return 0u;
}

void PushFourSymbols(std::vector<std::uint8_t>& v, int s0, int s1, int s2, int s3) {
    const std::uint8_t packed = EncodeSymbol(s0) |
        static_cast<std::uint8_t>(EncodeSymbol(s1) << 2) |
        static_cast<std::uint8_t>(EncodeSymbol(s2) << 4) |
        static_cast<std::uint8_t>(EncodeSymbol(s3) << 6);
    v.push_back(packed);
}

std::vector<std::uint8_t> BuildTwoBySixteenBlob() {
    std::vector<std::uint8_t> blob = {'V', 'N', '9', '7', 'T', '2', 0, 0};
    PushU32(blob, 1); PushU32(blob, 2); PushU32(blob, 16); PushU32(blob, 2);
    PushU32(blob, 16); PushU32(blob, 2); PushU32(blob, 16); PushU32(blob, 8);
    PushF32(blob, 1.0f); PushF32(blob, 0.25f);
    for (int i = 0; i < 4; ++i) PushFourSymbols(blob, 1, -1, 1, -1);
    for (int i = 0; i < 4; ++i) PushFourSymbols(blob, 1, 1, 0, -1);
    return blob;
}

void AssertNear(float actual, float expected, float tolerance = 1e-5f) {
    assert(std::fabs(actual - expected) <= tolerance);
}

}  // namespace

int main() {
    std::vector<std::uint8_t> blob = {'V', 'N', '9', '7', 'T', '2', 0, 0};
    PushU32(blob, 1); PushU32(blob, 2); PushU32(blob, 4); PushU32(blob, 2);
    PushU32(blob, 4); PushU32(blob, 2); PushU32(blob, 4); PushU32(blob, 2);
    PushF32(blob, 2.0f); PushF32(blob, 0.5f);
    blob.push_back(0x64); blob.push_back(0x92);

    vn97::PackedTernaryView view;
    assert(vn97::ParsePackedTernary(blob.data(), blob.size(), &view) == vn97::PackedTernaryStatus::kOk);

    auto bad_scale = blob;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::uint32_t nan_bits = 0;
    std::memcpy(&nan_bits, &nan, sizeof(nan_bits));
    for (int i = 0; i < 4; ++i) {
        bad_scale[40 + i] = static_cast<std::uint8_t>(
            (nan_bits >> (8 * i)) & 0xffu);
    }
    assert(
        vn97::ParsePackedTernary(
            bad_scale.data(),
            bad_scale.size(),
            &view) ==
        vn97::PackedTernaryStatus::kInvalidScale);

    bad_scale = blob;
    std::fill(bad_scale.begin() + 40, bad_scale.begin() + 44, 0);
    assert(
        vn97::ParsePackedTernary(
            bad_scale.data(),
            bad_scale.size(),
            &view) ==
        vn97::PackedTernaryStatus::kInvalidScale);

    const float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float scalar_output[2] = {};
    float auto_output[2] = {};
    assert(vn97::PackedTernaryMatVecF32WithBackend(view, input, nullptr, scalar_output,
        vn97::PackedTernaryBackend::kScalar) == vn97::PackedTernaryStatus::kOk);
    assert(vn97::PackedTernaryMatVecF32(view, input, nullptr, auto_output) == vn97::PackedTernaryStatus::kOk);
    AssertNear(scalar_output[0], 6.0f); AssertNear(scalar_output[1], -1.0f);
    AssertNear(auto_output[0], scalar_output[0]); AssertNear(auto_output[1], scalar_output[1]);

    assert(std::strcmp(vn97::PackedTernaryBackendName(vn97::PackedTernaryBackend::kScalar), "scalar") == 0);
    assert(vn97::PackedTernaryBackendAvailable(vn97::PackedTernaryBackend::kScalar));

    const auto wide_blob = BuildTwoBySixteenBlob();
    vn97::PackedTernaryView wide_view;
    assert(vn97::ParsePackedTernary(wide_blob.data(), wide_blob.size(), &wide_view) == vn97::PackedTernaryStatus::kOk);
    float wide_input[16] = {};
    for (int i = 0; i < 16; ++i) wide_input[i] = static_cast<float>(i + 1);
    const float bias[2] = {0.5f, -1.0f};
    float reference[2] = {};
    assert(vn97::PackedTernaryMatVecF32WithBackend(wide_view, wide_input, bias, reference,
        vn97::PackedTernaryBackend::kScalar) == vn97::PackedTernaryStatus::kOk);

    if (vn97::PackedTernaryBackendAvailable(vn97::PackedTernaryBackend::kArm64Neon)) {
        assert(vn97::ResolvePackedTernaryBackend(vn97::PackedTernaryBackend::kAuto) == vn97::PackedTernaryBackend::kArm64Neon);
        float neon_output[2] = {};
        assert(vn97::PackedTernaryMatVecF32WithBackend(wide_view, wide_input, bias, neon_output,
            vn97::PackedTernaryBackend::kArm64Neon) == vn97::PackedTernaryStatus::kOk);
        AssertNear(neon_output[0], reference[0]); AssertNear(neon_output[1], reference[1]);
    } else {
        assert(vn97::ResolvePackedTernaryBackend(vn97::PackedTernaryBackend::kAuto) == vn97::PackedTernaryBackend::kScalar);
        float unavailable_output[2] = {};
        assert(vn97::PackedTernaryMatVecF32WithBackend(wide_view, wide_input, bias, unavailable_output,
            vn97::PackedTernaryBackend::kArm64Neon) == vn97::PackedTernaryStatus::kBackendUnavailable);
    }
    return 0;
}
