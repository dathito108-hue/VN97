#include "vn97/packed_ternary.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
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

}  // namespace

int main() {
    std::vector<std::uint8_t> blob = {'V', 'N', '9', '7', 'T', '2', 0, 0};
    PushU32(blob, 1);
    PushU32(blob, 2);
    PushU32(blob, 4);
    PushU32(blob, 2);
    PushU32(blob, 4);
    PushU32(blob, 2);
    PushU32(blob, 4);
    PushU32(blob, 2);
    PushF32(blob, 2.0f);
    PushF32(blob, 0.5f);
    blob.push_back(0x64);
    blob.push_back(0x92);

    vn97::PackedTernaryView view;
    assert(vn97::ParsePackedTernary(blob.data(), blob.size(), &view) ==
           vn97::PackedTernaryStatus::kOk);

    const float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[2] = {};
    assert(vn97::PackedTernaryMatVecF32(view, input, nullptr, output) ==
           vn97::PackedTernaryStatus::kOk);
    assert(std::fabs(output[0] - 6.0f) < 1e-6f);
    assert(std::fabs(output[1] + 1.0f) < 1e-6f);
    return 0;
}
