#include "vn97/runtime.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <vector>

int main() {
    std::array<std::uint8_t, 32> id{};
    std::uint64_t h = 0;
    assert(vn97_runtime_create_external_identity(id.data(), id.size(), &h) != 0);
    id[0] = 97;
    assert(vn97_runtime_create_external_identity(id.data(), 31, &h) != 0);
    assert(vn97_runtime_create_external_identity(id.data(), id.size(), &h) == 0);
    vn97_runtime_info info{};
    assert(vn97_runtime_info_get(h, &info) == 0);
    assert(info.state_count == 1 && info.sequence_position == 0);
    float value = 2.0f;
    assert(vn97_runtime_state_write(h, &value, 1) != 0);
    assert(vn97_runtime_activate(h) == 0);
    assert(vn97_runtime_advance(h, 1) != 0);
    vn97::LanguageModelView model{};
    std::uint32_t token = 0;
    float output = 0;
    assert(vn97_runtime_infer_step(h, &model, &token, 1, &output, 1) != 0);
    assert(vn97_runtime_suspend(h) == 0);
    std::size_t size = 0, written = 0;
    assert(vn97_runtime_checkpoint_size(h, &size) == 0 && size == 104);
    std::vector<std::uint8_t> bytes(size);
    assert(vn97_runtime_checkpoint_write(h, bytes.data(), size, &written) == 0);
    assert(std::memcmp(bytes.data(), "VN97PLN3", 8) == 0);
    assert(vn97_runtime_destroy(h) == 0);
    assert(vn97_runtime_restore(bytes.data(), size, &h) == 0);
    int bound = 0;
    std::array<std::uint8_t, 32> restored{};
    assert(vn97_runtime_model_binding_get(h, &bound, restored.data(), restored.size()) == 0);
    assert(bound == 1 && restored == id);
    assert(vn97_runtime_state_write(h, &value, 1) != 0);
    assert(vn97_runtime_resume(h) == 0);
    assert(vn97_runtime_infer_step(h, &model, &token, 1, &output, 1) != 0);
    assert(vn97_runtime_suspend(h) == 0);
    std::vector<std::uint8_t> again(size);
    assert(vn97_runtime_checkpoint_write(h, again.data(), size, &written) == 0);
    assert(again == bytes);
    assert(vn97_runtime_destroy(h) == 0);
    bytes.back() ^= 1;
    assert(vn97_runtime_restore(bytes.data(), size, &h) != 0);
}
