#include "vn97/generation.h"
#include "vn97/model_image.h"
#include "vn97/runtime.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <unistd.h>
#include <vector>

namespace {
constexpr std::uint32_t kGlobal = 0xffffffffu;

void WriteU32(std::uint8_t* p, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) p[i] = static_cast<std::uint8_t>((v >> (8 * i)) & 0xffu);
}
void WriteU64(std::uint8_t* p, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = static_cast<std::uint8_t>((v >> (8 * i)) & 0xffu);
}
void WriteF32(std::uint8_t* p, float v) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    WriteU32(p, bits);
}
std::vector<std::uint8_t> Floats(std::size_t n, float value) {
    std::vector<std::uint8_t> out(n * 4);
    for (std::size_t i = 0; i < n; ++i) WriteF32(out.data() + i * 4, value);
    return out;
}
std::vector<std::uint8_t> PackedZero(std::uint32_t rows, std::uint32_t cols) {
    const std::size_t packed_bytes =
        (static_cast<std::size_t>(rows) * cols + 3) / 4;
    std::vector<std::uint8_t> out(
        40 + static_cast<std::size_t>(rows) * 4 + packed_bytes, 0);
    const std::uint8_t magic[8] = {'V','N','9','7','T','2',0,0};
    std::copy(magic, magic + 8, out.begin());
    WriteU32(out.data() + 8, 1);
    WriteU32(out.data() + 12, rows);
    WriteU32(out.data() + 16, cols);
    WriteU32(out.data() + 20, 1);
    WriteU32(out.data() + 24, cols);
    WriteU32(out.data() + 28, rows);
    WriteU32(out.data() + 32, cols);
    WriteU32(out.data() + 36, static_cast<std::uint32_t>(packed_bytes));
    for (std::uint32_t row = 0; row < rows; ++row) {
        WriteF32(out.data() + 40 + static_cast<std::size_t>(row) * 4, 1.0f);
    }
    return out;
}

struct Section {
    std::uint32_t type;
    std::uint32_t layer;
    std::vector<std::uint8_t> bytes;
};

std::vector<std::uint8_t> BuildImage(bool with_audio = false) {
    std::vector<Section> sections;
    sections.push_back({1, kGlobal, Floats(15, 0.1f)});
    if (with_audio) {
        sections.push_back({6, kGlobal, PackedZero(3, 320)});
        sections.push_back({7, kGlobal, Floats(3, 1.0f)});
    }
    sections.push_back({16, 0, Floats(3, 1.0f)});
    sections.push_back({17, 0, PackedZero(6, 3)});
    sections.push_back({18, 0, PackedZero(3, 3)});
    sections.push_back({19, 0, Floats(3, 0.0f)});
    sections.push_back({20, 0, PackedZero(2, 3)});
    sections.push_back({21, 0, PackedZero(2, 3)});
    sections.push_back({22, 0, PackedZero(3, 3)});
    auto a_log = Floats(6, 0.0f);
    const float rates[6] = {-0.69314718f, -1.0f, -0.2f, -0.3f, -0.4f, -0.5f};
    for (std::size_t i = 0; i < 6; ++i) WriteF32(a_log.data() + i * 4, rates[i]);
    sections.push_back({23, 0, std::move(a_log)});
    sections.push_back({4, kGlobal, Floats(3, 1.0f)});

    const std::size_t payload_offset = 96 + sections.size() * 32;
    std::size_t cursor = payload_offset;
    std::vector<std::uint8_t> payload;
    struct Meta { std::uint32_t type; std::uint32_t layer; std::size_t offset; std::size_t size; };
    std::vector<Meta> metas;
    for (const auto& section : sections) {
        const std::size_t aligned = (cursor + 3) & ~static_cast<std::size_t>(3);
        payload.insert(payload.end(), aligned - cursor, 0);
        metas.push_back({section.type, section.layer, aligned, section.bytes.size()});
        payload.insert(payload.end(), section.bytes.begin(), section.bytes.end());
        cursor = aligned + section.bytes.size();
    }

    std::vector<std::uint8_t> out(cursor, 0);
    const std::uint8_t magic[8] = {'V','N','9','7','M','I','1',0};
    std::copy(magic, magic + 8, out.begin());
    WriteU32(out.data() + 8, 1);
    WriteU32(out.data() + 12, 96);
    WriteU32(out.data() + 16, with_audio ? (1u << 2) : 0u);
    WriteU32(out.data() + 20, static_cast<std::uint32_t>(sections.size()));
    WriteU32(out.data() + 24, 32);
    WriteU32(out.data() + 28, 5);
    WriteU32(out.data() + 32, 3);
    WriteU32(out.data() + 36, 1);
    WriteU32(out.data() + 40, 2);
    WriteU32(out.data() + 44, 0);
    WriteF32(out.data() + 48, 1e-4f);
    WriteF32(out.data() + 52, 1.0f);
    WriteF32(out.data() + 56, 1e-5f);
    WriteU64(out.data() + 64, 96);
    WriteU64(out.data() + 72, payload_offset);
    WriteU64(out.data() + 80, cursor);
    for (std::size_t i = 0; i < metas.size(); ++i) {
        auto* entry = out.data() + 96 + i * 32;
        WriteU32(entry, metas[i].type);
        WriteU32(entry + 4, metas[i].layer);
        WriteU64(entry + 8, metas[i].offset);
        WriteU64(entry + 16, metas[i].size);
    }
    std::copy(
        payload.begin(), payload.end(),
        out.begin() + static_cast<std::ptrdiff_t>(payload_offset));
    return out;
}

std::array<std::uint8_t, 32> ExpectedId() {
    constexpr char hex[] =
        "7676fffe46e2d47366544998c29cd6cc8ffeb4c453b8897b3835e5750c627e87";
    std::array<std::uint8_t, 32> out{};
    auto digit = [](char c) -> std::uint8_t {
        return static_cast<std::uint8_t>(c <= '9' ? c - '0' : 10 + c - 'a');
    };
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<std::uint8_t>((digit(hex[i * 2]) << 4) | digit(hex[i * 2 + 1]));
    }
    return out;
}

std::array<std::uint8_t, 32> ExpectedAudioId() {
    constexpr char hex[] =
        "af3dcfca0932918bc427fd40dfe185fdcd37ce66266bd8919a465157c246153c";
    std::array<std::uint8_t, 32> out{};
    auto digit = [](char c) -> std::uint8_t {
        return static_cast<std::uint8_t>(
            c <= '9' ? c - '0' : 10 + c - 'a');
    };
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<std::uint8_t>(
            (digit(hex[i * 2]) << 4) |
            digit(hex[i * 2 + 1]));
    }
    return out;
}
}

int main() {
    const auto image = BuildImage();
    assert(image.size() == 824);
    const auto id = ExpectedId();

    char path[] = "/tmp/vn97-mi-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    unlink(path);
    std::size_t written = 0;
    while (written < image.size()) {
        const ssize_t n = write(fd, image.data() + written, image.size() - written);
        assert(n > 0);
        written += static_cast<std::size_t>(n);
    }

    std::unique_ptr<vn97::ActivatedModelImage> direct;
    assert(vn97::ActivatedModelImage::OpenFd(fd, 0, image.size(), id.data(), &direct) ==
           vn97::ModelImageStatus::kOk);
    assert(direct != nullptr);
    assert(direct->Info().vocab_size == 5);
    assert(direct->language_model().layers[0].a[0] < 0.0f);

    const auto audio_image = BuildImage(true);
    assert(audio_image.size() == 1192);
    const auto audio_id = ExpectedAudioId();

    char audio_path[] = "/tmp/vn97-mi-audio-XXXXXX";
    const int audio_fd = mkstemp(audio_path);
    assert(audio_fd >= 0);
    unlink(audio_path);
    std::size_t audio_written = 0;
    while (audio_written < audio_image.size()) {
        const ssize_t n = write(
            audio_fd,
            audio_image.data() + audio_written,
            audio_image.size() - audio_written);
        assert(n > 0);
        audio_written += static_cast<std::size_t>(n);
    }

    std::unique_ptr<vn97::ActivatedModelImage> audio_direct;
    assert(
        vn97::ActivatedModelImage::OpenFd(
            audio_fd,
            0,
            audio_image.size(),
            audio_id.data(),
            &audio_direct) ==
        vn97::ModelImageStatus::kOk);
    assert(audio_direct != nullptr);
    assert(audio_direct->Info().has_audio_projection);
    assert(audio_direct->Info().audio_frame_size == 320);
    assert(audio_direct->audio_projection() != nullptr);

    std::uint64_t audio_model_handle = 0;
    assert(
        vn97_model_open_fd(
            audio_fd,
            0,
            audio_image.size(),
            audio_id.data(),
            audio_id.size(),
            &audio_model_handle) == 0);
    close(audio_fd);

    vn97_runtime_config audio_config{
        1,
        1,
        3,
        2,
        static_cast<int>(vn97::RecurrentBackend::kScalar),
        static_cast<int>(vn97::PackedTernaryBackend::kScalar),
    };
    std::uint64_t audio_runtime = 0;
    assert(
        vn97_runtime_create(
            &audio_config,
            &audio_runtime) == 0);
    assert(vn97_runtime_activate(audio_runtime) == 0);

    std::array<float, 320> prepared_audio{};
    std::array<float, 5> audio_logits{};
    assert(
        vn97_model_runtime_prefill_audio(
            audio_model_handle,
            audio_runtime,
            4,
            prepared_audio.data(),
            prepared_audio.size(),
            1,
            audio_logits.data(),
            audio_logits.size()) == 0);
    for (float value : audio_logits) {
        assert(std::isfinite(value));
    }
    vn97_runtime_info audio_runtime_info{};
    assert(
        vn97_runtime_info_get(
            audio_runtime,
            &audio_runtime_info) == 0);
    assert(audio_runtime_info.sequence_position == 2);
    assert(vn97_runtime_destroy(audio_runtime) == 0);
    assert(vn97_model_destroy(audio_model_handle) == 0);

    auto wrong_id = id;
    wrong_id[0] ^= 1;
    std::unique_ptr<vn97::ActivatedModelImage> rejected;
    assert(vn97::ActivatedModelImage::OpenFd(fd, 0, image.size(), wrong_id.data(), &rejected) ==
           vn97::ModelImageStatus::kIdentityMismatch);
    assert(rejected == nullptr);

    std::uint64_t model_handle = 0;
    assert(vn97_model_open_fd(fd, 0, image.size(), id.data(), id.size(), &model_handle) == 0);
    assert(model_handle != 0);
    close(fd);

    vn97::RuntimeConfig config;
    config.layers = 1;
    config.batch = 1;
    config.d_model = 3;
    config.d_state = 2;
    config.recurrent_backend = vn97::RecurrentBackend::kScalar;
    config.packed_backend = vn97::PackedTernaryBackend::kScalar;
    vn97_runtime_config c_config{
        config.layers,
        config.batch,
        config.d_model,
        config.d_state,
        static_cast<int>(config.recurrent_backend),
        static_cast<int>(config.packed_backend),
    };
    const std::array<std::uint32_t, 2> hidden_prompt = {1, 2};
    std::uint64_t hidden_runtime = 0;
    assert(vn97_runtime_create(&c_config, &hidden_runtime) == 0);
    assert(vn97_runtime_activate(hidden_runtime) == 0);
    std::array<float, 3> hidden{};
    assert(vn97_model_runtime_prefill_hidden(
               model_handle,
               hidden_runtime,
               hidden_prompt.data(),
               hidden_prompt.size(),
               hidden_prompt.size(),
               hidden.data(),
               hidden.size()) == 0);
    for (float value : hidden) assert(std::isfinite(value));

    vn97_runtime_info hidden_info{};
    assert(vn97_runtime_info_get(hidden_runtime, &hidden_info) == 0);
    assert(hidden_info.sequence_position == hidden_prompt.size());
    int hidden_bound = 0;
    std::array<std::uint8_t, 32> hidden_bound_id{};
    assert(vn97_runtime_model_binding_get(
               hidden_runtime,
               &hidden_bound,
               hidden_bound_id.data(),
               hidden_bound_id.size()) == 0);
    assert(hidden_bound == 1);
    assert(hidden_bound_id == id);
    assert(vn97_runtime_destroy(hidden_runtime) == 0);

    std::uint64_t runtime_handle = 0;
    assert(vn97_runtime_create(&c_config, &runtime_handle) == 0);
    assert(vn97_runtime_activate(runtime_handle) == 0);

    const std::uint32_t token = 1;
    std::array<float, 5> logits{};
    assert(vn97_model_runtime_infer_step(
               model_handle, runtime_handle, &token, 1, logits.data(), logits.size()) == 0);
    for (float value : logits) assert(std::isfinite(value));

    int bound = 0;
    std::array<std::uint8_t, 32> bound_id{};
    assert(vn97_runtime_model_binding_get(
               runtime_handle, &bound, bound_id.data(), bound_id.size()) == 0);
    assert(bound == 1);
    assert(bound_id == id);

    const std::array<std::uint32_t, 2> prompt = {1, 2};
    assert(vn97_model_runtime_prefill(
               model_handle, runtime_handle, prompt.data(), prompt.size(), 2,
               logits.data(), logits.size()) == 0);

    std::array<std::uint32_t, 2> generated{};
    std::size_t generated_count = 0;
    assert(vn97_model_runtime_generate_greedy(
               model_handle, runtime_handle, &token, 1, 2,
               std::numeric_limits<std::uint32_t>::max(),
               generated.data(), generated.size(), &generated_count) == 0);
    assert(generated_count == 2);

    vn97_runtime_info before_generation{};
    assert(vn97_runtime_info_get(runtime_handle, &before_generation) == 0);
    vn97_generation_config generation_config{};
    generation_config.sampler.temperature = 1.0f;
    generation_config.sampler.top_k = 1;
    generation_config.sampler.top_p = 1.0f;
    generation_config.sampler.seed = 97;
    generation_config.eos_token = std::numeric_limits<std::uint32_t>::max();
    const std::array<std::uint32_t, 1> streaming_prompt = {1};
    std::uint64_t generation_handle = 0;
    assert(vn97_generation_open(
               model_handle,
               runtime_handle,
               streaming_prompt.data(),
               streaming_prompt.size(),
               &generation_config,
               &generation_handle) ==
           static_cast<int>(vn97::GenerationStatus::kOk));
    assert(generation_handle != 0);

    vn97_runtime_info after_prefill{};
    assert(vn97_runtime_info_get(runtime_handle, &after_prefill) == 0);
    assert(
        after_prefill.sequence_position ==
        before_generation.sequence_position + streaming_prompt.size());

    std::uint32_t streamed_token = 0;
    int streamed_eos = 1;
    assert(vn97_generation_next(
               generation_handle,
               &streamed_token,
               &streamed_eos) ==
           static_cast<int>(vn97::GenerationStatus::kOk));
    assert(streamed_token < 5);
    assert(streamed_eos == 0);

    vn97_runtime_info after_streamed_token{};
    assert(vn97_runtime_info_get(runtime_handle, &after_streamed_token) == 0);
    assert(
        after_streamed_token.sequence_position ==
        after_prefill.sequence_position + 1);
    assert(vn97_generation_destroy(generation_handle) ==
           static_cast<int>(vn97::GenerationStatus::kOk));
    assert(vn97_generation_next(
               generation_handle,
               &streamed_token,
               &streamed_eos) ==
           static_cast<int>(vn97::GenerationStatus::kInvalidHandle));

    assert(vn97_runtime_suspend(runtime_handle) == 0);
    std::size_t checkpoint_size = 0;
    assert(vn97_runtime_checkpoint_size(runtime_handle, &checkpoint_size) == 0);
    assert(checkpoint_size >= 100);
    std::vector<std::uint8_t> checkpoint(checkpoint_size);
    std::size_t checkpoint_written = 0;
    assert(vn97_runtime_checkpoint_write(
               runtime_handle, checkpoint.data(), checkpoint.size(), &checkpoint_written) == 0);
    const std::array<std::uint8_t, 8> run2 = {'V','N','9','7','R','U','N','2'};
    assert(std::equal(run2.begin(), run2.end(), checkpoint.begin()));

    assert(vn97_runtime_destroy(runtime_handle) == 0);
    assert(vn97_model_destroy(model_handle) == 0);
    return 0;
}
