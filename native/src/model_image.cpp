#include "vn97/model_image.h"

#include "vn97/packed_ternary.h"
#include "vn97/runtime.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>

namespace vn97 {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic = {'V','N','9','7','M','I','1',0};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kHeaderSize = 96;
constexpr std::uint32_t kSectionEntrySize = 32;
constexpr std::uint32_t kFlagFactorized = 1u << 0;
constexpr std::uint32_t kFlagTokenizer = 1u << 1;
constexpr std::uint32_t kKnownFlags = kFlagFactorized | kFlagTokenizer;
constexpr std::uint32_t kGlobalLayer = 0xffffffffu;
constexpr std::uint32_t kMaxLayers = 4096;
constexpr std::uint32_t kMaxSections = kMaxLayers * 8u + 4u;

constexpr std::uint32_t kEmbedding = 1;
constexpr std::uint32_t kTokenFactors = 2;
constexpr std::uint32_t kEmbeddingProjection = 3;
constexpr std::uint32_t kFinalNorm = 4;
constexpr std::uint32_t kTokenizer = 5;
constexpr std::uint32_t kLayerNorm = 16;
constexpr std::uint32_t kInProj = 17;
constexpr std::uint32_t kDtProj = 18;
constexpr std::uint32_t kDtBias = 19;
constexpr std::uint32_t kBProj = 20;
constexpr std::uint32_t kCProj = 21;
constexpr std::uint32_t kOutProj = 22;
constexpr std::uint32_t kALog = 23;

struct Section {
    std::uint32_t type = 0;
    std::uint32_t layer = 0;
    std::uint64_t offset = 0;
    std::uint64_t size = 0;
    std::uint32_t flags = 0;
    std::uint32_t reserved = 0;
};

bool AddOverflow(std::size_t a, std::size_t b, std::size_t* out) {
    if (out == nullptr || b > std::numeric_limits<std::size_t>::max() - a) return true;
    *out = a + b;
    return false;
}

bool MulOverflow(std::size_t a, std::size_t b, std::size_t* out) {
    if (out == nullptr || (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)) return true;
    *out = a * b;
    return false;
}

std::uint32_t ReadU32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) |
           (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
}

std::uint64_t ReadU64(const std::uint8_t* p) {
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= static_cast<std::uint64_t>(p[i]) << (8 * i);
    return value;
}

float ReadF32(const std::uint8_t* p) {
    const std::uint32_t bits = ReadU32(p);
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::size_t Align4(std::size_t value) {
    return (value + 3u) & ~static_cast<std::size_t>(3u);
}

bool AllZero(const std::uint8_t* data, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        if (data[i] != 0) return false;
    }
    return true;
}

bool AnyNonZero(const std::uint8_t* data, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        if (data[i] != 0) return true;
    }
    return false;
}

// Minimal SHA-256 used only to bind the mapped artifact to the digest recorded by M9.
class Sha256 {
public:
    Sha256() { Reset(); }

    void Update(const std::uint8_t* data, std::size_t size) {
        if (data == nullptr || size == 0) return;
        total_bytes_ += size;
        while (size != 0) {
            const std::size_t take = std::min(size, block_.size() - block_used_);
            std::memcpy(block_.data() + block_used_, data, take);
            block_used_ += take;
            data += take;
            size -= take;
            if (block_used_ == block_.size()) {
                Transform(block_.data());
                block_used_ = 0;
            }
        }
    }

    std::array<std::uint8_t, 32> Final() {
        const std::uint64_t bit_count = static_cast<std::uint64_t>(total_bytes_) * 8u;
        block_[block_used_++] = 0x80;
        if (block_used_ > 56) {
            std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_used_), block_.end(), 0);
            Transform(block_.data());
            block_used_ = 0;
        }
        std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_used_), block_.begin() + 56, 0);
        for (int i = 0; i < 8; ++i) {
            block_[63 - i] = static_cast<std::uint8_t>((bit_count >> (8 * i)) & 0xffu);
        }
        Transform(block_.data());

        std::array<std::uint8_t, 32> out{};
        for (std::size_t i = 0; i < state_.size(); ++i) {
            out[i * 4 + 0] = static_cast<std::uint8_t>((state_[i] >> 24) & 0xffu);
            out[i * 4 + 1] = static_cast<std::uint8_t>((state_[i] >> 16) & 0xffu);
            out[i * 4 + 2] = static_cast<std::uint8_t>((state_[i] >> 8) & 0xffu);
            out[i * 4 + 3] = static_cast<std::uint8_t>(state_[i] & 0xffu);
        }
        return out;
    }

private:
    static std::uint32_t RotR(std::uint32_t x, std::uint32_t n) {
        return (x >> n) | (x << (32u - n));
    }

    void Reset() {
        state_ = {
            0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
            0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
        };
        block_.fill(0);
        block_used_ = 0;
        total_bytes_ = 0;
    }

    void Transform(const std::uint8_t* block) {
        static constexpr std::array<std::uint32_t, 64> k = {
            0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
            0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
            0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
            0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
            0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
            0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
            0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
            0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u,
        };
        std::uint32_t w[64] = {};
        for (int i = 0; i < 16; ++i) {
            const std::uint8_t* p = block + i * 4;
            w[i] = (static_cast<std::uint32_t>(p[0]) << 24) |
                   (static_cast<std::uint32_t>(p[1]) << 16) |
                   (static_cast<std::uint32_t>(p[2]) << 8) |
                   static_cast<std::uint32_t>(p[3]);
        }
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = RotR(w[i - 15], 7) ^ RotR(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = RotR(w[i - 2], 17) ^ RotR(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        std::uint32_t a=state_[0], b=state_[1], c=state_[2], d=state_[3];
        std::uint32_t e=state_[4], f=state_[5], g=state_[6], h=state_[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t s1 = RotR(e,6) ^ RotR(e,11) ^ RotR(e,25);
            const std::uint32_t ch = (e & f) ^ ((~e) & g);
            const std::uint32_t t1 = h + s1 + ch + k[i] + w[i];
            const std::uint32_t s0 = RotR(a,2) ^ RotR(a,13) ^ RotR(a,22);
            const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t t2 = s0 + maj;
            h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        state_[0]+=a; state_[1]+=b; state_[2]+=c; state_[3]+=d;
        state_[4]+=e; state_[5]+=f; state_[6]+=g; state_[7]+=h;
    }

    std::array<std::uint32_t, 8> state_{};
    std::array<std::uint8_t, 64> block_{};
    std::size_t block_used_ = 0;
    std::size_t total_bytes_ = 0;
};

ModelImageStatus F32Section(
    const std::uint8_t* image,
    std::size_t image_bytes,
    const Section& section,
    std::size_t expected_count,
    const float** out) {
    if (out == nullptr) return ModelImageStatus::kNullArgument;
    std::size_t expected_bytes = 0;
    if (MulOverflow(expected_count, sizeof(float), &expected_bytes) ||
        section.size != expected_bytes || section.offset % alignof(float) != 0 ||
        section.offset > image_bytes || section.size > image_bytes - section.offset) {
        return ModelImageStatus::kInvalidSection;
    }
    *out = reinterpret_cast<const float*>(image + section.offset);
    return ModelImageStatus::kOk;
}

ModelImageStatus PackedSection(
    const std::uint8_t* image,
    std::size_t image_bytes,
    const Section& section,
    PackedTernaryView* out) {
    if (out == nullptr) return ModelImageStatus::kNullArgument;
    if (section.offset > image_bytes || section.size > image_bytes - section.offset) {
        return ModelImageStatus::kInvalidSection;
    }
    const auto status = ParsePackedTernary(
        image + static_cast<std::size_t>(section.offset),
        static_cast<std::size_t>(section.size),
        out);
    return status == PackedTernaryStatus::kOk
        ? ModelImageStatus::kOk
        : ModelImageStatus::kInvalidPackedTernary;
}

Section ReadSection(const std::uint8_t* p) {
    Section s;
    s.type = ReadU32(p + 0);
    s.layer = ReadU32(p + 4);
    s.offset = ReadU64(p + 8);
    s.size = ReadU64(p + 16);
    s.flags = ReadU32(p + 24);
    s.reserved = ReadU32(p + 28);
    return s;
}

}  // namespace

ActivatedModelImage::~ActivatedModelImage() {
    if (mapping_base_ != nullptr && mapping_bytes_ != 0) {
        munmap(mapping_base_, mapping_bytes_);
    }
    if (fd_ >= 0) close(fd_);
}

ModelImageStatus ActivatedModelImage::OpenFd(
    int fd,
    std::uint64_t offset,
    std::uint64_t length,
    const std::uint8_t expected_model_id[kActivatedModelIdentityBytes],
    std::unique_ptr<ActivatedModelImage>* out) {
    if (out == nullptr || expected_model_id == nullptr) return ModelImageStatus::kNullArgument;
    out->reset();
    if (fd < 0) return ModelImageStatus::kInvalidFd;
    if (length == 0 || length > kActivatedModelMaxBytes ||
        offset % alignof(float) != 0 ||
        length > std::numeric_limits<std::size_t>::max()) {
        return length > kActivatedModelMaxBytes ? ModelImageStatus::kTooLarge : ModelImageStatus::kInvalidRange;
    }
    if (!AnyNonZero(expected_model_id, kActivatedModelIdentityBytes)) {
        return ModelImageStatus::kIdentityMismatch;
    }

    struct stat st{};
    if (fstat(fd, &st) != 0) return ModelImageStatus::kIoError;
    if (!S_ISREG(st.st_mode) || st.st_size < 0) return ModelImageStatus::kInvalidFd;
    const std::uint64_t file_size = static_cast<std::uint64_t>(st.st_size);
    if (offset > file_size || length > file_size - offset) return ModelImageStatus::kInvalidRange;

    const long page_size_long = sysconf(_SC_PAGESIZE);
    if (page_size_long <= 0) return ModelImageStatus::kIoError;
    const std::uint64_t page_size = static_cast<std::uint64_t>(page_size_long);
    const std::uint64_t map_offset = offset - (offset % page_size);
    const std::uint64_t delta = offset - map_offset;
    if (delta > std::numeric_limits<std::size_t>::max() ||
        length > std::numeric_limits<std::size_t>::max() - static_cast<std::size_t>(delta)) {
        return ModelImageStatus::kInvalidRange;
    }
    const std::size_t map_bytes = static_cast<std::size_t>(length + delta);

    const int owned_fd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
    if (owned_fd < 0) return ModelImageStatus::kIoError;
    void* mapping = mmap(
        nullptr,
        map_bytes,
        PROT_READ,
        MAP_PRIVATE,
        owned_fd,
        static_cast<off_t>(map_offset));
    if (mapping == MAP_FAILED) {
        close(owned_fd);
        return ModelImageStatus::kMapFailed;
    }

    std::unique_ptr<ActivatedModelImage> image(new (std::nothrow) ActivatedModelImage());
    if (!image) {
        munmap(mapping, map_bytes);
        close(owned_fd);
        return ModelImageStatus::kOutOfMemory;
    }
    image->fd_ = owned_fd;
    image->mapping_base_ = mapping;
    image->mapping_bytes_ = map_bytes;
    image->image_ = static_cast<const std::uint8_t*>(mapping) + static_cast<std::size_t>(delta);
    image->image_bytes_ = static_cast<std::size_t>(length);

    const auto status = image->Parse(expected_model_id);
    if (status != ModelImageStatus::kOk) return status;
    *out = std::move(image);
    return ModelImageStatus::kOk;
}

ModelImageStatus ActivatedModelImage::Parse(
    const std::uint8_t expected_model_id[kActivatedModelIdentityBytes]) {
    if (image_ == nullptr || image_bytes_ < kHeaderSize) return ModelImageStatus::kInvalidHeader;

    Sha256 hasher;
    hasher.Update(image_, image_bytes_);
    const auto digest = hasher.Final();
    if (!std::equal(digest.begin(), digest.end(), expected_model_id)) {
        return ModelImageStatus::kIdentityMismatch;
    }

    if (!std::equal(kMagic.begin(), kMagic.end(), image_)) return ModelImageStatus::kBadMagic;
    if (ReadU32(image_ + 8) != kVersion) return ModelImageStatus::kUnsupportedVersion;
    if (ReadU32(image_ + 12) != kHeaderSize) return ModelImageStatus::kInvalidHeader;

    const std::uint32_t flags = ReadU32(image_ + 16);
    const std::uint32_t section_count = ReadU32(image_ + 20);
    const std::uint32_t section_entry_size = ReadU32(image_ + 24);
    const std::uint32_t vocab_size = ReadU32(image_ + 28);
    const std::uint32_t d_model = ReadU32(image_ + 32);
    const std::uint32_t n_layers = ReadU32(image_ + 36);
    const std::uint32_t d_state = ReadU32(image_ + 40);
    const std::uint32_t embedding_rank = ReadU32(image_ + 44);
    const float dt_min = ReadF32(image_ + 48);
    const float dt_max = ReadF32(image_ + 52);
    const float rms_eps = ReadF32(image_ + 56);
    const std::uint32_t reserved = ReadU32(image_ + 60);
    const std::uint64_t table_offset = ReadU64(image_ + 64);
    const std::uint64_t payload_offset = ReadU64(image_ + 72);
    const std::uint64_t total_size = ReadU64(image_ + 80);
    const std::uint64_t reserved64 = ReadU64(image_ + 88);

    if ((flags & ~kKnownFlags) != 0 || reserved != 0 || reserved64 != 0 ||
        section_entry_size != kSectionEntrySize || section_count == 0 || section_count > kMaxSections ||
        table_offset != kHeaderSize || total_size != image_bytes_ ||
        vocab_size <= 1 || d_model == 0 || n_layers == 0 || n_layers > kMaxLayers || d_state == 0 ||
        !std::isfinite(dt_min) || !std::isfinite(dt_max) || !std::isfinite(rms_eps) ||
        !(dt_min > 0.0f) || !(dt_max > dt_min) || !(rms_eps > 0.0f)) {
        return ModelImageStatus::kInvalidConfig;
    }

    const bool factorized = (flags & kFlagFactorized) != 0;
    const bool tokenizer_flag = (flags & kFlagTokenizer) != 0;
    if ((!factorized && embedding_rank != 0) ||
        (factorized && (embedding_rank == 0 || embedding_rank >= d_model))) {
        return ModelImageStatus::kInvalidConfig;
    }

    std::size_t table_bytes = 0;
    if (MulOverflow(section_count, kSectionEntrySize, &table_bytes)) return ModelImageStatus::kInvalidHeader;
    std::size_t computed_payload = 0;
    if (AddOverflow(kHeaderSize, table_bytes, &computed_payload) || payload_offset != computed_payload ||
        payload_offset > image_bytes_ || payload_offset % alignof(float) != 0) {
        return ModelImageStatus::kInvalidHeader;
    }

    const std::uint32_t expected_sections =
        (factorized ? 2u : 1u) + n_layers * 8u + 1u + (tokenizer_flag ? 1u : 0u);
    if (section_count != expected_sections) return ModelImageStatus::kInvalidSection;

    std::size_t cursor = static_cast<std::size_t>(payload_offset);
    std::uint32_t section_index = 0;
    auto next = [&](std::uint32_t expected_type, std::uint32_t expected_layer, Section* out) -> ModelImageStatus {
        if (out == nullptr || section_index >= section_count) return ModelImageStatus::kInvalidSection;
        const std::size_t entry_off = kHeaderSize + static_cast<std::size_t>(section_index) * kSectionEntrySize;
        const Section s = ReadSection(image_ + entry_off);
        const std::size_t aligned = Align4(cursor);
        if (aligned < cursor || aligned > image_bytes_) return ModelImageStatus::kInvalidSection;
        if (aligned > cursor && !AllZero(image_ + cursor, aligned - cursor)) return ModelImageStatus::kInvalidSection;
        if (s.type != expected_type || s.layer != expected_layer || s.flags != 0 || s.reserved != 0 ||
            s.size == 0 || s.offset != aligned || s.offset > image_bytes_ || s.size > image_bytes_ - s.offset) {
            return ModelImageStatus::kInvalidSection;
        }
        cursor = static_cast<std::size_t>(s.offset + s.size);
        ++section_index;
        *out = s;
        return ModelImageStatus::kOk;
    };

    try {
        layers_.reset(new LanguageLayerView[n_layers]{});
    } catch (...) {
        return ModelImageStatus::kOutOfMemory;
    }

    std::size_t a_per_layer = 0;
    std::size_t a_total = 0;
    if (MulOverflow(d_model, d_state, &a_per_layer) || MulOverflow(a_per_layer, n_layers, &a_total)) {
        return ModelImageStatus::kInvalidConfig;
    }
    stable_a_.reset(new (std::nothrow) float[a_total]);
    if (!stable_a_) return ModelImageStatus::kOutOfMemory;

    language_model_ = {};
    std::copy(digest.begin(), digest.end(), language_model_.model_id);
    language_model_.vocab_size = vocab_size;
    language_model_.d_model = d_model;
    language_model_.n_layers = n_layers;
    language_model_.d_state = d_state;
    language_model_.embedding_kind = factorized ? LanguageEmbeddingKind::kFactorizedF32 : LanguageEmbeddingKind::kFullF32;
    language_model_.embedding_rank = embedding_rank;
    language_model_.dt_min = dt_min;
    language_model_.dt_max = dt_max;
    language_model_.rms_eps = rms_eps;
    language_model_.layers = layers_.get();

    Section section;
    ModelImageStatus status = ModelImageStatus::kOk;
    if (!factorized) {
        status = next(kEmbedding, kGlobalLayer, &section);
        if (status != ModelImageStatus::kOk) return status;
        std::size_t count = 0;
        if (MulOverflow(vocab_size, d_model, &count)) return ModelImageStatus::kInvalidConfig;
        status = F32Section(image_, image_bytes_, section, count, &language_model_.embedding);
        if (status != ModelImageStatus::kOk) return status;
    } else {
        status = next(kTokenFactors, kGlobalLayer, &section);
        if (status != ModelImageStatus::kOk) return status;
        std::size_t count = 0;
        if (MulOverflow(vocab_size, embedding_rank, &count)) return ModelImageStatus::kInvalidConfig;
        status = F32Section(image_, image_bytes_, section, count, &language_model_.token_factors);
        if (status != ModelImageStatus::kOk) return status;

        status = next(kEmbeddingProjection, kGlobalLayer, &section);
        if (status != ModelImageStatus::kOk) return status;
        if (MulOverflow(embedding_rank, d_model, &count)) return ModelImageStatus::kInvalidConfig;
        status = F32Section(image_, image_bytes_, section, count, &language_model_.projection);
        if (status != ModelImageStatus::kOk) return status;
    }

    for (std::uint32_t layer = 0; layer < n_layers; ++layer) {
        auto& dst = layers_[layer];
        status = next(kLayerNorm, layer, &section);
        if (status != ModelImageStatus::kOk) return status;
        status = F32Section(image_, image_bytes_, section, d_model, &dst.norm_weight);
        if (status != ModelImageStatus::kOk) return status;

        status = next(kInProj, layer, &section);
        if (status != ModelImageStatus::kOk) return status;
        status = PackedSection(image_, image_bytes_, section, &dst.in_proj);
        if (status != ModelImageStatus::kOk) return status;

        status = next(kDtProj, layer, &section);
        if (status != ModelImageStatus::kOk) return status;
        status = PackedSection(image_, image_bytes_, section, &dst.dt_proj);
        if (status != ModelImageStatus::kOk) return status;

        status = next(kDtBias, layer, &section);
        if (status != ModelImageStatus::kOk) return status;
        status = F32Section(image_, image_bytes_, section, d_model, &dst.dt_bias);
        if (status != ModelImageStatus::kOk) return status;

        status = next(kBProj, layer, &section);
        if (status != ModelImageStatus::kOk) return status;
        status = PackedSection(image_, image_bytes_, section, &dst.b_proj);
        if (status != ModelImageStatus::kOk) return status;

        status = next(kCProj, layer, &section);
        if (status != ModelImageStatus::kOk) return status;
        status = PackedSection(image_, image_bytes_, section, &dst.c_proj);
        if (status != ModelImageStatus::kOk) return status;

        status = next(kOutProj, layer, &section);
        if (status != ModelImageStatus::kOk) return status;
        status = PackedSection(image_, image_bytes_, section, &dst.out_proj);
        if (status != ModelImageStatus::kOk) return status;

        status = next(kALog, layer, &section);
        if (status != ModelImageStatus::kOk) return status;
        const float* a_log = nullptr;
        status = F32Section(image_, image_bytes_, section, a_per_layer, &a_log);
        if (status != ModelImageStatus::kOk) return status;
        float* a = stable_a_.get() + static_cast<std::size_t>(layer) * a_per_layer;
        for (std::size_t i = 0; i < a_per_layer; ++i) {
            if (!std::isfinite(a_log[i])) return ModelImageStatus::kInvalidModel;
            const double decay = std::exp(static_cast<double>(a_log[i]));
            if (!std::isfinite(decay) || decay <= 0.0 ||
                decay > static_cast<double>(std::numeric_limits<float>::max())) {
                return ModelImageStatus::kInvalidModel;
            }
            a[i] = -static_cast<float>(decay);
            if (!std::isfinite(a[i]) || !(a[i] < 0.0f)) return ModelImageStatus::kInvalidModel;
        }
        dst.a = a;
    }

    status = next(kFinalNorm, kGlobalLayer, &section);
    if (status != ModelImageStatus::kOk) return status;
    status = F32Section(image_, image_bytes_, section, d_model, &language_model_.final_norm_weight);
    if (status != ModelImageStatus::kOk) return status;

    if (tokenizer_flag) {
        status = next(kTokenizer, kGlobalLayer, &section);
        if (status != ModelImageStatus::kOk) return status;
        const auto tokenizer_status = ParseTokenizer(
            image_ + static_cast<std::size_t>(section.offset),
            static_cast<std::size_t>(section.size),
            &tokenizer_);
        if (tokenizer_status != TokenizerStatus::kOk) return ModelImageStatus::kInvalidTokenizer;
        const std::uint64_t tokenizer_vocab =
            static_cast<std::uint64_t>(kTokenizerLearnedBase) + tokenizer_.learned_count;
        if (tokenizer_vocab != vocab_size) return ModelImageStatus::kInvalidTokenizer;
        has_tokenizer_ = true;
    }

    if (section_index != section_count || cursor != image_bytes_) {
        return ModelImageStatus::kInvalidSection;
    }

    const auto model_status = ValidateLanguageModel(language_model_);
    if (model_status != LanguageStatus::kOk) return ModelImageStatus::kInvalidModel;
    return ModelImageStatus::kOk;
}

ActivatedModelInfo ActivatedModelImage::Info() const {
    ActivatedModelInfo info;
    std::copy(language_model_.model_id, language_model_.model_id + 32, info.model_id);
    info.vocab_size = language_model_.vocab_size;
    info.d_model = language_model_.d_model;
    info.n_layers = language_model_.n_layers;
    info.d_state = language_model_.d_state;
    info.embedding_kind = language_model_.embedding_kind;
    info.embedding_rank = language_model_.embedding_rank;
    info.has_tokenizer = has_tokenizer_;
    info.image_bytes = image_bytes_;
    return info;
}

}  // namespace vn97

namespace {
std::mutex g_model_registry_mutex;
std::unordered_map<std::uint64_t, std::shared_ptr<vn97::ActivatedModelImage>> g_models;
std::uint64_t g_next_model_handle = 1;

std::shared_ptr<vn97::ActivatedModelImage> LookupModel(std::uint64_t handle) {
    std::lock_guard<std::mutex> lock(g_model_registry_mutex);
    const auto it = g_models.find(handle);
    return it == g_models.end() ? nullptr : it->second;
}
}

extern "C" {

int vn97_model_open_fd(
    int fd,
    std::uint64_t offset,
    std::uint64_t length,
    const std::uint8_t* expected_model_id,
    std::size_t expected_model_id_size,
    std::uint64_t* handle_out) {
    if (expected_model_id == nullptr || handle_out == nullptr) {
        return static_cast<int>(vn97::ModelImageStatus::kNullArgument);
    }
    if (expected_model_id_size != vn97::kActivatedModelIdentityBytes) {
        return static_cast<int>(vn97::ModelImageStatus::kIdentityMismatch);
    }
    std::unique_ptr<vn97::ActivatedModelImage> opened;
    const auto status = vn97::ActivatedModelImage::OpenFd(
        fd, offset, length, expected_model_id, &opened);
    if (status != vn97::ModelImageStatus::kOk) return static_cast<int>(status);

    std::shared_ptr<vn97::ActivatedModelImage> shared(opened.release());
    std::lock_guard<std::mutex> lock(g_model_registry_mutex);
    if (g_next_model_handle == 0) return static_cast<int>(vn97::ModelImageStatus::kInvalidHandle);
    const std::uint64_t handle = g_next_model_handle++;
    g_models.emplace(handle, std::move(shared));
    *handle_out = handle;
    return static_cast<int>(vn97::ModelImageStatus::kOk);
}

int vn97_model_destroy(std::uint64_t handle) {
    std::lock_guard<std::mutex> lock(g_model_registry_mutex);
    return g_models.erase(handle) == 1
        ? static_cast<int>(vn97::ModelImageStatus::kOk)
        : static_cast<int>(vn97::ModelImageStatus::kInvalidHandle);
}

int vn97_model_info_get(std::uint64_t handle, vn97_model_info* out) {
    if (out == nullptr) return static_cast<int>(vn97::ModelImageStatus::kNullArgument);
    const auto model = LookupModel(handle);
    if (!model) return static_cast<int>(vn97::ModelImageStatus::kInvalidHandle);
    const auto info = model->Info();
    std::copy(info.model_id, info.model_id + 32, out->model_id);
    out->vocab_size = info.vocab_size;
    out->d_model = info.d_model;
    out->n_layers = info.n_layers;
    out->d_state = info.d_state;
    out->embedding_kind = static_cast<std::uint32_t>(info.embedding_kind);
    out->embedding_rank = info.embedding_rank;
    out->has_tokenizer = info.has_tokenizer ? 1 : 0;
    out->image_bytes = info.image_bytes;
    return static_cast<int>(vn97::ModelImageStatus::kOk);
}

int vn97_model_runtime_infer_step(
    std::uint64_t model_handle,
    std::uint64_t runtime_handle,
    const std::uint32_t* input_ids,
    std::size_t input_count,
    float* logits,
    std::size_t logits_count) {
    const auto model = LookupModel(model_handle);
    if (!model) return static_cast<int>(vn97::RuntimeStatus::kModelMismatch);
    return vn97_runtime_infer_step(
        runtime_handle,
        &model->language_model(),
        input_ids,
        input_count,
        logits,
        logits_count);
}

int vn97_model_runtime_prefill(
    std::uint64_t model_handle,
    std::uint64_t runtime_handle,
    const std::uint32_t* input_ids,
    std::size_t input_count,
    std::size_t step_count,
    float* final_logits,
    std::size_t logits_count) {
    const auto model = LookupModel(model_handle);
    if (!model) return static_cast<int>(vn97::RuntimeStatus::kModelMismatch);
    return vn97_runtime_prefill(
        runtime_handle,
        &model->language_model(),
        input_ids,
        input_count,
        step_count,
        final_logits,
        logits_count);
}

int vn97_model_runtime_generate_greedy(
    std::uint64_t model_handle,
    std::uint64_t runtime_handle,
    const std::uint32_t* prompt_ids,
    std::size_t prompt_count,
    std::size_t max_new_tokens,
    std::uint32_t eos_token,
    std::uint32_t* output_ids,
    std::size_t output_capacity,
    std::size_t* output_count) {
    const auto model = LookupModel(model_handle);
    if (!model) return static_cast<int>(vn97::RuntimeStatus::kModelMismatch);
    return vn97_runtime_generate_greedy(
        runtime_handle,
        &model->language_model(),
        prompt_ids,
        prompt_count,
        max_new_tokens,
        eos_token,
        output_ids,
        output_capacity,
        output_count);
}

static int MapTokenizerStatus(vn97::TokenizerStatus status) {
    if (status == vn97::TokenizerStatus::kOk) {
        return static_cast<int>(vn97::ModelImageStatus::kOk);
    }
    if (status == vn97::TokenizerStatus::kOutputTooSmall) {
        return static_cast<int>(vn97::ModelImageStatus::kOutputTooSmall);
    }
    return static_cast<int>(vn97::ModelImageStatus::kInvalidTokenizer);
}

int vn97_model_tokenizer_encode(
    std::uint64_t handle,
    const std::uint8_t* input,
    std::size_t input_size,
    std::uint32_t flags,
    std::uint32_t* output_ids,
    std::size_t output_capacity,
    std::size_t* output_count) {
    const auto model = LookupModel(handle);
    if (!model) return static_cast<int>(vn97::ModelImageStatus::kInvalidHandle);
    const auto* tokenizer = model->tokenizer();
    if (tokenizer == nullptr) return static_cast<int>(vn97::ModelImageStatus::kInvalidTokenizer);
    return MapTokenizerStatus(vn97::TokenizerEncodeBytes(
        *tokenizer, input, input_size, flags, output_ids, output_capacity, output_count));
}

int vn97_model_tokenizer_decoded_size(
    std::uint64_t handle,
    const std::uint32_t* token_ids,
    std::size_t token_count,
    int skip_control,
    std::size_t* output_size) {
    if (output_size == nullptr) return static_cast<int>(vn97::ModelImageStatus::kNullArgument);
    const auto model = LookupModel(handle);
    if (!model) return static_cast<int>(vn97::ModelImageStatus::kInvalidHandle);
    const auto* tokenizer = model->tokenizer();
    if (tokenizer == nullptr) return static_cast<int>(vn97::ModelImageStatus::kInvalidTokenizer);
    const auto status = vn97::TokenizerDecodeBytes(
        *tokenizer, token_ids, token_count, skip_control != 0, nullptr, 0, output_size);
    if (status == vn97::TokenizerStatus::kOk || status == vn97::TokenizerStatus::kOutputTooSmall) {
        return static_cast<int>(vn97::ModelImageStatus::kOk);
    }
    return static_cast<int>(vn97::ModelImageStatus::kInvalidTokenizer);
}

int vn97_model_tokenizer_decode(
    std::uint64_t handle,
    const std::uint32_t* token_ids,
    std::size_t token_count,
    int skip_control,
    std::uint8_t* output,
    std::size_t output_capacity,
    std::size_t* output_size) {
    const auto model = LookupModel(handle);
    if (!model) return static_cast<int>(vn97::ModelImageStatus::kInvalidHandle);
    const auto* tokenizer = model->tokenizer();
    if (tokenizer == nullptr) return static_cast<int>(vn97::ModelImageStatus::kInvalidTokenizer);
    return MapTokenizerStatus(vn97::TokenizerDecodeBytes(
        *tokenizer, token_ids, token_count, skip_control != 0, output, output_capacity, output_size));
}

}
