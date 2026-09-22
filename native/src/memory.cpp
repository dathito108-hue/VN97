#include "vn97/memory.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_set>

namespace vn97 {
namespace {

constexpr std::uint8_t kMagic[8] = {'V', 'N', '9', '7', 'M', 'E', 'M', '1'};
constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kHeaderSize = 16;
constexpr std::size_t kFramePrefixSize = 8;
constexpr std::size_t kRecordFixedSize = 76;
constexpr std::uint32_t kMaxVectorDim = 8192;
constexpr std::uint32_t kMaxSourceBytes = 1u << 20;
constexpr std::uint32_t kMaxContentBytes = 16u << 20;
constexpr std::uint32_t kMaxFrameBody = 64u << 20;

std::uint16_t ReadU16LE(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0]) |
        static_cast<std::uint16_t>(p[1] << 8);
}

std::uint32_t ReadU32LE(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) |
        (static_cast<std::uint32_t>(p[1]) << 8) |
        (static_cast<std::uint32_t>(p[2]) << 16) |
        (static_cast<std::uint32_t>(p[3]) << 24);
}

std::uint64_t ReadU64LE(const std::uint8_t* p) {
    return static_cast<std::uint64_t>(ReadU32LE(p)) |
        (static_cast<std::uint64_t>(ReadU32LE(p + 4)) << 32);
}

float ReadF32LE(const std::uint8_t* p) {
    const std::uint32_t bits = ReadU32LE(p);
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

bool AddOverflows(std::size_t a, std::size_t b) {
    return a > std::numeric_limits<std::size_t>::max() - b;
}

bool MulOverflows(std::size_t a, std::size_t b) {
    return b != 0 &&
        a > std::numeric_limits<std::size_t>::max() / b;
}

std::uint32_t Crc32(const std::uint8_t* data, std::size_t size) {
    std::uint32_t crc = 0xffffffffu;
    for (std::size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            const std::uint32_t mask =
                static_cast<std::uint32_t>(-
                    static_cast<std::int32_t>(crc & 1u));
            crc = (crc >> 1) ^ (0xedb88320u & mask);
        }
    }
    return ~crc;
}

bool ValidUtf8(const std::uint8_t* data, std::size_t size) {
    std::size_t i = 0;
    while (i < size) {
        const std::uint8_t c = data[i++];
        if (c <= 0x7fu) {
            continue;
        }

        std::uint32_t codepoint = 0;
        std::size_t continuation = 0;
        std::uint32_t min_value = 0;
        if ((c & 0xe0u) == 0xc0u) {
            codepoint = c & 0x1fu;
            continuation = 1;
            min_value = 0x80u;
        } else if ((c & 0xf0u) == 0xe0u) {
            codepoint = c & 0x0fu;
            continuation = 2;
            min_value = 0x800u;
        } else if ((c & 0xf8u) == 0xf0u) {
            codepoint = c & 0x07u;
            continuation = 3;
            min_value = 0x10000u;
        } else {
            return false;
        }

        if (i + continuation > size) {
            return false;
        }
        for (std::size_t j = 0; j < continuation; ++j) {
            const std::uint8_t next = data[i++];
            if ((next & 0xc0u) != 0x80u) {
                return false;
            }
            codepoint = (codepoint << 6) | (next & 0x3fu);
        }

        if (
            codepoint < min_value ||
            codepoint > 0x10ffffu ||
            (codepoint >= 0xd800u && codepoint <= 0xdfffu)
        ) {
            return false;
        }
    }
    return true;
}

MemoryStatus ValidateRecord(
    const std::uint8_t* body,
    std::size_t body_size,
    std::uint32_t vector_dim,
    std::uint64_t previous_id,
    std::uint64_t* record_id_out) {
    if (body_size < kRecordFixedSize) {
        return MemoryStatus::kInvalidRecord;
    }

    const std::uint64_t record_id = ReadU64LE(body + 0);
    const std::uint8_t kind = body[16];
    const std::uint8_t flags = body[17];
    const std::uint16_t reserved = ReadU16LE(body + 18);
    const float importance = ReadF32LE(body + 20);
    const std::uint32_t source_len = ReadU32LE(body + 24);
    const std::uint32_t content_len = ReadU32LE(body + 28);
    const std::uint32_t vector_count = ReadU32LE(body + 32);
    const std::uint64_t parent_id = ReadU64LE(body + 36);

    if (record_id == 0 || record_id <= previous_id) {
        return MemoryStatus::kInvalidRecord;
    }
    if (parent_id != 0 && parent_id >= record_id) {
        return MemoryStatus::kInvalidRecord;
    }
    if (kind != 1u && kind != 2u) {
        return MemoryStatus::kInvalidRecord;
    }
    if (flags != 0u || reserved != 0u) {
        return MemoryStatus::kInvalidRecord;
    }
    if (
        !std::isfinite(importance) ||
        importance < 0.0f ||
        importance > 1.0f
    ) {
        return MemoryStatus::kInvalidRecord;
    }
    if (
        source_len > kMaxSourceBytes ||
        content_len > kMaxContentBytes
    ) {
        return MemoryStatus::kInvalidRecord;
    }
    if (vector_count != 0u && vector_count != vector_dim) {
        return MemoryStatus::kInvalidRecord;
    }
    if (MulOverflows(vector_count, 4u)) {
        return MemoryStatus::kInvalidRecord;
    }

    std::size_t expected = kRecordFixedSize;
    if (AddOverflows(expected, source_len)) {
        return MemoryStatus::kInvalidRecord;
    }
    expected += source_len;
    if (AddOverflows(expected, content_len)) {
        return MemoryStatus::kInvalidRecord;
    }
    expected += content_len;
    const std::size_t vector_bytes =
        static_cast<std::size_t>(vector_count) * 4u;
    if (AddOverflows(expected, vector_bytes)) {
        return MemoryStatus::kInvalidRecord;
    }
    expected += vector_bytes;
    if (expected != body_size) {
        return MemoryStatus::kInvalidRecord;
    }

    const std::uint8_t* source =
        body + kRecordFixedSize;
    const std::uint8_t* content =
        source + source_len;
    const std::uint8_t* vector =
        content + content_len;
    if (
        !ValidUtf8(source, source_len) ||
        !ValidUtf8(content, content_len)
    ) {
        return MemoryStatus::kInvalidRecord;
    }
    for (std::uint32_t i = 0; i < vector_count; ++i) {
        if (
            !std::isfinite(
                ReadF32LE(
                    vector +
                    static_cast<std::size_t>(i) * 4u
                )
            )
        ) {
            return MemoryStatus::kInvalidRecord;
        }
    }

    *record_id_out = record_id;
    return MemoryStatus::kOk;
}

}  // namespace

MemoryStatus ScanMemoryJournal(
    const std::uint8_t* blob,
    std::size_t blob_size,
    bool allow_torn_tail,
    MemoryScanResult* out) {
    if (blob == nullptr || out == nullptr) {
        return MemoryStatus::kNullArgument;
    }
    if (blob_size < kHeaderSize) {
        return MemoryStatus::kBlobTooShort;
    }
    if (std::memcmp(blob, kMagic, sizeof(kMagic)) != 0) {
        return MemoryStatus::kBadMagic;
    }
    if (ReadU32LE(blob + 8) != kVersion) {
        return MemoryStatus::kUnsupportedVersion;
    }

    const std::uint32_t vector_dim =
        ReadU32LE(blob + 12);
    if (vector_dim > kMaxVectorDim) {
        return MemoryStatus::kInvalidHeader;
    }

    MemoryScanResult result;
    result.vector_dim = vector_dim;
    result.valid_bytes = kHeaderSize;
    std::uint64_t previous_id = 0;
    std::unordered_set<std::uint64_t> seen_ids;

    while (result.valid_bytes < blob_size) {
        const std::size_t frame_start =
            result.valid_bytes;
        const std::size_t remaining =
            blob_size - frame_start;

        if (remaining < kFramePrefixSize) {
            if (!allow_torn_tail) {
                return MemoryStatus::kTruncatedTail;
            }
            result.tail_truncated = true;
            break;
        }

        const std::uint8_t* prefix =
            blob + frame_start;
        const std::uint32_t body_size =
            ReadU32LE(prefix);
        const std::uint32_t expected_crc =
            ReadU32LE(prefix + 4);

        if (
            body_size < kRecordFixedSize ||
            body_size > kMaxFrameBody
        ) {
            return MemoryStatus::kInvalidFrame;
        }
        if (body_size > remaining - kFramePrefixSize) {
            if (!allow_torn_tail) {
                return MemoryStatus::kTruncatedTail;
            }
            result.tail_truncated = true;
            break;
        }

        const std::uint8_t* body =
            prefix + kFramePrefixSize;
        if (Crc32(body, body_size) != expected_crc) {
            return MemoryStatus::kChecksumMismatch;
        }

        std::uint64_t record_id = 0;
        const MemoryStatus record_status =
            ValidateRecord(
                body,
                body_size,
                vector_dim,
                previous_id,
                &record_id
            );
        if (record_status != MemoryStatus::kOk) {
            return record_status;
        }

        const std::uint64_t parent_id =
            ReadU64LE(body + 36);
        if (
            parent_id != 0 &&
            seen_ids.find(parent_id) == seen_ids.end()
        ) {
            return MemoryStatus::kInvalidRecord;
        }

        seen_ids.insert(record_id);
        previous_id = record_id;
        result.last_record_id = record_id;
        ++result.record_count;
        result.valid_bytes =
            frame_start + kFramePrefixSize + body_size;
    }

    *out = result;
    return MemoryStatus::kOk;
}

}  // namespace vn97

extern "C" {

int vn97_memory_scan(
    const std::uint8_t* blob,
    std::size_t blob_size,
    int allow_torn_tail,
    std::uint32_t* vector_dim,
    std::size_t* record_count,
    std::uint64_t* last_record_id,
    std::size_t* valid_bytes,
    int* tail_truncated) {
    if (
        vector_dim == nullptr ||
        record_count == nullptr ||
        last_record_id == nullptr ||
        valid_bytes == nullptr ||
        tail_truncated == nullptr
    ) {
        return static_cast<int>(
            vn97::MemoryStatus::kNullArgument
        );
    }

    vn97::MemoryScanResult result;
    const vn97::MemoryStatus status =
        vn97::ScanMemoryJournal(
            blob,
            blob_size,
            allow_torn_tail != 0,
            &result
        );
    if (status != vn97::MemoryStatus::kOk) {
        return static_cast<int>(status);
    }

    *vector_dim = result.vector_dim;
    *record_count = result.record_count;
    *last_record_id = result.last_record_id;
    *valid_bytes = result.valid_bytes;
    *tail_truncated =
        result.tail_truncated ? 1 : 0;
    return 0;
}

}
