#pragma once

#include "vn97/memory.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace vn97::memory_internal {

inline constexpr std::uint8_t kMagic[8] = {
    'V', 'N', '9', '7', 'M', 'E', 'M', '1'
};
inline constexpr std::uint32_t kVersion = 1;
inline constexpr std::size_t kHeaderSize = 16;
inline constexpr std::size_t kFramePrefixSize = 8;
inline constexpr std::size_t kRecordFixedSize = 76;
inline constexpr std::uint32_t kMaxVectorDim = 8192;
inline constexpr std::uint32_t kMaxSourceBytes = 1u << 20;
inline constexpr std::uint32_t kMaxContentBytes = 16u << 20;
inline constexpr std::uint32_t kMaxFrameBody = 64u << 20;
inline constexpr double kLn2 = 0.693147180559945309417232121458176568;

inline std::uint16_t ReadU16LE(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0]) |
        static_cast<std::uint16_t>(static_cast<std::uint16_t>(p[1]) << 8);
}

inline std::uint32_t ReadU32LE(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) |
        (static_cast<std::uint32_t>(p[1]) << 8) |
        (static_cast<std::uint32_t>(p[2]) << 16) |
        (static_cast<std::uint32_t>(p[3]) << 24);
}

inline std::uint64_t ReadU64LE(const std::uint8_t* p) {
    return static_cast<std::uint64_t>(ReadU32LE(p)) |
        (static_cast<std::uint64_t>(ReadU32LE(p + 4)) << 32);
}

inline float ReadF32LE(const std::uint8_t* p) {
    const std::uint32_t bits = ReadU32LE(p);
    float value = 0.0f;
    static_assert(sizeof(value) == sizeof(bits), "float32 required");
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline void WriteU16LE(std::uint8_t* p, std::uint16_t value) {
    p[0] = static_cast<std::uint8_t>(value);
    p[1] = static_cast<std::uint8_t>(value >> 8);
}

inline void WriteU32LE(std::uint8_t* p, std::uint32_t value) {
    p[0] = static_cast<std::uint8_t>(value);
    p[1] = static_cast<std::uint8_t>(value >> 8);
    p[2] = static_cast<std::uint8_t>(value >> 16);
    p[3] = static_cast<std::uint8_t>(value >> 24);
}

inline void WriteU64LE(std::uint8_t* p, std::uint64_t value) {
    WriteU32LE(p, static_cast<std::uint32_t>(value));
    WriteU32LE(p + 4, static_cast<std::uint32_t>(value >> 32));
}

inline void WriteF32LE(std::uint8_t* p, float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    WriteU32LE(p, bits);
}

inline bool AddOverflows(std::size_t a, std::size_t b) {
    return a > std::numeric_limits<std::size_t>::max() - b;
}

inline bool MulOverflows(std::size_t a, std::size_t b) {
    return b != 0 &&
        a > std::numeric_limits<std::size_t>::max() / b;
}

inline std::uint32_t Crc32(const std::uint8_t* data, std::size_t size) {
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

inline bool ValidUtf8(const std::uint8_t* data, std::size_t size) {
    if (size != 0 && data == nullptr) {
        return false;
    }
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

std::array<std::uint8_t, 32> Sha256(
    const std::uint8_t* data,
    std::size_t size);

inline MemoryStatus ValidateHeader(
    const std::uint8_t* blob,
    std::size_t blob_size,
    std::uint32_t* vector_dim) {
    if (blob == nullptr || vector_dim == nullptr) {
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
    const std::uint32_t dim = ReadU32LE(blob + 12);
    if (dim > kMaxVectorDim) {
        return MemoryStatus::kInvalidHeader;
    }
    *vector_dim = dim;
    return MemoryStatus::kOk;
}

inline bool HasRecordId(
    const std::vector<MemoryIndexEntry>& entries,
    std::uint64_t record_id) {
    const auto it = std::lower_bound(
        entries.begin(),
        entries.end(),
        record_id,
        [](const MemoryIndexEntry& entry, std::uint64_t value) {
            return entry.record_id < value;
        });
    return it != entries.end() && it->record_id == record_id;
}

}  // namespace vn97::memory_internal
