#pragma once

#include <cstddef>
#include <cstdint>

namespace vn97 {

enum class MemoryStatus {
    kOk = 0,
    kNullArgument,
    kBlobTooShort,
    kBadMagic,
    kUnsupportedVersion,
    kInvalidHeader,
    kInvalidFrame,
    kChecksumMismatch,
    kInvalidRecord,
    kTruncatedTail,
};

struct MemoryScanResult {
    std::uint32_t vector_dim = 0;
    std::size_t record_count = 0;
    std::uint64_t last_record_id = 0;
    std::size_t valid_bytes = 0;
    bool tail_truncated = false;
};

MemoryStatus ScanMemoryJournal(
    const std::uint8_t* blob,
    std::size_t blob_size,
    bool allow_torn_tail,
    MemoryScanResult* out);

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
    int* tail_truncated);

}
