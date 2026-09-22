#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

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
    kInvalidParameter,
    kOutputTooSmall,
    kIoError,
    kLocked,
};

enum class MemoryKind : std::uint8_t {
    kEpisodic = 1,
    kSemantic = 2,
};

struct MemoryScanResult {
    std::uint32_t vector_dim = 0;
    std::size_t record_count = 0;
    std::uint64_t last_record_id = 0;
    std::size_t valid_bytes = 0;
    bool tail_truncated = false;
};

struct MemoryIndexEntry {
    std::uint64_t record_id = 0;
    std::uint64_t timestamp_ns = 0;
    std::uint64_t parent_id = 0;
    std::uint64_t frame_offset = 0;
    std::uint64_t frame_size = 0;
    std::uint64_t source_offset = 0;
    std::uint64_t content_offset = 0;
    std::uint64_t vector_offset = 0;
    std::uint32_t source_size = 0;
    std::uint32_t content_size = 0;
    std::uint32_t vector_count = 0;
    float importance = 0.0f;
    float inverse_vector_norm = 0.0f;
    MemoryKind kind = MemoryKind::kEpisodic;
};

struct MemoryIndex {
    std::uint32_t vector_dim = 0;
    std::size_t indexed_bytes = 0;
    bool tail_truncated = false;
    std::vector<MemoryIndexEntry> entries;
};

struct MemoryRetrievalOptions {
    std::size_t top_k = 5;
    std::uint32_t kind_mask = 0x3u;
    float semantic_weight = 1.0f;
    float recency_weight = 0.0f;
    float importance_weight = 0.0f;
    std::uint64_t now_ns = 0;
    std::uint64_t recency_half_life_ns = 86400000000000ull;
};

struct MemoryHit {
    std::uint64_t record_id = 0;
    float score = 0.0f;
    float semantic_score = 0.0f;
    float recency_score = 0.0f;
    float importance_score = 0.0f;
};

struct MemoryAppendInput {
    MemoryKind kind = MemoryKind::kEpisodic;
    std::uint64_t timestamp_ns = 0;
    float importance = 0.5f;
    const std::uint8_t* source = nullptr;
    std::size_t source_size = 0;
    const std::uint8_t* content = nullptr;
    std::size_t content_size = 0;
    const float* vector = nullptr;
    std::size_t vector_count = 0;
    std::uint64_t parent_id = 0;
    bool durable = true;
};

struct MemoryRetentionPolicy {
    std::size_t max_records = 0;
    std::uint64_t max_age_ns = 0;
    float min_importance = 0.0f;
};

struct MemoryRecordView {
    std::uint64_t record_id = 0;
    std::uint64_t timestamp_ns = 0;
    std::uint64_t parent_id = 0;
    MemoryKind kind = MemoryKind::kEpisodic;
    float importance = 0.0f;
    const std::uint8_t* source = nullptr;
    std::size_t source_size = 0;
    const std::uint8_t* content = nullptr;
    std::size_t content_size = 0;
    const std::uint8_t* vector_f32_le = nullptr;
    std::size_t vector_count = 0;
};

MemoryStatus ScanMemoryJournal(
    const std::uint8_t* blob,
    std::size_t blob_size,
    bool allow_torn_tail,
    MemoryScanResult* out);

MemoryStatus BuildMemoryIndex(
    const std::uint8_t* blob,
    std::size_t blob_size,
    bool allow_torn_tail,
    MemoryIndex* out);

MemoryStatus ExtendMemoryIndex(
    const std::uint8_t* blob,
    std::size_t blob_size,
    bool allow_torn_tail,
    MemoryIndex* index);

MemoryStatus RetrieveMemory(
    const std::uint8_t* blob,
    std::size_t blob_size,
    const MemoryIndex& index,
    const float* query,
    std::size_t query_count,
    const MemoryRetrievalOptions& options,
    std::vector<MemoryHit>* out);

MemoryStatus ViewMemoryRecord(
    const std::uint8_t* blob,
    std::size_t blob_size,
    const MemoryIndex& index,
    std::uint64_t record_id,
    MemoryRecordView* out);

class MemoryStore {
public:
    ~MemoryStore();
    MemoryStore(const MemoryStore&) = delete;
    MemoryStore& operator=(const MemoryStore&) = delete;

    static MemoryStatus Create(
        const char* path,
        std::uint32_t vector_dim,
        bool overwrite,
        MemoryStore** out);

    static MemoryStatus Open(
        const char* path,
        bool recover_torn_tail,
        MemoryStore** out);

    MemoryStatus Append(
        const MemoryAppendInput& input,
        std::uint64_t* record_id_out);

    MemoryStatus Retrieve(
        const float* query,
        std::size_t query_count,
        const MemoryRetrievalOptions& options,
        std::vector<MemoryHit>* out) const;

    MemoryStatus ViewRecord(
        std::uint64_t record_id,
        MemoryRecordView* out) const;

    MemoryStatus Compact(
        const MemoryRetentionPolicy& policy,
        std::uint64_t now_ns,
        std::size_t* retained_out,
        std::size_t* removed_out);

    const MemoryIndex& index() const { return index_; }
    std::uint32_t vector_dim() const { return index_.vector_dim; }
    std::size_t record_count() const { return index_.entries.size(); }
    std::uint64_t last_record_id() const;
    std::size_t valid_bytes() const { return index_.indexed_bytes; }

private:
    MemoryStore() = default;

    MemoryStatus MapAndIndex(bool recover_torn_tail);
    MemoryStatus RemapAndExtend(std::size_t previous_bytes);
    void Unmap();

    int fd_ = -1;
    std::uint8_t* mapping_ = nullptr;
    std::size_t mapping_size_ = 0;
    MemoryIndex index_;
    std::vector<char> path_;
};

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

int vn97_memory_store_create(
    const char* path,
    std::uint32_t vector_dim,
    int overwrite,
    void** out_store);

int vn97_memory_store_open(
    const char* path,
    int recover_torn_tail,
    void** out_store);

void vn97_memory_store_close(void* store);

int vn97_memory_store_append(
    void* store,
    std::uint8_t kind,
    std::uint64_t timestamp_ns,
    float importance,
    const std::uint8_t* source,
    std::size_t source_size,
    const std::uint8_t* content,
    std::size_t content_size,
    const float* vector,
    std::size_t vector_count,
    std::uint64_t parent_id,
    int durable,
    std::uint64_t* record_id_out);

int vn97_memory_store_retrieve(
    void* store,
    const float* query,
    std::size_t query_count,
    std::size_t top_k,
    std::uint32_t kind_mask,
    float semantic_weight,
    float recency_weight,
    float importance_weight,
    std::uint64_t now_ns,
    std::uint64_t recency_half_life_ns,
    std::uint64_t* record_ids,
    float* scores,
    float* semantic_scores,
    float* recency_scores,
    float* importance_scores,
    std::size_t output_capacity,
    std::size_t* output_count);

int vn97_memory_store_compact(
    void* store,
    std::size_t max_records,
    std::uint64_t max_age_ns,
    float min_importance,
    std::uint64_t now_ns,
    std::size_t* retained_out,
    std::size_t* removed_out);

int vn97_memory_store_stats(
    void* store,
    std::uint32_t* vector_dim,
    std::size_t* record_count,
    std::uint64_t* last_record_id,
    std::size_t* valid_bytes);

}
