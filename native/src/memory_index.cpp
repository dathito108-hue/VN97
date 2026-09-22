#include "memory_engine_internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

namespace vn97 {
namespace {

using namespace memory_internal;

MemoryStatus ParseFrame(
    const std::uint8_t* blob,
    std::size_t blob_size,
    std::size_t frame_offset,
    std::uint32_t vector_dim,
    std::uint64_t previous_id,
    const std::vector<MemoryIndexEntry>& existing_entries,
    bool allow_torn_tail,
    MemoryIndexEntry* entry,
    std::size_t* next_offset,
    bool* tail_truncated) {
    if (entry == nullptr || next_offset == nullptr || tail_truncated == nullptr) {
        return MemoryStatus::kNullArgument;
    }
    *tail_truncated = false;
    if (frame_offset > blob_size) {
        return MemoryStatus::kInvalidFrame;
    }
    const std::size_t remaining = blob_size - frame_offset;
    if (remaining < kFramePrefixSize) {
        if (allow_torn_tail) {
            *tail_truncated = true;
            *next_offset = frame_offset;
            return MemoryStatus::kOk;
        }
        return MemoryStatus::kTruncatedTail;
    }

    const std::uint8_t* prefix = blob + frame_offset;
    const std::uint32_t body_size = ReadU32LE(prefix);
    const std::uint32_t expected_crc = ReadU32LE(prefix + 4);
    if (body_size < kRecordFixedSize || body_size > kMaxFrameBody) {
        return MemoryStatus::kInvalidFrame;
    }
    if (body_size > remaining - kFramePrefixSize) {
        if (allow_torn_tail) {
            *tail_truncated = true;
            *next_offset = frame_offset;
            return MemoryStatus::kOk;
        }
        return MemoryStatus::kTruncatedTail;
    }

    const std::uint8_t* body = prefix + kFramePrefixSize;
    if (Crc32(body, body_size) != expected_crc) {
        return MemoryStatus::kChecksumMismatch;
    }

    const std::uint64_t record_id = ReadU64LE(body + 0);
    const std::uint64_t timestamp_ns = ReadU64LE(body + 8);
    const std::uint8_t kind_raw = body[16];
    const std::uint8_t flags = body[17];
    const std::uint16_t reserved = ReadU16LE(body + 18);
    const float importance = ReadF32LE(body + 20);
    const std::uint32_t source_size = ReadU32LE(body + 24);
    const std::uint32_t content_size = ReadU32LE(body + 28);
    const std::uint32_t vector_count = ReadU32LE(body + 32);
    const std::uint64_t parent_id = ReadU64LE(body + 36);
    const std::uint8_t* content_digest = body + 44;

    if (record_id == 0 || record_id <= previous_id) {
        return MemoryStatus::kInvalidRecord;
    }
    if (parent_id != 0 && parent_id >= record_id) {
        return MemoryStatus::kInvalidRecord;
    }
    if (kind_raw != 1u && kind_raw != 2u) {
        return MemoryStatus::kInvalidRecord;
    }
    if (flags != 0u || reserved != 0u) {
        return MemoryStatus::kInvalidRecord;
    }
    if (!std::isfinite(importance) || importance < 0.0f || importance > 1.0f) {
        return MemoryStatus::kInvalidRecord;
    }
    if (source_size > kMaxSourceBytes || content_size > kMaxContentBytes) {
        return MemoryStatus::kInvalidRecord;
    }
    if (vector_count != 0u && vector_count != vector_dim) {
        return MemoryStatus::kInvalidRecord;
    }
    if (MulOverflows(vector_count, 4u)) {
        return MemoryStatus::kInvalidRecord;
    }

    std::size_t expected_size = kRecordFixedSize;
    if (AddOverflows(expected_size, source_size)) {
        return MemoryStatus::kInvalidRecord;
    }
    expected_size += source_size;
    if (AddOverflows(expected_size, content_size)) {
        return MemoryStatus::kInvalidRecord;
    }
    expected_size += content_size;
    const std::size_t vector_bytes = static_cast<std::size_t>(vector_count) * 4u;
    if (AddOverflows(expected_size, vector_bytes)) {
        return MemoryStatus::kInvalidRecord;
    }
    expected_size += vector_bytes;
    if (expected_size != body_size) {
        return MemoryStatus::kInvalidRecord;
    }

    const std::uint8_t* source = body + kRecordFixedSize;
    const std::uint8_t* content = source + source_size;
    const std::uint8_t* vector = content + content_size;
    if (!ValidUtf8(source, source_size) || !ValidUtf8(content, content_size)) {
        return MemoryStatus::kInvalidRecord;
    }
    const auto actual_digest = Sha256(content, content_size);
    if (std::memcmp(actual_digest.data(), content_digest, actual_digest.size()) != 0) {
        return MemoryStatus::kInvalidRecord;
    }
    if (parent_id != 0 && !HasRecordId(existing_entries, parent_id)) {
        return MemoryStatus::kInvalidRecord;
    }

    double norm_square = 0.0;
    for (std::uint32_t i = 0; i < vector_count; ++i) {
        const float value = ReadF32LE(vector + static_cast<std::size_t>(i) * 4u);
        if (!std::isfinite(value)) {
            return MemoryStatus::kInvalidRecord;
        }
        norm_square += static_cast<double>(value) * static_cast<double>(value);
    }

    MemoryIndexEntry parsed;
    parsed.record_id = record_id;
    parsed.timestamp_ns = timestamp_ns;
    parsed.parent_id = parent_id;
    parsed.frame_offset = frame_offset;
    parsed.frame_size = kFramePrefixSize + body_size;
    parsed.source_offset = frame_offset + kFramePrefixSize + kRecordFixedSize;
    parsed.content_offset = parsed.source_offset + source_size;
    parsed.vector_offset = parsed.content_offset + content_size;
    parsed.source_size = source_size;
    parsed.content_size = content_size;
    parsed.vector_count = vector_count;
    parsed.importance = importance;
    parsed.inverse_vector_norm =
        norm_square > 0.0 ? static_cast<float>(1.0 / std::sqrt(norm_square)) : 0.0f;
    parsed.kind = static_cast<MemoryKind>(kind_raw);

    *entry = parsed;
    *next_offset = frame_offset + parsed.frame_size;
    return MemoryStatus::kOk;
}

}  // namespace

MemoryStatus BuildMemoryIndex(
    const std::uint8_t* blob,
    std::size_t blob_size,
    bool allow_torn_tail,
    MemoryIndex* out) {
    if (out == nullptr) {
        return MemoryStatus::kNullArgument;
    }
    std::uint32_t vector_dim = 0;
    const MemoryStatus header_status =
        memory_internal::ValidateHeader(blob, blob_size, &vector_dim);
    if (header_status != MemoryStatus::kOk) {
        return header_status;
    }

    MemoryIndex index;
    index.vector_dim = vector_dim;
    index.indexed_bytes = memory_internal::kHeaderSize;
    while (index.indexed_bytes < blob_size) {
        MemoryIndexEntry entry;
        std::size_t next_offset = index.indexed_bytes;
        bool tail_truncated = false;
        const std::uint64_t previous_id =
            index.entries.empty() ? 0 : index.entries.back().record_id;
        const MemoryStatus status = ParseFrame(
            blob,
            blob_size,
            index.indexed_bytes,
            vector_dim,
            previous_id,
            index.entries,
            allow_torn_tail,
            &entry,
            &next_offset,
            &tail_truncated);
        if (status != MemoryStatus::kOk) {
            return status;
        }
        if (tail_truncated) {
            index.tail_truncated = true;
            break;
        }
        index.entries.push_back(entry);
        index.indexed_bytes = next_offset;
    }

    *out = std::move(index);
    return MemoryStatus::kOk;
}

MemoryStatus ExtendMemoryIndex(
    const std::uint8_t* blob,
    std::size_t blob_size,
    bool allow_torn_tail,
    MemoryIndex* index) {
    if (blob == nullptr || index == nullptr) {
        return MemoryStatus::kNullArgument;
    }
    std::uint32_t vector_dim = 0;
    const MemoryStatus header_status =
        memory_internal::ValidateHeader(blob, blob_size, &vector_dim);
    if (header_status != MemoryStatus::kOk) {
        return header_status;
    }
    if (
        vector_dim != index->vector_dim ||
        index->indexed_bytes < memory_internal::kHeaderSize ||
        index->indexed_bytes > blob_size
    ) {
        return MemoryStatus::kInvalidParameter;
    }

    MemoryIndex updated = *index;
    updated.tail_truncated = false;
    while (updated.indexed_bytes < blob_size) {
        MemoryIndexEntry entry;
        std::size_t next_offset = updated.indexed_bytes;
        bool tail_truncated = false;
        const std::uint64_t previous_id =
            updated.entries.empty() ? 0 : updated.entries.back().record_id;
        const MemoryStatus status = ParseFrame(
            blob,
            blob_size,
            updated.indexed_bytes,
            vector_dim,
            previous_id,
            updated.entries,
            allow_torn_tail,
            &entry,
            &next_offset,
            &tail_truncated);
        if (status != MemoryStatus::kOk) {
            return status;
        }
        if (tail_truncated) {
            updated.tail_truncated = true;
            break;
        }
        updated.entries.push_back(entry);
        updated.indexed_bytes = next_offset;
    }
    *index = std::move(updated);
    return MemoryStatus::kOk;
}

MemoryStatus RetrieveMemory(
    const std::uint8_t* blob,
    std::size_t blob_size,
    const MemoryIndex& index,
    const float* query,
    std::size_t query_count,
    const MemoryRetrievalOptions& options,
    std::vector<MemoryHit>* out) {
    if (blob == nullptr || query == nullptr || out == nullptr) {
        return MemoryStatus::kNullArgument;
    }
    if (
        query_count != index.vector_dim ||
        query_count == 0 ||
        index.indexed_bytes > blob_size ||
        options.top_k == 0 ||
        options.kind_mask == 0u ||
        options.recency_half_life_ns == 0
    ) {
        return MemoryStatus::kInvalidParameter;
    }
    if (
        !std::isfinite(options.semantic_weight) || options.semantic_weight < 0.0f ||
        !std::isfinite(options.recency_weight) || options.recency_weight < 0.0f ||
        !std::isfinite(options.importance_weight) || options.importance_weight < 0.0f ||
        options.semantic_weight + options.recency_weight +
                options.importance_weight <= 0.0f
    ) {
        return MemoryStatus::kInvalidParameter;
    }

    double query_norm_square = 0.0;
    for (std::size_t i = 0; i < query_count; ++i) {
        if (!std::isfinite(query[i])) {
            return MemoryStatus::kInvalidParameter;
        }
        query_norm_square +=
            static_cast<double>(query[i]) * static_cast<double>(query[i]);
    }
    if (query_norm_square == 0.0) {
        return MemoryStatus::kInvalidParameter;
    }
    const double query_inverse_norm = 1.0 / std::sqrt(query_norm_square);

    std::uint64_t now_ns = options.now_ns;
    if (now_ns == 0) {
        for (const auto& entry : index.entries) {
            now_ns = std::max(now_ns, entry.timestamp_ns);
        }
    }

    struct RankedHit {
        MemoryHit hit;
        std::uint64_t timestamp_ns;
    };
    std::vector<RankedHit> ranked;
    ranked.reserve(index.entries.size());

    for (const auto& entry : index.entries) {
        const std::uint32_t kind_bit =
            entry.kind == MemoryKind::kEpisodic ? 0x1u : 0x2u;
        if (
            (options.kind_mask & kind_bit) == 0u ||
            entry.vector_count != index.vector_dim
        ) {
            continue;
        }
        if (
            entry.vector_offset > blob_size ||
            static_cast<std::size_t>(entry.vector_count) >
                (blob_size - static_cast<std::size_t>(entry.vector_offset)) / 4u
        ) {
            return MemoryStatus::kInvalidRecord;
        }

        double dot = 0.0;
        const std::uint8_t* vector =
            blob + static_cast<std::size_t>(entry.vector_offset);
        for (std::size_t i = 0; i < query_count; ++i) {
            const float value = memory_internal::ReadF32LE(vector + i * 4u);
            dot += static_cast<double>(query[i]) * static_cast<double>(value);
        }

        double semantic = 0.0;
        if (entry.inverse_vector_norm > 0.0f) {
            semantic =
                dot * query_inverse_norm *
                static_cast<double>(entry.inverse_vector_norm);
            semantic = std::max(-1.0, std::min(1.0, semantic));
        }
        const std::uint64_t age =
            now_ns > entry.timestamp_ns ? now_ns - entry.timestamp_ns : 0;
        const double recency = std::exp(
            -memory_internal::kLn2 * static_cast<double>(age) /
            static_cast<double>(options.recency_half_life_ns));
        const double score =
            static_cast<double>(options.semantic_weight) * semantic +
            static_cast<double>(options.recency_weight) * recency +
            static_cast<double>(options.importance_weight) * entry.importance;

        RankedHit ranked_hit;
        ranked_hit.hit.record_id = entry.record_id;
        ranked_hit.hit.score = static_cast<float>(score);
        ranked_hit.hit.semantic_score = static_cast<float>(semantic);
        ranked_hit.hit.recency_score = static_cast<float>(recency);
        ranked_hit.hit.importance_score = entry.importance;
        ranked_hit.timestamp_ns = entry.timestamp_ns;
        ranked.push_back(ranked_hit);
    }

    std::sort(
        ranked.begin(),
        ranked.end(),
        [](const RankedHit& a, const RankedHit& b) {
            if (a.hit.score != b.hit.score) {
                return a.hit.score > b.hit.score;
            }
            if (a.timestamp_ns != b.timestamp_ns) {
                return a.timestamp_ns > b.timestamp_ns;
            }
            return a.hit.record_id > b.hit.record_id;
        });
    if (ranked.size() > options.top_k) {
        ranked.resize(options.top_k);
    }

    out->clear();
    out->reserve(ranked.size());
    for (const auto& ranked_hit : ranked) {
        out->push_back(ranked_hit.hit);
    }
    return MemoryStatus::kOk;
}

MemoryStatus ViewMemoryRecord(
    const std::uint8_t* blob,
    std::size_t blob_size,
    const MemoryIndex& index,
    std::uint64_t record_id,
    MemoryRecordView* out) {
    if (blob == nullptr || out == nullptr) {
        return MemoryStatus::kNullArgument;
    }
    const auto it = std::lower_bound(
        index.entries.begin(),
        index.entries.end(),
        record_id,
        [](const MemoryIndexEntry& entry, std::uint64_t value) {
            return entry.record_id < value;
        });
    if (it == index.entries.end() || it->record_id != record_id) {
        return MemoryStatus::kInvalidParameter;
    }
    if (
        it->source_offset + it->source_size > blob_size ||
        it->content_offset + it->content_size > blob_size ||
        it->vector_offset + static_cast<std::uint64_t>(it->vector_count) * 4u >
            blob_size
    ) {
        return MemoryStatus::kInvalidRecord;
    }

    MemoryRecordView view;
    view.record_id = it->record_id;
    view.timestamp_ns = it->timestamp_ns;
    view.parent_id = it->parent_id;
    view.kind = it->kind;
    view.importance = it->importance;
    view.source = blob + static_cast<std::size_t>(it->source_offset);
    view.source_size = it->source_size;
    view.content = blob + static_cast<std::size_t>(it->content_offset);
    view.content_size = it->content_size;
    view.vector_f32_le = it->vector_count == 0
        ? nullptr
        : blob + static_cast<std::size_t>(it->vector_offset);
    view.vector_count = it->vector_count;
    *out = view;
    return MemoryStatus::kOk;
}

}  // namespace vn97
