#include "memory_engine_internal.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace vn97 {
namespace {

using namespace memory_internal;

bool WriteAll(int fd, const std::uint8_t* data, std::size_t size) {
    std::size_t offset = 0;
    while (offset < size) {
        const ssize_t written = ::write(fd, data + offset, size - offset);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (written == 0) {
            return false;
        }
        offset += static_cast<std::size_t>(written);
    }
    return true;
}

bool FsyncDirectoryForPath(const char* path) {
    if (path == nullptr) {
        return false;
    }
    const char* slash = std::strrchr(path, '/');
    std::vector<char> directory;
    if (slash == nullptr) {
        directory = {'.', '\0'};
    } else if (slash == path) {
        directory = {'/', '\0'};
    } else {
        directory.assign(path, slash);
        directory.push_back('\0');
    }
    const int dir_fd = ::open(directory.data(), O_RDONLY | O_DIRECTORY);
    if (dir_fd < 0) {
        return false;
    }
    const bool ok = ::fsync(dir_fd) == 0;
    ::close(dir_fd);
    return ok;
}

bool ValidKind(MemoryKind kind) {
    return kind == MemoryKind::kEpisodic || kind == MemoryKind::kSemantic;
}

MemoryStatus BuildFrame(
    std::uint64_t record_id,
    std::uint32_t vector_dim,
    const MemoryAppendInput& input,
    std::vector<std::uint8_t>* frame_out) {
    if (frame_out == nullptr) {
        return MemoryStatus::kNullArgument;
    }
    if (!ValidKind(input.kind)) {
        return MemoryStatus::kInvalidParameter;
    }
    if (
        !std::isfinite(input.importance) ||
        input.importance < 0.0f ||
        input.importance > 1.0f
    ) {
        return MemoryStatus::kInvalidParameter;
    }
    if (
        (input.source_size != 0 && input.source == nullptr) ||
        (input.content_size != 0 && input.content == nullptr) ||
        (input.vector_count != 0 && input.vector == nullptr)
    ) {
        return MemoryStatus::kNullArgument;
    }
    if (
        input.source_size > kMaxSourceBytes ||
        input.content_size > kMaxContentBytes ||
        !ValidUtf8(input.source, input.source_size) ||
        !ValidUtf8(input.content, input.content_size)
    ) {
        return MemoryStatus::kInvalidParameter;
    }
    if (input.vector_count != 0 && input.vector_count != vector_dim) {
        return MemoryStatus::kInvalidParameter;
    }
    for (std::size_t i = 0; i < input.vector_count; ++i) {
        if (!std::isfinite(input.vector[i])) {
            return MemoryStatus::kInvalidParameter;
        }
    }

    if (MulOverflows(input.vector_count, sizeof(float))) {
        return MemoryStatus::kInvalidParameter;
    }
    std::size_t body_size = kRecordFixedSize;
    if (AddOverflows(body_size, input.source_size)) {
        return MemoryStatus::kInvalidParameter;
    }
    body_size += input.source_size;
    if (AddOverflows(body_size, input.content_size)) {
        return MemoryStatus::kInvalidParameter;
    }
    body_size += input.content_size;
    const std::size_t vector_bytes = input.vector_count * sizeof(float);
    if (AddOverflows(body_size, vector_bytes)) {
        return MemoryStatus::kInvalidParameter;
    }
    body_size += vector_bytes;
    if (body_size > kMaxFrameBody) {
        return MemoryStatus::kInvalidParameter;
    }

    std::vector<std::uint8_t> frame(kFramePrefixSize + body_size, 0u);
    std::uint8_t* body = frame.data() + kFramePrefixSize;
    WriteU64LE(body + 0, record_id);
    WriteU64LE(body + 8, input.timestamp_ns);
    body[16] = static_cast<std::uint8_t>(input.kind);
    body[17] = 0;
    WriteU16LE(body + 18, 0);
    WriteF32LE(body + 20, input.importance);
    WriteU32LE(body + 24, static_cast<std::uint32_t>(input.source_size));
    WriteU32LE(body + 28, static_cast<std::uint32_t>(input.content_size));
    WriteU32LE(body + 32, static_cast<std::uint32_t>(input.vector_count));
    WriteU64LE(body + 36, input.parent_id);
    const auto digest = Sha256(input.content, input.content_size);
    std::memcpy(body + 44, digest.data(), digest.size());

    std::size_t offset = kRecordFixedSize;
    if (input.source_size != 0) {
        std::memcpy(body + offset, input.source, input.source_size);
    }
    offset += input.source_size;
    if (input.content_size != 0) {
        std::memcpy(body + offset, input.content, input.content_size);
    }
    offset += input.content_size;
    for (std::size_t i = 0; i < input.vector_count; ++i) {
        WriteF32LE(body + offset + i * 4u, input.vector[i]);
    }

    WriteU32LE(frame.data(), static_cast<std::uint32_t>(body_size));
    WriteU32LE(frame.data() + 4, Crc32(body, body_size));
    *frame_out = std::move(frame);
    return MemoryStatus::kOk;
}

}  // namespace

MemoryStore::~MemoryStore() {
    Unmap();
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

void MemoryStore::Unmap() {
    if (mapping_ != nullptr && mapping_size_ != 0) {
        ::munmap(mapping_, mapping_size_);
    }
    mapping_ = nullptr;
    mapping_size_ = 0;
}

std::uint64_t MemoryStore::last_record_id() const {
    return index_.entries.empty() ? 0 : index_.entries.back().record_id;
}

MemoryStatus MemoryStore::MapAndIndex(bool recover_torn_tail) {
    struct stat st {};
    if (::fstat(fd_, &st) != 0 || st.st_size < 0) {
        return MemoryStatus::kIoError;
    }
    if (
        static_cast<std::uint64_t>(st.st_size) >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())
    ) {
        return MemoryStatus::kIoError;
    }
    const std::size_t size = static_cast<std::size_t>(st.st_size);
    if (size < kHeaderSize) {
        return MemoryStatus::kBlobTooShort;
    }

    Unmap();
    void* mapped = ::mmap(nullptr, size, PROT_READ, MAP_SHARED, fd_, 0);
    if (mapped == MAP_FAILED) {
        mapping_ = nullptr;
        mapping_size_ = 0;
        return MemoryStatus::kIoError;
    }
    mapping_ = static_cast<std::uint8_t*>(mapped);
    mapping_size_ = size;

    MemoryIndex built;
    MemoryStatus status = BuildMemoryIndex(
        mapping_, mapping_size_, recover_torn_tail, &built);
    if (status != MemoryStatus::kOk) {
        return status;
    }
    if (built.tail_truncated && recover_torn_tail) {
        const std::size_t valid_bytes = built.indexed_bytes;
        Unmap();
        if (
            ::ftruncate(fd_, static_cast<off_t>(valid_bytes)) != 0 ||
            ::fsync(fd_) != 0
        ) {
            return MemoryStatus::kIoError;
        }
        if (::fstat(fd_, &st) != 0 || st.st_size < 0) {
            return MemoryStatus::kIoError;
        }
        const std::size_t recovered_size = static_cast<std::size_t>(st.st_size);
        mapped = ::mmap(nullptr, recovered_size, PROT_READ, MAP_SHARED, fd_, 0);
        if (mapped == MAP_FAILED) {
            mapping_ = nullptr;
            mapping_size_ = 0;
            return MemoryStatus::kIoError;
        }
        mapping_ = static_cast<std::uint8_t*>(mapped);
        mapping_size_ = recovered_size;
        status = BuildMemoryIndex(mapping_, mapping_size_, false, &built);
        if (status != MemoryStatus::kOk) {
            return status;
        }
    }
    index_ = std::move(built);
    return MemoryStatus::kOk;
}

MemoryStatus MemoryStore::RemapAndExtend(std::size_t previous_bytes) {
    if (index_.indexed_bytes != previous_bytes) {
        return MemoryStatus::kInvalidParameter;
    }
    struct stat st {};
    if (::fstat(fd_, &st) != 0 || st.st_size < 0) {
        return MemoryStatus::kIoError;
    }
    if (
        static_cast<std::uint64_t>(st.st_size) >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())
    ) {
        return MemoryStatus::kIoError;
    }
    const std::size_t new_size = static_cast<std::size_t>(st.st_size);
    Unmap();
    void* mapped = ::mmap(nullptr, new_size, PROT_READ, MAP_SHARED, fd_, 0);
    if (mapped == MAP_FAILED) {
        mapping_ = nullptr;
        mapping_size_ = 0;
        return MemoryStatus::kIoError;
    }
    mapping_ = static_cast<std::uint8_t*>(mapped);
    mapping_size_ = new_size;

    MemoryIndex extended = index_;
    const MemoryStatus status = ExtendMemoryIndex(
        mapping_, mapping_size_, false, &extended);
    if (status != MemoryStatus::kOk) {
        return status;
    }
    index_ = std::move(extended);
    return MemoryStatus::kOk;
}

MemoryStatus MemoryStore::Create(
    const char* path,
    std::uint32_t vector_dim,
    bool overwrite,
    MemoryStore** out) {
    if (path == nullptr || out == nullptr) {
        return MemoryStatus::kNullArgument;
    }
    *out = nullptr;
    if (vector_dim > kMaxVectorDim) {
        return MemoryStatus::kInvalidParameter;
    }

    int flags = O_RDWR | O_CREAT | O_CLOEXEC;
    if (!overwrite) {
        flags |= O_EXCL;
    }
    const int fd = ::open(path, flags, 0600);
    if (fd < 0) {
        return MemoryStatus::kIoError;
    }
    if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
        ::close(fd);
        return MemoryStatus::kLocked;
    }
    if (overwrite && ::ftruncate(fd, 0) != 0) {
        ::close(fd);
        return MemoryStatus::kIoError;
    }

    std::array<std::uint8_t, kHeaderSize> header = {};
    std::memcpy(header.data(), kMagic, sizeof(kMagic));
    WriteU32LE(header.data() + 8, kVersion);
    WriteU32LE(header.data() + 12, vector_dim);
    if (!WriteAll(fd, header.data(), header.size()) || ::fsync(fd) != 0) {
        ::close(fd);
        return MemoryStatus::kIoError;
    }

    MemoryStore* store = new (std::nothrow) MemoryStore();
    if (store == nullptr) {
        ::close(fd);
        return MemoryStatus::kIoError;
    }
    store->fd_ = fd;
    store->path_.assign(path, path + std::strlen(path) + 1);
    const MemoryStatus status = store->MapAndIndex(false);
    if (status != MemoryStatus::kOk) {
        delete store;
        return status;
    }
    *out = store;
    return MemoryStatus::kOk;
}

MemoryStatus MemoryStore::Open(
    const char* path,
    bool recover_torn_tail,
    MemoryStore** out) {
    if (path == nullptr || out == nullptr) {
        return MemoryStatus::kNullArgument;
    }
    *out = nullptr;
    const int fd = ::open(path, O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        return MemoryStatus::kIoError;
    }
    if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
        ::close(fd);
        return MemoryStatus::kLocked;
    }

    MemoryStore* store = new (std::nothrow) MemoryStore();
    if (store == nullptr) {
        ::close(fd);
        return MemoryStatus::kIoError;
    }
    store->fd_ = fd;
    store->path_.assign(path, path + std::strlen(path) + 1);
    const MemoryStatus status = store->MapAndIndex(recover_torn_tail);
    if (status != MemoryStatus::kOk) {
        delete store;
        return status;
    }
    *out = store;
    return MemoryStatus::kOk;
}

MemoryStatus MemoryStore::Append(
    const MemoryAppendInput& input,
    std::uint64_t* record_id_out) {
    if (record_id_out == nullptr) {
        return MemoryStatus::kNullArgument;
    }
    if (input.parent_id != 0 && !HasRecordId(index_.entries, input.parent_id)) {
        return MemoryStatus::kInvalidParameter;
    }
    const std::uint64_t previous_id = last_record_id();
    if (previous_id == std::numeric_limits<std::uint64_t>::max()) {
        return MemoryStatus::kInvalidParameter;
    }
    const std::uint64_t record_id = previous_id + 1;

    std::vector<std::uint8_t> frame;
    const MemoryStatus frame_status = BuildFrame(
        record_id, index_.vector_dim, input, &frame);
    if (frame_status != MemoryStatus::kOk) {
        return frame_status;
    }

    const std::size_t previous_bytes = index_.indexed_bytes;
    if (mapping_size_ != previous_bytes || ::lseek(fd_, 0, SEEK_END) < 0) {
        return MemoryStatus::kIoError;
    }
    if (!WriteAll(fd_, frame.data(), frame.size())) {
        (void)::ftruncate(fd_, static_cast<off_t>(previous_bytes));
        return MemoryStatus::kIoError;
    }
    if (input.durable && ::fsync(fd_) != 0) {
        (void)::ftruncate(fd_, static_cast<off_t>(previous_bytes));
        return MemoryStatus::kIoError;
    }

    const MemoryStatus extend_status = RemapAndExtend(previous_bytes);
    if (extend_status != MemoryStatus::kOk) {
        return extend_status;
    }
    *record_id_out = record_id;
    return MemoryStatus::kOk;
}

MemoryStatus MemoryStore::Retrieve(
    const float* query,
    std::size_t query_count,
    const MemoryRetrievalOptions& options,
    std::vector<MemoryHit>* out) const {
    return RetrieveMemory(
        mapping_, mapping_size_, index_, query, query_count, options, out);
}

MemoryStatus MemoryStore::ViewRecord(
    std::uint64_t record_id,
    MemoryRecordView* out) const {
    return ViewMemoryRecord(mapping_, mapping_size_, index_, record_id, out);
}

MemoryStatus MemoryStore::Compact(
    const MemoryRetentionPolicy& policy,
    std::uint64_t now_ns,
    std::size_t* retained_out,
    std::size_t* removed_out) {
    if (retained_out == nullptr || removed_out == nullptr) {
        return MemoryStatus::kNullArgument;
    }
    if (
        !std::isfinite(policy.min_importance) ||
        policy.min_importance < 0.0f ||
        policy.min_importance > 1.0f
    ) {
        return MemoryStatus::kInvalidParameter;
    }
    if (index_.entries.empty()) {
        *retained_out = 0;
        *removed_out = 0;
        return MemoryStatus::kOk;
    }

    if (now_ns == 0) {
        for (const auto& entry : index_.entries) {
            now_ns = std::max(now_ns, entry.timestamp_ns);
        }
    }

    std::vector<bool> keep(index_.entries.size(), false);
    std::size_t keep_count = 0;
    for (std::size_t i = 0; i < index_.entries.size(); ++i) {
        const auto& entry = index_.entries[i];
        const std::uint64_t age =
            now_ns > entry.timestamp_ns ? now_ns - entry.timestamp_ns : 0;
        const bool accepted =
            entry.importance >= policy.min_importance &&
            (policy.max_age_ns == 0 || age <= policy.max_age_ns);
        keep[i] = accepted;
        keep_count += accepted ? 1u : 0u;
    }
    if (!keep.back()) {
        keep.back() = true;
        ++keep_count;
    }

    if (policy.max_records != 0 && keep_count > policy.max_records) {
        std::vector<std::size_t> ranked;
        ranked.reserve(keep_count);
        for (std::size_t i = 0; i < keep.size(); ++i) {
            if (keep[i]) {
                ranked.push_back(i);
            }
        }
        std::sort(
            ranked.begin(),
            ranked.end(),
            [&](std::size_t a, std::size_t b) {
                const auto& left = index_.entries[a];
                const auto& right = index_.entries[b];
                if (left.importance != right.importance) {
                    return left.importance > right.importance;
                }
                if (left.timestamp_ns != right.timestamp_ns) {
                    return left.timestamp_ns > right.timestamp_ns;
                }
                return left.record_id > right.record_id;
            });
        std::fill(keep.begin(), keep.end(), false);
        keep_count = 0;
        const std::size_t limit = std::min(policy.max_records, ranked.size());
        for (std::size_t i = 0; i < limit; ++i) {
            keep[ranked[i]] = true;
            ++keep_count;
        }
        if (!keep.back()) {
            keep.back() = true;
            ++keep_count;
        }
    }

    bool changed = true;
    while (changed) {
        changed = false;
        for (std::size_t i = 0; i < index_.entries.size(); ++i) {
            if (!keep[i]) {
                continue;
            }
            const std::uint64_t parent_id = index_.entries[i].parent_id;
            if (parent_id == 0) {
                continue;
            }
            const auto parent_it = std::lower_bound(
                index_.entries.begin(),
                index_.entries.end(),
                parent_id,
                [](const MemoryIndexEntry& entry, std::uint64_t value) {
                    return entry.record_id < value;
                });
            if (
                parent_it == index_.entries.end() ||
                parent_it->record_id != parent_id
            ) {
                return MemoryStatus::kInvalidRecord;
            }
            const std::size_t parent_index =
                static_cast<std::size_t>(parent_it - index_.entries.begin());
            if (!keep[parent_index]) {
                keep[parent_index] = true;
                ++keep_count;
                changed = true;
            }
        }
    }

    std::vector<char> tmp_template(path_.begin(), path_.end() - 1);
    const char suffix[] = ".tmpXXXXXX";
    tmp_template.insert(tmp_template.end(), suffix, suffix + sizeof(suffix));
    const int tmp_fd = ::mkstemp(tmp_template.data());
    if (tmp_fd < 0) {
        return MemoryStatus::kIoError;
    }
    bool tmp_exists = true;
    if (
        ::fchmod(tmp_fd, 0600) != 0 ||
        ::flock(tmp_fd, LOCK_EX | LOCK_NB) != 0
    ) {
        ::close(tmp_fd);
        ::unlink(tmp_template.data());
        return MemoryStatus::kIoError;
    }

    MemoryStatus result = MemoryStatus::kOk;
    if (!WriteAll(tmp_fd, mapping_, kHeaderSize)) {
        result = MemoryStatus::kIoError;
    }
    if (result == MemoryStatus::kOk) {
        for (std::size_t i = 0; i < index_.entries.size(); ++i) {
            if (!keep[i]) {
                continue;
            }
            const auto& entry = index_.entries[i];
            if (
                entry.frame_offset + entry.frame_size > mapping_size_ ||
                !WriteAll(
                    tmp_fd,
                    mapping_ + static_cast<std::size_t>(entry.frame_offset),
                    static_cast<std::size_t>(entry.frame_size))
            ) {
                result = MemoryStatus::kIoError;
                break;
            }
        }
    }
    if (result == MemoryStatus::kOk && ::fsync(tmp_fd) != 0) {
        result = MemoryStatus::kIoError;
    }
    if (
        result == MemoryStatus::kOk &&
        ::rename(tmp_template.data(), path_.data()) != 0
    ) {
        result = MemoryStatus::kIoError;
    } else if (result == MemoryStatus::kOk) {
        tmp_exists = false;
        (void)FsyncDirectoryForPath(path_.data());
    }

    if (result != MemoryStatus::kOk) {
        ::close(tmp_fd);
        if (tmp_exists) {
            ::unlink(tmp_template.data());
        }
        return result;
    }

    Unmap();
    if (fd_ >= 0) {
        ::close(fd_);
    }
    fd_ = tmp_fd;
    index_ = MemoryIndex();
    result = MapAndIndex(false);
    if (result != MemoryStatus::kOk) {
        return result;
    }

    *retained_out = keep_count;
    *removed_out = keep.size() - keep_count;
    return MemoryStatus::kOk;
}

}  // namespace vn97

extern "C" {

int vn97_memory_store_create(
    const char* path,
    std::uint32_t vector_dim,
    int overwrite,
    void** out_store) {
    if (out_store == nullptr) {
        return static_cast<int>(vn97::MemoryStatus::kNullArgument);
    }
    vn97::MemoryStore* store = nullptr;
    const auto status = vn97::MemoryStore::Create(
        path, vector_dim, overwrite != 0, &store);
    *out_store = store;
    return static_cast<int>(status);
}

int vn97_memory_store_open(
    const char* path,
    int recover_torn_tail,
    void** out_store) {
    if (out_store == nullptr) {
        return static_cast<int>(vn97::MemoryStatus::kNullArgument);
    }
    vn97::MemoryStore* store = nullptr;
    const auto status = vn97::MemoryStore::Open(
        path, recover_torn_tail != 0, &store);
    *out_store = store;
    return static_cast<int>(status);
}

void vn97_memory_store_close(void* store) {
    delete static_cast<vn97::MemoryStore*>(store);
}

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
    std::uint64_t* record_id_out) {
    if (store == nullptr) {
        return static_cast<int>(vn97::MemoryStatus::kNullArgument);
    }
    vn97::MemoryAppendInput input;
    input.kind = static_cast<vn97::MemoryKind>(kind);
    input.timestamp_ns = timestamp_ns;
    input.importance = importance;
    input.source = source;
    input.source_size = source_size;
    input.content = content;
    input.content_size = content_size;
    input.vector = vector;
    input.vector_count = vector_count;
    input.parent_id = parent_id;
    input.durable = durable != 0;
    return static_cast<int>(
        static_cast<vn97::MemoryStore*>(store)->Append(input, record_id_out));
}

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
    std::size_t* output_count) {
    if (store == nullptr || output_count == nullptr) {
        return static_cast<int>(vn97::MemoryStatus::kNullArgument);
    }
    if (
        output_capacity != 0 &&
        (record_ids == nullptr || scores == nullptr ||
         semantic_scores == nullptr || recency_scores == nullptr ||
         importance_scores == nullptr)
    ) {
        return static_cast<int>(vn97::MemoryStatus::kNullArgument);
    }

    vn97::MemoryRetrievalOptions options;
    options.top_k = top_k;
    options.kind_mask = kind_mask;
    options.semantic_weight = semantic_weight;
    options.recency_weight = recency_weight;
    options.importance_weight = importance_weight;
    options.now_ns = now_ns;
    options.recency_half_life_ns = recency_half_life_ns;

    std::vector<vn97::MemoryHit> hits;
    const auto status = static_cast<vn97::MemoryStore*>(store)->Retrieve(
        query, query_count, options, &hits);
    if (status != vn97::MemoryStatus::kOk) {
        return static_cast<int>(status);
    }
    *output_count = hits.size();
    if (hits.size() > output_capacity) {
        return static_cast<int>(vn97::MemoryStatus::kOutputTooSmall);
    }
    for (std::size_t i = 0; i < hits.size(); ++i) {
        record_ids[i] = hits[i].record_id;
        scores[i] = hits[i].score;
        semantic_scores[i] = hits[i].semantic_score;
        recency_scores[i] = hits[i].recency_score;
        importance_scores[i] = hits[i].importance_score;
    }
    return 0;
}

int vn97_memory_store_compact(
    void* store,
    std::size_t max_records,
    std::uint64_t max_age_ns,
    float min_importance,
    std::uint64_t now_ns,
    std::size_t* retained_out,
    std::size_t* removed_out) {
    if (store == nullptr) {
        return static_cast<int>(vn97::MemoryStatus::kNullArgument);
    }
    vn97::MemoryRetentionPolicy policy;
    policy.max_records = max_records;
    policy.max_age_ns = max_age_ns;
    policy.min_importance = min_importance;
    return static_cast<int>(
        static_cast<vn97::MemoryStore*>(store)->Compact(
            policy, now_ns, retained_out, removed_out));
}

int vn97_memory_store_stats(
    void* store,
    std::uint32_t* vector_dim,
    std::size_t* record_count,
    std::uint64_t* last_record_id,
    std::size_t* valid_bytes) {
    if (
        store == nullptr || vector_dim == nullptr ||
        record_count == nullptr || last_record_id == nullptr ||
        valid_bytes == nullptr
    ) {
        return static_cast<int>(vn97::MemoryStatus::kNullArgument);
    }
    const auto* memory_store = static_cast<const vn97::MemoryStore*>(store);
    *vector_dim = memory_store->vector_dim();
    *record_count = memory_store->record_count();
    *last_record_id = memory_store->last_record_id();
    *valid_bytes = memory_store->valid_bytes();
    return 0;
}

}
