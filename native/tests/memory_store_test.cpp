#include "vn97/memory.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

std::string TempPath() {
    char path[] = "/tmp/vn97-memory-store-XXXXXX";
    const int fd = ::mkstemp(path);
    assert(fd >= 0);
    ::close(fd);
    assert(::unlink(path) == 0);
    return std::string(path);
}

void AssertNear(float actual, float expected, float tolerance = 2e-5f) {
    assert(std::fabs(actual - expected) <= tolerance);
}

std::size_t FileSize(const std::string& path) {
    struct stat st {};
    assert(::stat(path.c_str(), &st) == 0);
    assert(st.st_size >= 0);
    return static_cast<std::size_t>(st.st_size);
}

vn97::MemoryAppendInput MakeAppend(
    vn97::MemoryKind kind,
    std::uint64_t timestamp_ns,
    float importance,
    const char* source,
    const char* content,
    const float* vector,
    std::size_t vector_count,
    std::uint64_t parent_id = 0) {
    vn97::MemoryAppendInput input;
    input.kind = kind;
    input.timestamp_ns = timestamp_ns;
    input.importance = importance;
    input.source = reinterpret_cast<const std::uint8_t*>(source);
    input.source_size = std::strlen(source);
    input.content = reinterpret_cast<const std::uint8_t*>(content);
    input.content_size = std::strlen(content);
    input.vector = vector;
    input.vector_count = vector_count;
    input.parent_id = parent_id;
    input.durable = true;
    return input;
}

}  // namespace

int main() {
    const std::string path = TempPath();
    vn97::MemoryStore* store = nullptr;
    assert(
        vn97::MemoryStore::Create(
            path.c_str(), 3, false, &store) ==
        vn97::MemoryStatus::kOk);
    assert(store != nullptr);
    assert(store->vector_dim() == 3);
    assert(store->record_count() == 0);

    vn97::MemoryStore* second = nullptr;
    assert(
        vn97::MemoryStore::Open(
            path.c_str(), false, &second) ==
        vn97::MemoryStatus::kLocked);
    assert(second == nullptr);

    const float exact[] = {1.0f, 0.0f, 0.0f};
    const float near[] = {0.9f, 0.1f, 0.0f};
    const float opposite[] = {-1.0f, 0.0f, 0.0f};
    const float other[] = {0.0f, 1.0f, 0.0f};

    std::uint64_t root = 0;
    assert(
        store->Append(
            MakeAppend(
                vn97::MemoryKind::kEpisodic,
                100,
                0.2f,
                "chat:1",
                "root exact",
                exact,
                3),
            &root) ==
        vn97::MemoryStatus::kOk);
    assert(root == 1);
    const std::size_t after_root = store->valid_bytes();
    assert(after_root == FileSize(path));
    assert(store->record_count() == 1);

    std::uint64_t child = 0;
    assert(
        store->Append(
            MakeAppend(
                vn97::MemoryKind::kSemantic,
                200,
                1.0f,
                "derived",
                "important child",
                near,
                3,
                root),
            &child) ==
        vn97::MemoryStatus::kOk);
    assert(child == 2);
    assert(store->record_count() == 2);
    assert(store->valid_bytes() > after_root);
    assert(store->valid_bytes() == FileSize(path));

    std::uint64_t discarded = 0;
    assert(
        store->Append(
            MakeAppend(
                vn97::MemoryKind::kSemantic,
                300,
                0.0f,
                "test",
                "opposite",
                opposite,
                3),
            &discarded) ==
        vn97::MemoryStatus::kOk);
    assert(discarded == 3);

    std::uint64_t latest = 0;
    assert(
        store->Append(
            MakeAppend(
                vn97::MemoryKind::kEpisodic,
                400,
                0.0f,
                "test",
                "latest",
                other,
                3),
            &latest) ==
        vn97::MemoryStatus::kOk);
    assert(latest == 4);
    assert(store->record_count() == 4);
    assert(store->last_record_id() == 4);

    vn97::MemoryRecordView view;
    assert(
        store->ViewRecord(child, &view) ==
        vn97::MemoryStatus::kOk);
    assert(view.record_id == child);
    assert(view.parent_id == root);
    assert(view.kind == vn97::MemoryKind::kSemantic);
    assert(view.content_size == std::strlen("important child"));
    assert(
        std::memcmp(
            view.content,
            "important child",
            view.content_size) == 0);
    assert(view.vector_count == 3);
    assert(view.vector_f32_le != nullptr);

    vn97::MemoryRetrievalOptions semantic_only;
    semantic_only.top_k = 4;
    semantic_only.kind_mask = 0x3u;
    semantic_only.semantic_weight = 1.0f;
    semantic_only.recency_weight = 0.0f;
    semantic_only.importance_weight = 0.0f;
    semantic_only.now_ns = 400;
    std::vector<vn97::MemoryHit> hits;
    assert(
        store->Retrieve(
            exact, 3, semantic_only, &hits) ==
        vn97::MemoryStatus::kOk);
    assert(hits.size() == 4);
    assert(hits[0].record_id == root);
    assert(hits[1].record_id == child);
    assert(hits.back().record_id == discarded);
    AssertNear(hits[0].semantic_score, 1.0f);
    AssertNear(hits.back().semantic_score, -1.0f);

    vn97::MemoryRetrievalOptions hybrid;
    hybrid.top_k = 2;
    hybrid.kind_mask = 0x2u;
    hybrid.semantic_weight = 1.0f;
    hybrid.recency_weight = 0.2f;
    hybrid.importance_weight = 0.2f;
    hybrid.now_ns = 400;
    hybrid.recency_half_life_ns = 100;
    assert(
        store->Retrieve(
            exact, 3, hybrid, &hits) ==
        vn97::MemoryStatus::kOk);
    assert(hits.size() == 2);
    assert(hits[0].record_id == child);
    assert(hits[1].record_id == discarded);

    vn97::MemoryRetentionPolicy policy;
    policy.max_records = 1;
    policy.min_importance = 0.9f;
    std::size_t retained = 0;
    std::size_t removed = 0;
    assert(
        store->Compact(
            policy,
            400,
            &retained,
            &removed) ==
        vn97::MemoryStatus::kOk);
    assert(retained == 3);
    assert(removed == 1);
    assert(store->record_count() == 3);
    assert(store->last_record_id() == 4);
    assert(store->ViewRecord(root, &view) == vn97::MemoryStatus::kOk);
    assert(store->ViewRecord(child, &view) == vn97::MemoryStatus::kOk);
    assert(store->ViewRecord(latest, &view) == vn97::MemoryStatus::kOk);
    assert(
        store->ViewRecord(discarded, &view) ==
        vn97::MemoryStatus::kInvalidParameter);

    std::uint64_t after_compact = 0;
    assert(
        store->Append(
            MakeAppend(
                vn97::MemoryKind::kEpisodic,
                500,
                0.5f,
                "test",
                "after compact",
                exact,
                3),
            &after_compact) ==
        vn97::MemoryStatus::kOk);
    assert(after_compact == 5);
    assert(store->record_count() == 4);
    delete store;

    assert(
        vn97::MemoryStore::Open(
            path.c_str(), false, &store) ==
        vn97::MemoryStatus::kOk);
    assert(store->record_count() == 4);
    assert(store->last_record_id() == 5);
    delete store;

    const int fd = ::open(path.c_str(), O_WRONLY | O_APPEND);
    assert(fd >= 0);
    const std::uint8_t torn[] = {0x20, 0x00, 0x00};
    assert(::write(fd, torn, sizeof(torn)) == static_cast<ssize_t>(sizeof(torn)));
    ::close(fd);
    const std::size_t torn_size = FileSize(path);
    assert(
        vn97::MemoryStore::Open(
            path.c_str(), false, &store) ==
        vn97::MemoryStatus::kTruncatedTail);
    assert(store == nullptr);
    assert(
        vn97::MemoryStore::Open(
            path.c_str(), true, &store) ==
        vn97::MemoryStatus::kOk);
    assert(FileSize(path) + sizeof(torn) == torn_size);
    assert(store->last_record_id() == 5);
    delete store;

    assert(::unlink(path.c_str()) == 0);
    return 0;
}
