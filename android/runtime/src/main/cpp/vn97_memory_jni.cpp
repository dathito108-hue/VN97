#include <jni.h>

#include "vn97/memory.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

using vn97::MemoryKind;
using vn97::MemoryRecordView;
using vn97::MemoryRetrievalOptions;
using vn97::MemoryStatus;
using vn97::MemoryStore;

std::mutex g_mutex;
std::unordered_map<std::uint64_t, std::unique_ptr<MemoryStore>> g_stores;
std::uint64_t g_next_handle = 1;

int Status(MemoryStatus status) {
    return static_cast<int>(status);
}

int Invalid() {
    return Status(MemoryStatus::kInvalidParameter);
}

bool HasLength(JNIEnv* env, jarray array, jsize minimum) {
    return array != nullptr && env->GetArrayLength(array) >= minimum;
}

MemoryStore* FindStoreLocked(jlong handle) {
    if (handle <= 0) {
        return nullptr;
    }
    const auto it = g_stores.find(static_cast<std::uint64_t>(handle));
    return it == g_stores.end() ? nullptr : it->second.get();
}

bool FitsJlongU64(std::uint64_t value) {
    return value <= static_cast<std::uint64_t>(std::numeric_limits<jlong>::max());
}

bool FitsJlongSize(std::size_t value) {
    return value <= static_cast<std::size_t>(std::numeric_limits<jlong>::max());
}

bool FitsJint(std::size_t value) {
    return value <= static_cast<std::size_t>(std::numeric_limits<jint>::max());
}

jlong InsertStoreLocked(std::unique_ptr<MemoryStore> store) {
    if (!store) {
        return 0;
    }
    for (;;) {
        if (g_next_handle == 0 ||
            g_next_handle > static_cast<std::uint64_t>(std::numeric_limits<jlong>::max())) {
            return 0;
        }
        const std::uint64_t candidate = g_next_handle++;
        if (g_stores.find(candidate) == g_stores.end()) {
            g_stores.emplace(candidate, std::move(store));
            return static_cast<jlong>(candidate);
        }
    }
}

}  // namespace

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeMemoryBindings_nativeMemoryCreate(
    JNIEnv* env,
    jobject,
    jstring path,
    jint vector_dim,
    jboolean overwrite,
    jlongArray handle_out) {
    if (path == nullptr || vector_dim <= 0 || !HasLength(env, handle_out, 1)) {
        return Invalid();
    }
    const char* utf = env->GetStringUTFChars(path, nullptr);
    if (utf == nullptr) {
        return Status(MemoryStatus::kIoError);
    }
    MemoryStore* raw = nullptr;
    const MemoryStatus status = MemoryStore::Create(
        utf,
        static_cast<std::uint32_t>(vector_dim),
        overwrite == JNI_TRUE,
        &raw);
    env->ReleaseStringUTFChars(path, utf);
    if (status != MemoryStatus::kOk) {
        return Status(status);
    }
    std::unique_ptr<MemoryStore> store(raw);
    std::lock_guard<std::mutex> guard(g_mutex);
    const jlong handle = InsertStoreLocked(std::move(store));
    if (handle == 0) {
        return Status(MemoryStatus::kIoError);
    }
    env->SetLongArrayRegion(handle_out, 0, 1, &handle);
    return Status(MemoryStatus::kOk);
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeMemoryBindings_nativeMemoryOpen(
    JNIEnv* env,
    jobject,
    jstring path,
    jboolean recover_torn_tail,
    jlongArray handle_out) {
    if (path == nullptr || !HasLength(env, handle_out, 1)) {
        return Invalid();
    }
    const char* utf = env->GetStringUTFChars(path, nullptr);
    if (utf == nullptr) {
        return Status(MemoryStatus::kIoError);
    }
    MemoryStore* raw = nullptr;
    const MemoryStatus status = MemoryStore::Open(
        utf,
        recover_torn_tail == JNI_TRUE,
        &raw);
    env->ReleaseStringUTFChars(path, utf);
    if (status != MemoryStatus::kOk) {
        return Status(status);
    }
    std::unique_ptr<MemoryStore> store(raw);
    std::lock_guard<std::mutex> guard(g_mutex);
    const jlong handle = InsertStoreLocked(std::move(store));
    if (handle == 0) {
        return Status(MemoryStatus::kIoError);
    }
    env->SetLongArrayRegion(handle_out, 0, 1, &handle);
    return Status(MemoryStatus::kOk);
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeMemoryBindings_nativeMemoryClose(
    JNIEnv*, jobject, jlong handle) {
    std::lock_guard<std::mutex> guard(g_mutex);
    if (handle <= 0) {
        return Invalid();
    }
    const auto erased = g_stores.erase(static_cast<std::uint64_t>(handle));
    return erased == 1 ? Status(MemoryStatus::kOk) : Invalid();
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeMemoryBindings_nativeMemoryStats(
    JNIEnv* env,
    jobject,
    jlong handle,
    jintArray ints_out,
    jlongArray longs_out) {
    if (!HasLength(env, ints_out, 1) || !HasLength(env, longs_out, 3)) {
        return Invalid();
    }
    std::lock_guard<std::mutex> guard(g_mutex);
    const MemoryStore* store = FindStoreLocked(handle);
    if (store == nullptr) {
        return Invalid();
    }
    if (!FitsJlongSize(store->record_count()) ||
        !FitsJlongU64(store->last_record_id()) ||
        !FitsJlongSize(store->valid_bytes())) {
        return Invalid();
    }
    const jint ints[1] = {static_cast<jint>(store->vector_dim())};
    const jlong longs[3] = {
        static_cast<jlong>(store->record_count()),
        static_cast<jlong>(store->last_record_id()),
        static_cast<jlong>(store->valid_bytes()),
    };
    env->SetIntArrayRegion(ints_out, 0, 1, ints);
    env->SetLongArrayRegion(longs_out, 0, 3, longs);
    return Status(MemoryStatus::kOk);
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeMemoryBindings_nativeMemoryAppend(
    JNIEnv* env,
    jobject,
    jlong handle,
    jint kind,
    jlong timestamp_ns,
    jfloat importance,
    jbyteArray source,
    jbyteArray content,
    jfloatArray vector,
    jlong parent_id,
    jboolean durable,
    jlongArray record_id_out) {
    if (timestamp_ns < 0 || parent_id < 0 || source == nullptr || content == nullptr ||
        !HasLength(env, record_id_out, 1)) {
        return Invalid();
    }
    const jsize source_size = env->GetArrayLength(source);
    const jsize content_size = env->GetArrayLength(content);
    const jsize vector_count = vector == nullptr ? 0 : env->GetArrayLength(vector);
    std::vector<std::uint8_t> source_bytes(static_cast<std::size_t>(source_size));
    std::vector<std::uint8_t> content_bytes(static_cast<std::size_t>(content_size));
    std::vector<float> vector_values(static_cast<std::size_t>(vector_count));
    if (source_size > 0) {
        env->GetByteArrayRegion(
            source, 0, source_size, reinterpret_cast<jbyte*>(source_bytes.data()));
    }
    if (content_size > 0) {
        env->GetByteArrayRegion(
            content, 0, content_size, reinterpret_cast<jbyte*>(content_bytes.data()));
    }
    if (vector_count > 0) {
        env->GetFloatArrayRegion(vector, 0, vector_count, vector_values.data());
    }
    if (env->ExceptionCheck()) {
        return Status(MemoryStatus::kIoError);
    }

    std::lock_guard<std::mutex> guard(g_mutex);
    MemoryStore* store = FindStoreLocked(handle);
    if (store == nullptr) {
        return Invalid();
    }
    vn97::MemoryAppendInput input;
    input.kind = static_cast<MemoryKind>(kind);
    input.timestamp_ns = static_cast<std::uint64_t>(timestamp_ns);
    input.importance = importance;
    input.source = source_bytes.empty() ? nullptr : source_bytes.data();
    input.source_size = source_bytes.size();
    input.content = content_bytes.empty() ? nullptr : content_bytes.data();
    input.content_size = content_bytes.size();
    input.vector = vector_values.empty() ? nullptr : vector_values.data();
    input.vector_count = vector_values.size();
    input.parent_id = static_cast<std::uint64_t>(parent_id);
    input.durable = durable == JNI_TRUE;
    std::uint64_t record_id = 0;
    const MemoryStatus status = store->Append(input, &record_id);
    if (status != MemoryStatus::kOk) {
        return Status(status);
    }
    if (!FitsJlongU64(record_id)) {
        return Invalid();
    }
    const jlong result = static_cast<jlong>(record_id);
    env->SetLongArrayRegion(record_id_out, 0, 1, &result);
    return Status(MemoryStatus::kOk);
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeMemoryBindings_nativeMemoryRetrieve(
    JNIEnv* env,
    jobject,
    jlong handle,
    jfloatArray query,
    jint top_k,
    jint kind_mask,
    jfloat semantic_weight,
    jfloat recency_weight,
    jfloat importance_weight,
    jlong now_ns,
    jlong recency_half_life_ns,
    jlongArray record_ids_out,
    jfloatArray scores_out,
    jfloatArray semantic_scores_out,
    jfloatArray recency_scores_out,
    jfloatArray importance_scores_out,
    jintArray count_out) {
    if (query == nullptr || top_k <= 0 || kind_mask < 0 || now_ns < 0 ||
        recency_half_life_ns <= 0 || !HasLength(env, count_out, 1) ||
        !HasLength(env, record_ids_out, top_k) || !HasLength(env, scores_out, top_k) ||
        !HasLength(env, semantic_scores_out, top_k) || !HasLength(env, recency_scores_out, top_k) ||
        !HasLength(env, importance_scores_out, top_k)) {
        return Invalid();
    }
    const jsize query_count = env->GetArrayLength(query);
    std::vector<float> query_values(static_cast<std::size_t>(query_count));
    if (query_count > 0) {
        env->GetFloatArrayRegion(query, 0, query_count, query_values.data());
    }
    if (env->ExceptionCheck()) {
        return Status(MemoryStatus::kIoError);
    }

    std::lock_guard<std::mutex> guard(g_mutex);
    const MemoryStore* store = FindStoreLocked(handle);
    if (store == nullptr) {
        return Invalid();
    }
    MemoryRetrievalOptions options;
    options.top_k = static_cast<std::size_t>(top_k);
    options.kind_mask = static_cast<std::uint32_t>(kind_mask);
    options.semantic_weight = semantic_weight;
    options.recency_weight = recency_weight;
    options.importance_weight = importance_weight;
    options.now_ns = static_cast<std::uint64_t>(now_ns);
    options.recency_half_life_ns = static_cast<std::uint64_t>(recency_half_life_ns);
    std::vector<vn97::MemoryHit> hits;
    const MemoryStatus status = store->Retrieve(
        query_values.data(), query_values.size(), options, &hits);
    if (status != MemoryStatus::kOk) {
        return Status(status);
    }
    if (hits.size() > static_cast<std::size_t>(top_k) || !FitsJint(hits.size())) {
        return Invalid();
    }
    std::vector<jlong> ids(hits.size());
    std::vector<jfloat> scores(hits.size());
    std::vector<jfloat> semantic_scores(hits.size());
    std::vector<jfloat> recency_scores(hits.size());
    std::vector<jfloat> importance_scores(hits.size());
    for (std::size_t i = 0; i < hits.size(); ++i) {
        if (!FitsJlongU64(hits[i].record_id)) {
            return Invalid();
        }
        ids[i] = static_cast<jlong>(hits[i].record_id);
        scores[i] = hits[i].score;
        semantic_scores[i] = hits[i].semantic_score;
        recency_scores[i] = hits[i].recency_score;
        importance_scores[i] = hits[i].importance_score;
    }
    if (!hits.empty()) {
        const jsize n = static_cast<jsize>(hits.size());
        env->SetLongArrayRegion(record_ids_out, 0, n, ids.data());
        env->SetFloatArrayRegion(scores_out, 0, n, scores.data());
        env->SetFloatArrayRegion(semantic_scores_out, 0, n, semantic_scores.data());
        env->SetFloatArrayRegion(recency_scores_out, 0, n, recency_scores.data());
        env->SetFloatArrayRegion(importance_scores_out, 0, n, importance_scores.data());
    }
    const jint n = static_cast<jint>(hits.size());
    env->SetIntArrayRegion(count_out, 0, 1, &n);
    return Status(MemoryStatus::kOk);
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeMemoryBindings_nativeMemoryRecordInfo(
    JNIEnv* env,
    jobject,
    jlong handle,
    jlong record_id,
    jintArray ints_out,
    jlongArray longs_out,
    jfloatArray floats_out) {
    if (record_id <= 0 || !HasLength(env, ints_out, 3) ||
        !HasLength(env, longs_out, 2) || !HasLength(env, floats_out, 1)) {
        return Invalid();
    }
    std::lock_guard<std::mutex> guard(g_mutex);
    const MemoryStore* store = FindStoreLocked(handle);
    if (store == nullptr) {
        return Invalid();
    }
    MemoryRecordView record;
    const MemoryStatus status = store->ViewRecord(
        static_cast<std::uint64_t>(record_id), &record);
    if (status != MemoryStatus::kOk) {
        return Status(status);
    }
    if (!FitsJlongU64(record.timestamp_ns) || !FitsJlongU64(record.parent_id) ||
        !FitsJint(record.source_size) || !FitsJint(record.content_size)) {
        return Invalid();
    }
    const jint ints[3] = {
        static_cast<jint>(record.kind),
        static_cast<jint>(record.source_size),
        static_cast<jint>(record.content_size),
    };
    const jlong longs[2] = {
        static_cast<jlong>(record.timestamp_ns),
        static_cast<jlong>(record.parent_id),
    };
    const jfloat floats[1] = {record.importance};
    env->SetIntArrayRegion(ints_out, 0, 3, ints);
    env->SetLongArrayRegion(longs_out, 0, 2, longs);
    env->SetFloatArrayRegion(floats_out, 0, 1, floats);
    return Status(MemoryStatus::kOk);
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeMemoryBindings_nativeMemoryRecordRead(
    JNIEnv* env,
    jobject,
    jlong handle,
    jlong record_id,
    jbyteArray source_out,
    jbyteArray content_out) {
    if (record_id <= 0 || source_out == nullptr || content_out == nullptr) {
        return Invalid();
    }
    std::lock_guard<std::mutex> guard(g_mutex);
    const MemoryStore* store = FindStoreLocked(handle);
    if (store == nullptr) {
        return Invalid();
    }
    MemoryRecordView record;
    const MemoryStatus status = store->ViewRecord(
        static_cast<std::uint64_t>(record_id), &record);
    if (status != MemoryStatus::kOk) {
        return Status(status);
    }
    if (!FitsJint(record.source_size) || !FitsJint(record.content_size) ||
        env->GetArrayLength(source_out) < static_cast<jsize>(record.source_size) ||
        env->GetArrayLength(content_out) < static_cast<jsize>(record.content_size)) {
        return Invalid();
    }
    if (record.source_size > 0) {
        env->SetByteArrayRegion(
            source_out,
            0,
            static_cast<jsize>(record.source_size),
            reinterpret_cast<const jbyte*>(record.source));
    }
    if (record.content_size > 0) {
        env->SetByteArrayRegion(
            content_out,
            0,
            static_cast<jsize>(record.content_size),
            reinterpret_cast<const jbyte*>(record.content));
    }
    return env->ExceptionCheck() ? Status(MemoryStatus::kIoError)
                                 : Status(MemoryStatus::kOk);
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeMemoryBindings_nativeMemoryCompact(
    JNIEnv* env,
    jobject,
    jlong handle,
    jlong max_records,
    jlong max_age_ns,
    jfloat min_importance,
    jlong now_ns,
    jlongArray counts_out) {
    if (max_records < 0 || max_age_ns < 0 || now_ns < 0 ||
        !std::isfinite(min_importance) || !HasLength(env, counts_out, 2)) {
        return Invalid();
    }
    if (static_cast<std::uint64_t>(max_records) >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return Invalid();
    }
    std::lock_guard<std::mutex> guard(g_mutex);
    MemoryStore* store = FindStoreLocked(handle);
    if (store == nullptr) {
        return Invalid();
    }
    vn97::MemoryRetentionPolicy policy;
    policy.max_records = static_cast<std::size_t>(max_records);
    policy.max_age_ns = static_cast<std::uint64_t>(max_age_ns);
    policy.min_importance = min_importance;
    std::size_t retained = 0;
    std::size_t removed = 0;
    const MemoryStatus status = store->Compact(
        policy,
        static_cast<std::uint64_t>(now_ns),
        &retained,
        &removed);
    if (status != MemoryStatus::kOk) {
        return Status(status);
    }
    if (!FitsJlongSize(retained) || !FitsJlongSize(removed)) {
        return Invalid();
    }
    const jlong counts[2] = {
        static_cast<jlong>(retained),
        static_cast<jlong>(removed),
    };
    env->SetLongArrayRegion(counts_out, 0, 2, counts);
    return Status(MemoryStatus::kOk);
}
