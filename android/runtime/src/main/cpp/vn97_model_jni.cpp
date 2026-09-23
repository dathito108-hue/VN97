#include <jni.h>

#include "vn97/model_image.h"
#include "vn97/runtime.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {
constexpr jint kModelNullArgument = 1;
constexpr jint kModelInvalidFd = 3;
constexpr jint kModelInvalidRange = 4;
constexpr jint kRuntimeInvalidConfig = 2;
constexpr jint kRuntimeCounterOverflow = 9;

bool HasLength(JNIEnv* env, jarray array, jsize minimum) {
    return array != nullptr && env->GetArrayLength(array) >= minimum;
}

bool ReadTokenIds(JNIEnv* env, jintArray input, std::vector<std::uint32_t>* out) {
    if (input == nullptr || out == nullptr) return false;
    const jsize size = env->GetArrayLength(input);
    std::vector<jint> raw(static_cast<std::size_t>(size));
    if (size != 0) env->GetIntArrayRegion(input, 0, size, raw.data());
    if (env->ExceptionCheck()) return false;
    out->resize(static_cast<std::size_t>(size));
    for (jsize i = 0; i < size; ++i) {
        if (raw[static_cast<std::size_t>(i)] < 0) return false;
        (*out)[static_cast<std::size_t>(i)] =
            static_cast<std::uint32_t>(raw[static_cast<std::size_t>(i)]);
    }
    return true;
}

void WriteTokenIds(
    JNIEnv* env,
    jintArray output,
    const std::uint32_t* ids,
    std::size_t count) {
    std::vector<jint> raw(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (ids[i] > static_cast<std::uint32_t>(std::numeric_limits<jint>::max())) {
            env->ThrowNew(env->FindClass("java/lang/IllegalStateException"),
                          "native token ID does not fit a JVM Int");
            return;
        }
        raw[i] = static_cast<jint>(ids[i]);
    }
    if (count != 0) {
        env->SetIntArrayRegion(output, 0, static_cast<jsize>(count), raw.data());
    }
}
}  // namespace

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeRuntimeBindings_nativeModelOpen(
    JNIEnv* env,
    jobject,
    jint fd,
    jlong offset,
    jlong length,
    jbyteArray expected_model_id,
    jlongArray handle_out) {
    if (expected_model_id == nullptr || !HasLength(env, handle_out, 1)) {
        return kModelNullArgument;
    }
    if (fd < 0) return kModelInvalidFd;
    if (offset < 0 || length <= 0) return kModelInvalidRange;
    if (env->GetArrayLength(expected_model_id) != 32) return kModelInvalidRange;

    std::array<std::uint8_t, 32> expected{};
    env->GetByteArrayRegion(
        expected_model_id,
        0,
        32,
        reinterpret_cast<jbyte*>(expected.data()));
    if (env->ExceptionCheck()) return kModelNullArgument;

    std::uint64_t handle = 0;
    const int status = vn97_model_open_fd(
        fd,
        static_cast<std::uint64_t>(offset),
        static_cast<std::uint64_t>(length),
        expected.data(),
        expected.size(),
        &handle);
    if (status != 0) return status;
    if (handle > static_cast<std::uint64_t>(std::numeric_limits<jlong>::max())) {
        vn97_model_destroy(handle);
        return kModelInvalidRange;
    }
    const jlong result = static_cast<jlong>(handle);
    env->SetLongArrayRegion(handle_out, 0, 1, &result);
    return 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeRuntimeBindings_nativeModelDestroy(
    JNIEnv*, jobject, jlong handle) {
    if (handle <= 0) return static_cast<jint>(vn97::ModelImageStatus::kInvalidHandle);
    return vn97_model_destroy(static_cast<std::uint64_t>(handle));
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeRuntimeBindings_nativeModelInfo(
    JNIEnv* env,
    jobject,
    jlong handle,
    jintArray ints_out,
    jlongArray longs_out,
    jbyteArray model_id_out) {
    if (!HasLength(env, ints_out, 7) || !HasLength(env, longs_out, 1) ||
        !HasLength(env, model_id_out, 32)) {
        return kModelNullArgument;
    }
    vn97_model_info info{};
    const int status = vn97_model_info_get(static_cast<std::uint64_t>(handle), &info);
    if (status != 0) return status;
    if (info.image_bytes > static_cast<std::size_t>(std::numeric_limits<jlong>::max())) {
        return kModelInvalidRange;
    }
    const jint ints[7] = {
        static_cast<jint>(info.vocab_size),
        static_cast<jint>(info.d_model),
        static_cast<jint>(info.n_layers),
        static_cast<jint>(info.d_state),
        static_cast<jint>(info.embedding_kind),
        static_cast<jint>(info.embedding_rank),
        info.has_tokenizer != 0 ? 1 : 0,
    };
    const jlong longs[1] = {static_cast<jlong>(info.image_bytes)};
    env->SetIntArrayRegion(ints_out, 0, 7, ints);
    env->SetLongArrayRegion(longs_out, 0, 1, longs);
    env->SetByteArrayRegion(
        model_id_out,
        0,
        32,
        reinterpret_cast<const jbyte*>(info.model_id));
    return 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeRuntimeBindings_nativeModelEncode(
    JNIEnv* env,
    jobject,
    jlong handle,
    jbyteArray input,
    jint flags,
    jintArray output,
    jintArray count_out) {
    if (input == nullptr || output == nullptr || !HasLength(env, count_out, 1)) {
        return kModelNullArgument;
    }
    const jsize input_size = env->GetArrayLength(input);
    const jsize output_capacity = env->GetArrayLength(output);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(input_size));
    if (input_size != 0) {
        env->GetByteArrayRegion(
            input, 0, input_size, reinterpret_cast<jbyte*>(bytes.data()));
        if (env->ExceptionCheck()) return kModelNullArgument;
    }
    std::vector<std::uint32_t> ids(static_cast<std::size_t>(output_capacity));
    std::size_t count = 0;
    const int status = vn97_model_tokenizer_encode(
        static_cast<std::uint64_t>(handle),
        bytes.empty() ? nullptr : bytes.data(),
        bytes.size(),
        static_cast<std::uint32_t>(flags),
        ids.empty() ? nullptr : ids.data(),
        ids.size(),
        &count);
    if (status != 0) return status;
    if (count > ids.size() || count > static_cast<std::size_t>(std::numeric_limits<jint>::max())) {
        return static_cast<jint>(vn97::ModelImageStatus::kOutputTooSmall);
    }
    WriteTokenIds(env, output, ids.data(), count);
    if (env->ExceptionCheck()) return kModelNullArgument;
    const jint count_value = static_cast<jint>(count);
    env->SetIntArrayRegion(count_out, 0, 1, &count_value);
    return 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeRuntimeBindings_nativeModelDecodedSize(
    JNIEnv* env,
    jobject,
    jlong handle,
    jintArray token_ids,
    jboolean skip_control,
    jlongArray size_out) {
    if (!HasLength(env, size_out, 1)) return kModelNullArgument;
    std::vector<std::uint32_t> ids;
    if (!ReadTokenIds(env, token_ids, &ids)) return kModelNullArgument;
    std::size_t required = 0;
    const int status = vn97_model_tokenizer_decoded_size(
        static_cast<std::uint64_t>(handle),
        ids.empty() ? nullptr : ids.data(),
        ids.size(),
        skip_control == JNI_TRUE ? 1 : 0,
        &required);
    if (status != 0) return status;
    if (required > static_cast<std::size_t>(std::numeric_limits<jlong>::max())) {
        return kModelInvalidRange;
    }
    const jlong value = static_cast<jlong>(required);
    env->SetLongArrayRegion(size_out, 0, 1, &value);
    return 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeRuntimeBindings_nativeModelDecode(
    JNIEnv* env,
    jobject,
    jlong handle,
    jintArray token_ids,
    jboolean skip_control,
    jbyteArray output,
    jlongArray written_out) {
    if (output == nullptr || !HasLength(env, written_out, 1)) return kModelNullArgument;
    std::vector<std::uint32_t> ids;
    if (!ReadTokenIds(env, token_ids, &ids)) return kModelNullArgument;
    const jsize capacity = env->GetArrayLength(output);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(capacity));
    std::size_t written = 0;
    const int status = vn97_model_tokenizer_decode(
        static_cast<std::uint64_t>(handle),
        ids.empty() ? nullptr : ids.data(),
        ids.size(),
        skip_control == JNI_TRUE ? 1 : 0,
        bytes.empty() ? nullptr : bytes.data(),
        bytes.size(),
        &written);
    if (status != 0) return status;
    if (written > bytes.size() || written > static_cast<std::size_t>(std::numeric_limits<jlong>::max())) {
        return static_cast<jint>(vn97::ModelImageStatus::kOutputTooSmall);
    }
    if (written != 0) {
        env->SetByteArrayRegion(
            output,
            0,
            static_cast<jsize>(written),
            reinterpret_cast<const jbyte*>(bytes.data()));
    }
    const jlong value = static_cast<jlong>(written);
    env->SetLongArrayRegion(written_out, 0, 1, &value);
    return 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeRuntimeBindings_nativeInferStep(
    JNIEnv* env,
    jobject,
    jlong runtime_handle,
    jlong model_handle,
    jintArray input_ids,
    jfloatArray logits) {
    if (logits == nullptr) return kRuntimeInvalidConfig;
    std::vector<std::uint32_t> ids;
    if (!ReadTokenIds(env, input_ids, &ids)) return kRuntimeInvalidConfig;
    const jsize logits_count = env->GetArrayLength(logits);
    jfloat* values = env->GetFloatArrayElements(logits, nullptr);
    if (values == nullptr) return kRuntimeInvalidConfig;
    const int status = vn97_model_runtime_infer_step(
        static_cast<std::uint64_t>(model_handle),
        static_cast<std::uint64_t>(runtime_handle),
        ids.empty() ? nullptr : ids.data(),
        ids.size(),
        values,
        static_cast<std::size_t>(logits_count));
    env->ReleaseFloatArrayElements(logits, values, status == 0 ? 0 : JNI_ABORT);
    return status;
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeRuntimeBindings_nativePrefill(
    JNIEnv* env,
    jobject,
    jlong runtime_handle,
    jlong model_handle,
    jintArray input_ids,
    jint step_count,
    jfloatArray final_logits) {
    if (step_count <= 0 || final_logits == nullptr) return kRuntimeInvalidConfig;
    std::vector<std::uint32_t> ids;
    if (!ReadTokenIds(env, input_ids, &ids)) return kRuntimeInvalidConfig;
    const jsize logits_count = env->GetArrayLength(final_logits);
    jfloat* values = env->GetFloatArrayElements(final_logits, nullptr);
    if (values == nullptr) return kRuntimeInvalidConfig;
    const int status = vn97_model_runtime_prefill(
        static_cast<std::uint64_t>(model_handle),
        static_cast<std::uint64_t>(runtime_handle),
        ids.empty() ? nullptr : ids.data(),
        ids.size(),
        static_cast<std::size_t>(step_count),
        values,
        static_cast<std::size_t>(logits_count));
    env->ReleaseFloatArrayElements(final_logits, values, status == 0 ? 0 : JNI_ABORT);
    return status;
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeRuntimeBindings_nativePrefillHidden(
    JNIEnv* env,
    jobject,
    jlong runtime_handle,
    jlong model_handle,
    jintArray input_ids,
    jint step_count,
    jfloatArray final_hidden) {
    if (step_count <= 0 || final_hidden == nullptr) return kRuntimeInvalidConfig;
    std::vector<std::uint32_t> ids;
    if (!ReadTokenIds(env, input_ids, &ids)) return kRuntimeInvalidConfig;
    const jsize hidden_count = env->GetArrayLength(final_hidden);
    jfloat* values = env->GetFloatArrayElements(final_hidden, nullptr);
    if (values == nullptr) return kRuntimeInvalidConfig;
    const int status = vn97_model_runtime_prefill_hidden(
        static_cast<std::uint64_t>(model_handle),
        static_cast<std::uint64_t>(runtime_handle),
        ids.empty() ? nullptr : ids.data(),
        ids.size(),
        static_cast<std::size_t>(step_count),
        values,
        static_cast<std::size_t>(hidden_count));
    env->ReleaseFloatArrayElements(final_hidden, values, status == 0 ? 0 : JNI_ABORT);
    return status;
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeRuntimeBindings_nativeGenerateGreedy(
    JNIEnv* env,
    jobject,
    jlong runtime_handle,
    jlong model_handle,
    jintArray prompt_ids,
    jint max_new_tokens,
    jint eos_token,
    jintArray output_ids,
    jintArray count_out) {
    if (max_new_tokens < 0 || output_ids == nullptr || !HasLength(env, count_out, 1)) {
        return kRuntimeInvalidConfig;
    }
    std::vector<std::uint32_t> prompt;
    if (!ReadTokenIds(env, prompt_ids, &prompt)) return kRuntimeInvalidConfig;
    const jsize capacity = env->GetArrayLength(output_ids);
    std::vector<std::uint32_t> output(static_cast<std::size_t>(capacity));
    std::size_t count = 0;
    const std::uint32_t eos = eos_token < 0
        ? std::numeric_limits<std::uint32_t>::max()
        : static_cast<std::uint32_t>(eos_token);
    const int status = vn97_model_runtime_generate_greedy(
        static_cast<std::uint64_t>(model_handle),
        static_cast<std::uint64_t>(runtime_handle),
        prompt.empty() ? nullptr : prompt.data(),
        prompt.size(),
        static_cast<std::size_t>(max_new_tokens),
        eos,
        output.empty() ? nullptr : output.data(),
        output.size(),
        &count);
    if (status != 0) return status;
    if (count > output.size() || count > static_cast<std::size_t>(std::numeric_limits<jint>::max())) {
        return kRuntimeCounterOverflow;
    }
    WriteTokenIds(env, output_ids, output.data(), count);
    if (env->ExceptionCheck()) return kRuntimeInvalidConfig;
    const jint count_value = static_cast<jint>(count);
    env->SetIntArrayRegion(count_out, 0, 1, &count_value);
    return 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeRuntimeBindings_nativeModelBinding(
    JNIEnv* env,
    jobject,
    jlong runtime_handle,
    jintArray bound_out,
    jbyteArray model_id_out) {
    if (!HasLength(env, bound_out, 1) || !HasLength(env, model_id_out, 32)) {
        return kRuntimeInvalidConfig;
    }
    int bound = 0;
    std::array<std::uint8_t, 32> model_id{};
    const int status = vn97_runtime_model_binding_get(
        static_cast<std::uint64_t>(runtime_handle),
        &bound,
        model_id.data(),
        model_id.size());
    if (status != 0) return status;
    const jint bound_value = bound != 0 ? 1 : 0;
    env->SetIntArrayRegion(bound_out, 0, 1, &bound_value);
    env->SetByteArrayRegion(
        model_id_out,
        0,
        32,
        reinterpret_cast<const jbyte*>(model_id.data()));
    return 0;
}
