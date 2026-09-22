#include <jni.h>

#include "vn97/generation.h"
#include "vn97/sampler.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {
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
}  // namespace

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeSamplerBindings_nativeSample(
    JNIEnv* env,
    jobject,
    jfloatArray logits,
    jfloat temperature,
    jint top_k,
    jfloat top_p,
    jlong seed,
    jlong step,
    jintArray token_out) {
    if (logits == nullptr || !HasLength(env, token_out, 1)) {
        return static_cast<jint>(vn97::SamplerStatus::kNullArgument);
    }
    if (top_k < 0 || step < 0) {
        return static_cast<jint>(vn97::SamplerStatus::kInvalidConfig);
    }
    const jsize count = env->GetArrayLength(logits);
    if (count <= 0) return static_cast<jint>(vn97::SamplerStatus::kEmptyLogits);
    jfloat* values = env->GetFloatArrayElements(logits, nullptr);
    if (values == nullptr) return static_cast<jint>(vn97::SamplerStatus::kOutOfMemory);

    vn97::SamplerConfig config;
    config.temperature = temperature;
    config.top_k = static_cast<std::uint32_t>(top_k);
    config.top_p = top_p;
    config.seed = static_cast<std::uint64_t>(seed);
    std::uint32_t token = 0;
    const auto status = vn97::SampleLogitsF32(
        values,
        static_cast<std::size_t>(count),
        config,
        static_cast<std::uint64_t>(step),
        &token);
    env->ReleaseFloatArrayElements(logits, values, JNI_ABORT);
    if (status != vn97::SamplerStatus::kOk) return static_cast<jint>(status);
    const jint result = static_cast<jint>(token);
    env->SetIntArrayRegion(token_out, 0, 1, &result);
    return static_cast<jint>(vn97::SamplerStatus::kOk);
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeGenerationBindings_nativeOpen(
    JNIEnv* env,
    jobject,
    jlong model_handle,
    jlong runtime_handle,
    jintArray prompt_ids,
    jfloat temperature,
    jint top_k,
    jfloat top_p,
    jlong seed,
    jint eos_token,
    jlongArray handle_out) {
    if (!HasLength(env, handle_out, 1)) {
        return static_cast<jint>(vn97::GenerationStatus::kNullArgument);
    }
    if (model_handle <= 0 || runtime_handle <= 0 || top_k < 0 || eos_token < -1) {
        return static_cast<jint>(vn97::GenerationStatus::kInvalidConfig);
    }
    std::vector<std::uint32_t> prompt;
    if (!ReadTokenIds(env, prompt_ids, &prompt) || prompt.empty()) {
        return static_cast<jint>(vn97::GenerationStatus::kInvalidConfig);
    }

    vn97_generation_config config{};
    config.sampler.temperature = temperature;
    config.sampler.top_k = static_cast<std::uint32_t>(top_k);
    config.sampler.top_p = top_p;
    config.sampler.seed = static_cast<std::uint64_t>(seed);
    config.eos_token = eos_token < 0
        ? std::numeric_limits<std::uint32_t>::max()
        : static_cast<std::uint32_t>(eos_token);

    std::uint64_t handle = 0;
    const int status = vn97_generation_open(
        static_cast<std::uint64_t>(model_handle),
        static_cast<std::uint64_t>(runtime_handle),
        prompt.data(),
        prompt.size(),
        &config,
        &handle);
    if (status != 0) return status;
    if (handle > static_cast<std::uint64_t>(std::numeric_limits<jlong>::max())) {
        vn97_generation_destroy(handle);
        return static_cast<jint>(vn97::GenerationStatus::kCounterOverflow);
    }
    const jlong value = static_cast<jlong>(handle);
    env->SetLongArrayRegion(handle_out, 0, 1, &value);
    return static_cast<jint>(vn97::GenerationStatus::kOk);
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeGenerationBindings_nativeNext(
    JNIEnv* env,
    jobject,
    jlong handle,
    jintArray token_out,
    jintArray eos_out) {
    if (!HasLength(env, token_out, 1) || !HasLength(env, eos_out, 1)) {
        return static_cast<jint>(vn97::GenerationStatus::kNullArgument);
    }
    if (handle <= 0) return static_cast<jint>(vn97::GenerationStatus::kInvalidHandle);
    std::uint32_t token = 0;
    int eos = 0;
    const int status = vn97_generation_next(
        static_cast<std::uint64_t>(handle),
        &token,
        &eos);
    if (status != 0) return status;
    if (token > static_cast<std::uint32_t>(std::numeric_limits<jint>::max())) {
        return static_cast<jint>(vn97::GenerationStatus::kInvalidConfig);
    }
    const jint token_value = static_cast<jint>(token);
    const jint eos_value = eos != 0 ? 1 : 0;
    env->SetIntArrayRegion(token_out, 0, 1, &token_value);
    env->SetIntArrayRegion(eos_out, 0, 1, &eos_value);
    return static_cast<jint>(vn97::GenerationStatus::kOk);
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeGenerationBindings_nativeDestroy(
    JNIEnv*, jobject, jlong handle) {
    if (handle <= 0) return static_cast<jint>(vn97::GenerationStatus::kInvalidHandle);
    return vn97_generation_destroy(static_cast<std::uint64_t>(handle));
}
