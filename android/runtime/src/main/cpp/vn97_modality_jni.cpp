#include <jni.h>

#include "vn97/modality.h"

#include <cstddef>
#include <limits>
#include <vector>

namespace {
bool HasLength(JNIEnv* env, jarray array, jsize minimum) {
    return array != nullptr && env->GetArrayLength(array) >= minimum;
}
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeAudioModalityBindings_nativeFrameCount(
    JNIEnv* env,
    jobject,
    jint sample_count,
    jint frame_size,
    jint hop_size,
    jintArray out) {
    if (!HasLength(env, out, 1)) {
        return static_cast<jint>(vn97::ModalityStatus::kNullArgument);
    }
    if (sample_count <= 0 || frame_size <= 0 || hop_size <= 0) {
        return static_cast<jint>(vn97::ModalityStatus::kInvalidShape);
    }
    const std::size_t count = vn97_audio_frame_count(
        static_cast<std::size_t>(sample_count),
        static_cast<std::size_t>(frame_size),
        static_cast<std::size_t>(hop_size));
    if (
        count == 0 ||
        count > static_cast<std::size_t>(std::numeric_limits<jint>::max())
    ) {
        return static_cast<jint>(vn97::ModalityStatus::kInvalidShape);
    }
    const jint value = static_cast<jint>(count);
    env->SetIntArrayRegion(out, 0, 1, &value);
    return static_cast<jint>(vn97::ModalityStatus::kOk);
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeAudioModalityBindings_nativePrepare(
    JNIEnv* env,
    jobject,
    jshortArray pcm16,
    jint frame_size,
    jint hop_size,
    jfloat eps,
    jfloatArray out) {
    if (pcm16 == nullptr || out == nullptr) {
        return static_cast<jint>(vn97::ModalityStatus::kNullArgument);
    }
    if (frame_size <= 0 || hop_size <= 0) {
        return static_cast<jint>(vn97::ModalityStatus::kInvalidShape);
    }

    const jsize sample_count = env->GetArrayLength(pcm16);
    const jsize output_count = env->GetArrayLength(out);
    if (sample_count <= 0 || output_count <= 0) {
        return static_cast<jint>(vn97::ModalityStatus::kInvalidShape);
    }

    jshort* input = env->GetShortArrayElements(pcm16, nullptr);
    jfloat* output = env->GetFloatArrayElements(out, nullptr);
    if (input == nullptr || output == nullptr) {
        if (input != nullptr) env->ReleaseShortArrayElements(pcm16, input, JNI_ABORT);
        if (output != nullptr) env->ReleaseFloatArrayElements(out, output, JNI_ABORT);
        return static_cast<jint>(vn97::ModalityStatus::kNullArgument);
    }

    std::vector<float> waveform(static_cast<std::size_t>(sample_count));
    for (jsize i = 0; i < sample_count; ++i) {
        waveform[static_cast<std::size_t>(i)] =
            static_cast<float>(input[i]) / 32768.0f;
    }

    const int status = vn97_prepare_audio_frames_f32(
        waveform.data(),
        output,
        static_cast<std::size_t>(output_count),
        1,
        static_cast<std::size_t>(sample_count),
        static_cast<std::size_t>(frame_size),
        static_cast<std::size_t>(hop_size),
        eps);

    env->ReleaseShortArrayElements(pcm16, input, JNI_ABORT);
    env->ReleaseFloatArrayElements(
        out,
        output,
        status == static_cast<int>(vn97::ModalityStatus::kOk) ? 0 : JNI_ABORT);
    return status;
}
