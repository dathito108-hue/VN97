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


extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeVisionModalityBindings_nativePatchCount(
    JNIEnv* env,
    jobject,
    jint height,
    jint width,
    jint patch_size,
    jintArray out) {
    if (!HasLength(env, out, 1)) {
        return static_cast<jint>(vn97::ModalityStatus::kNullArgument);
    }
    if (height <= 0 || width <= 0 || patch_size <= 0) {
        return static_cast<jint>(vn97::ModalityStatus::kInvalidShape);
    }
    const std::size_t count = vn97_vision_patch_count(
        static_cast<std::size_t>(height),
        static_cast<std::size_t>(width),
        static_cast<std::size_t>(patch_size));
    if (
        count == 0 ||
        count > static_cast<std::size_t>(
            std::numeric_limits<jint>::max())
    ) {
        return static_cast<jint>(vn97::ModalityStatus::kInvalidShape);
    }
    const jint value = static_cast<jint>(count);
    env->SetIntArrayRegion(out, 0, 1, &value);
    return static_cast<jint>(vn97::ModalityStatus::kOk);
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeVisionModalityBindings_nativePrepareRgb888(
    JNIEnv* env,
    jobject,
    jbyteArray rgb888,
    jint width,
    jint height,
    jint patch_size,
    jfloat eps,
    jfloatArray out) {
    if (rgb888 == nullptr || out == nullptr) {
        return static_cast<jint>(vn97::ModalityStatus::kNullArgument);
    }
    if (width <= 0 || height <= 0 || patch_size <= 0) {
        return static_cast<jint>(vn97::ModalityStatus::kInvalidShape);
    }

    const std::size_t pixel_count =
        static_cast<std::size_t>(width) *
        static_cast<std::size_t>(height);
    if (
        pixel_count >
        std::numeric_limits<std::size_t>::max() / 3u
    ) {
        return static_cast<jint>(vn97::ModalityStatus::kSizeOverflow);
    }
    const std::size_t expected_bytes = pixel_count * 3u;
    if (
        expected_bytes >
        static_cast<std::size_t>(
            std::numeric_limits<jsize>::max()) ||
        env->GetArrayLength(rgb888) !=
            static_cast<jsize>(expected_bytes)
    ) {
        return static_cast<jint>(vn97::ModalityStatus::kInvalidShape);
    }

    const std::size_t output_count =
        static_cast<std::size_t>(env->GetArrayLength(out));
    jbyte* input =
        env->GetByteArrayElements(rgb888, nullptr);
    jfloat* output =
        env->GetFloatArrayElements(out, nullptr);
    if (input == nullptr || output == nullptr) {
        if (input != nullptr) {
            env->ReleaseByteArrayElements(
                rgb888,
                input,
                JNI_ABORT);
        }
        if (output != nullptr) {
            env->ReleaseFloatArrayElements(
                out,
                output,
                JNI_ABORT);
        }
        return static_cast<jint>(vn97::ModalityStatus::kNullArgument);
    }

    std::vector<float> nchw(pixel_count * 3u);
    for (std::size_t index = 0; index < pixel_count; ++index) {
        const std::size_t source = index * 3u;
        const float r = static_cast<float>(
            static_cast<std::uint8_t>(input[source + 0u])) /
            255.0f;
        const float g = static_cast<float>(
            static_cast<std::uint8_t>(input[source + 1u])) /
            255.0f;
        const float b = static_cast<float>(
            static_cast<std::uint8_t>(input[source + 2u])) /
            255.0f;
        nchw[index] = r;
        nchw[pixel_count + index] = g;
        nchw[pixel_count * 2u + index] = b;
    }

    const int status = vn97_prepare_vision_patches_f32(
        nchw.data(),
        output,
        output_count,
        1,
        3,
        static_cast<std::size_t>(height),
        static_cast<std::size_t>(width),
        static_cast<std::size_t>(patch_size),
        eps);

    env->ReleaseByteArrayElements(
        rgb888,
        input,
        JNI_ABORT);
    env->ReleaseFloatArrayElements(
        out,
        output,
        status == static_cast<int>(vn97::ModalityStatus::kOk)
            ? 0
            : JNI_ABORT);
    return status;
}
