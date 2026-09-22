#include <jni.h>

#include "vn97/runtime.h"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <limits>
#include <unistd.h>

namespace {
constexpr jint kNullArgument = 1;
constexpr jint kInvalidConfig = 2;
constexpr jint kCounterOverflow = 9;

bool HasLength(JNIEnv* env, jarray array, jsize minimum) {
    return array != nullptr && env->GetArrayLength(array) >= minimum;
}
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeRuntimeBindings_nativeCreate(
    JNIEnv* env,
    jobject,
    jintArray config,
    jlongArray handle_out) {
    if (!HasLength(env, config, 6) || !HasLength(env, handle_out, 1)) return kNullArgument;
    jint values[6] = {};
    env->GetIntArrayRegion(config, 0, 6, values);
    vn97_runtime_config native_config{
        static_cast<std::uint32_t>(values[0]),
        static_cast<std::uint32_t>(values[1]),
        static_cast<std::uint32_t>(values[2]),
        static_cast<std::uint32_t>(values[3]),
        values[4],
        values[5],
    };
    if (values[0] <= 0 || values[1] <= 0 || values[2] <= 0 || values[3] <= 0) return kInvalidConfig;
    std::uint64_t handle = 0;
    const int status = vn97_runtime_create(&native_config, &handle);
    if (status == 0) {
        if (handle > static_cast<std::uint64_t>(std::numeric_limits<jlong>::max())) {
            vn97_runtime_destroy(handle);
            return kCounterOverflow;
        }
        const jlong result = static_cast<jlong>(handle);
        env->SetLongArrayRegion(handle_out, 0, 1, &result);
    }
    return status;
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeRuntimeBindings_nativeRestore(
    JNIEnv* env,
    jobject,
    jbyteArray blob,
    jlongArray handle_out) {
    if (blob == nullptr || !HasLength(env, handle_out, 1)) return kNullArgument;
    const jsize size = env->GetArrayLength(blob);
    jbyte* bytes = env->GetByteArrayElements(blob, nullptr);
    if (bytes == nullptr) return kNullArgument;
    std::uint64_t handle = 0;
    const int status = vn97_runtime_restore(
        reinterpret_cast<const std::uint8_t*>(bytes),
        static_cast<std::size_t>(size),
        &handle);
    env->ReleaseByteArrayElements(blob, bytes, JNI_ABORT);
    if (status == 0) {
        if (handle > static_cast<std::uint64_t>(std::numeric_limits<jlong>::max())) {
            vn97_runtime_destroy(handle);
            return kCounterOverflow;
        }
        const jlong result = static_cast<jlong>(handle);
        env->SetLongArrayRegion(handle_out, 0, 1, &result);
    }
    return status;
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeRuntimeBindings_nativeDestroy(JNIEnv*, jobject, jlong handle) {
    return vn97_runtime_destroy(static_cast<std::uint64_t>(handle));
}
extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeRuntimeBindings_nativeActivate(JNIEnv*, jobject, jlong handle) {
    return vn97_runtime_activate(static_cast<std::uint64_t>(handle));
}
extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeRuntimeBindings_nativeSuspend(JNIEnv*, jobject, jlong handle) {
    return vn97_runtime_suspend(static_cast<std::uint64_t>(handle));
}
extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeRuntimeBindings_nativeResume(JNIEnv*, jobject, jlong handle) {
    return vn97_runtime_resume(static_cast<std::uint64_t>(handle));
}
extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeRuntimeBindings_nativeAdvance(
    JNIEnv*, jobject, jlong handle, jlong token_count) {
    if (token_count < 0) return kInvalidConfig;
    return vn97_runtime_advance(
        static_cast<std::uint64_t>(handle),
        static_cast<std::uint64_t>(token_count));
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeRuntimeBindings_nativeInfo(
    JNIEnv* env,
    jobject,
    jlong handle,
    jintArray ints_out,
    jlongArray longs_out) {
    if (!HasLength(env, ints_out, 9) || !HasLength(env, longs_out, 2)) return kNullArgument;
    vn97_runtime_info info{};
    const int status = vn97_runtime_info_get(static_cast<std::uint64_t>(handle), &info);
    if (status != 0) return status;
    const jint ints[9] = {
        static_cast<jint>(info.layers),
        static_cast<jint>(info.batch),
        static_cast<jint>(info.d_model),
        static_cast<jint>(info.d_state),
        info.requested_recurrent_backend,
        info.requested_packed_backend,
        info.resolved_recurrent_backend,
        info.resolved_packed_backend,
        info.lifecycle,
    };
    const jlong longs[2] = {
        static_cast<jlong>(info.state_count),
        static_cast<jlong>(info.sequence_position),
    };
    env->SetIntArrayRegion(ints_out, 0, 9, ints);
    env->SetLongArrayRegion(longs_out, 0, 2, longs);
    return 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeRuntimeBindings_nativeStateRead(
    JNIEnv* env, jobject, jlong handle, jfloatArray out) {
    if (out == nullptr) return kNullArgument;
    const jsize count = env->GetArrayLength(out);
    jfloat* values = env->GetFloatArrayElements(out, nullptr);
    if (values == nullptr) return kNullArgument;
    const int status = vn97_runtime_state_read(
        static_cast<std::uint64_t>(handle),
        values,
        static_cast<std::size_t>(count));
    env->ReleaseFloatArrayElements(out, values, status == 0 ? 0 : JNI_ABORT);
    return status;
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeRuntimeBindings_nativeStateWrite(
    JNIEnv* env, jobject, jlong handle, jfloatArray state) {
    if (state == nullptr) return kNullArgument;
    const jsize count = env->GetArrayLength(state);
    jfloat* values = env->GetFloatArrayElements(state, nullptr);
    if (values == nullptr) return kNullArgument;
    const int status = vn97_runtime_state_write(
        static_cast<std::uint64_t>(handle),
        values,
        static_cast<std::size_t>(count));
    env->ReleaseFloatArrayElements(state, values, JNI_ABORT);
    return status;
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeRuntimeBindings_nativeCheckpointSize(
    JNIEnv* env, jobject, jlong handle, jlongArray out) {
    if (!HasLength(env, out, 1)) return kNullArgument;
    std::size_t size = 0;
    const int status = vn97_runtime_checkpoint_size(static_cast<std::uint64_t>(handle), &size);
    if (status == 0) {
        const jlong result = static_cast<jlong>(size);
        env->SetLongArrayRegion(out, 0, 1, &result);
    }
    return status;
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeRuntimeBindings_nativeCheckpointWrite(
    JNIEnv* env,
    jobject,
    jlong handle,
    jbyteArray out,
    jlongArray written_out) {
    if (out == nullptr || !HasLength(env, written_out, 1)) return kNullArgument;
    const jsize capacity = env->GetArrayLength(out);
    jbyte* bytes = env->GetByteArrayElements(out, nullptr);
    if (bytes == nullptr) return kNullArgument;
    std::size_t written = 0;
    const int status = vn97_runtime_checkpoint_write(
        static_cast<std::uint64_t>(handle),
        reinterpret_cast<std::uint8_t*>(bytes),
        static_cast<std::size_t>(capacity),
        &written);
    env->ReleaseByteArrayElements(out, bytes, status == 0 ? 0 : JNI_ABORT);
    if (status == 0) {
        const jlong result = static_cast<jlong>(written);
        env->SetLongArrayRegion(written_out, 0, 1, &result);
    }
    return status;
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_runtime_NativeRuntimeBindings_nativeFsyncDirectory(
    JNIEnv* env,
    jobject,
    jstring path) {
    if (path == nullptr) return EINVAL;
    const char* utf = env->GetStringUTFChars(path, nullptr);
    if (utf == nullptr) return ENOMEM;
    const int fd = open(utf, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    const int open_errno = fd < 0 ? errno : 0;
    env->ReleaseStringUTFChars(path, utf);
    if (fd < 0) return open_errno;
    const int status = fsync(fd) == 0 ? 0 : errno;
    close(fd);
    return status;
}
