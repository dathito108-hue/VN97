#include <jni.h>

#include "vn97/avatar_asset.h"

#include <cstddef>
#include <cstdint>

namespace {
constexpr jint kNullArgument = 1;
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_vn97_avatar_NativeAvatarAssetBindings_nativeValidate(
    JNIEnv* env,
    jobject,
    jbyteArray blob,
    jintArray metadata_out) {
    if (blob == nullptr || metadata_out == nullptr ||
        env->GetArrayLength(metadata_out) < 3) {
        return kNullArgument;
    }
    const jsize size = env->GetArrayLength(blob);
    jbyte* bytes = env->GetByteArrayElements(blob, nullptr);
    if (bytes == nullptr) return kNullArgument;

    vn97::AvatarAssetView view;
    const auto status = vn97::ParseAvatarAsset(
        reinterpret_cast<const std::uint8_t*>(bytes),
        static_cast<std::size_t>(size),
        &view);
    env->ReleaseByteArrayElements(blob, bytes, JNI_ABORT);

    if (status == vn97::AvatarAssetStatus::kOk) {
        const jint metadata[3] = {
            static_cast<jint>(view.vertex_count),
            static_cast<jint>(view.index_count),
            static_cast<jint>(view.joint_count),
        };
        env->SetIntArrayRegion(metadata_out, 0, 3, metadata);
    }
    return static_cast<jint>(status);
}
