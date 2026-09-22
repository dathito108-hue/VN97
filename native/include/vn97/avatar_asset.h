#pragma once

#include <cstddef>
#include <cstdint>

namespace vn97 {

enum class AvatarAssetStatus {
    kOk = 0,
    kNullArgument,
    kBlobTooShort,
    kBadMagic,
    kUnsupportedVersion,
    kInvalidHeader,
    kInvalidGeometry,
    kInvalidRig,
    kChecksumMismatch,
    kSizeOverflow,
    kIndexOutOfRange,
    kNonFinite,
};

struct AvatarAssetView {
    const std::uint8_t* blob = nullptr;
    std::size_t blob_size = 0;
    std::uint32_t vertex_count = 0;
    std::uint32_t index_count = 0;
    std::uint32_t joint_count = 0;
    std::size_t vertex_offset = 0;
    std::size_t index_offset = 0;
    std::size_t joint_offset = 0;
};

struct AvatarVertex {
    float position[3] = {};
    float normal[3] = {};
    std::uint8_t joint_indices[4] = {};
    std::uint8_t joint_weights[4] = {};
};

struct AvatarJoint {
    std::int32_t parent = -1;
    float translation[3] = {};
    float rotation_xyzw[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    float scale[3] = {1.0f, 1.0f, 1.0f};
};

AvatarAssetStatus ParseAvatarAsset(
    const std::uint8_t* blob,
    std::size_t blob_size,
    AvatarAssetView* out);

AvatarAssetStatus ReadAvatarVertex(
    const AvatarAssetView& asset,
    std::size_t vertex_index,
    AvatarVertex* out);

AvatarAssetStatus ReadAvatarIndex(
    const AvatarAssetView& asset,
    std::size_t index_index,
    std::uint32_t* out);

AvatarAssetStatus ReadAvatarJoint(
    const AvatarAssetView& asset,
    std::size_t joint_index,
    AvatarJoint* out);

}  // namespace vn97

extern "C" {

int vn97_avatar_asset_parse(
    const std::uint8_t* blob,
    std::size_t blob_size,
    std::uint32_t* vertex_count,
    std::uint32_t* index_count,
    std::uint32_t* joint_count);

}
