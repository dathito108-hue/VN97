#include "vn97/avatar_asset.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace vn97 {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic = {'V','N','9','7','A','V','1','\0'};
constexpr std::uint16_t kVersion = 1;
constexpr std::uint16_t kHeaderSize = 80;
constexpr std::uint32_t kVertexStride = 32;
constexpr std::uint32_t kJointStride = 48;
constexpr std::uint32_t kMaxVertices = 200000;
constexpr std::uint32_t kMaxIndices = 600000;
constexpr std::uint32_t kMaxJoints = 128;
constexpr float kMaxCoordinate = 1000000.0f;
constexpr float kMaxScale = 1000.0f;

std::uint16_t ReadU16(const std::uint8_t* in) {
    return static_cast<std::uint16_t>(in[0]) |
           (static_cast<std::uint16_t>(in[1]) << 8);
}

std::uint32_t ReadU32(const std::uint8_t* in) {
    std::uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
        value |= static_cast<std::uint32_t>(in[i]) << (8 * i);
    }
    return value;
}

std::uint64_t ReadU64(const std::uint8_t* in) {
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(in[i]) << (8 * i);
    }
    return value;
}

std::int32_t ReadI32(const std::uint8_t* in) {
    const std::uint32_t bits = ReadU32(in);
    std::int32_t value = 0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float ReadF32(const std::uint8_t* in) {
    const std::uint32_t bits = ReadU32(in);
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::uint32_t Crc32(const std::uint8_t* data, std::size_t size) {
    std::uint32_t crc = 0xffffffffu;
    for (std::size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            const std::uint32_t mask = 0u - (crc & 1u);
            crc = (crc >> 1) ^ (0xedb88320u & mask);
        }
    }
    return ~crc;
}

bool MulOverflow(std::size_t a, std::size_t b, std::size_t* out) {
    if (out == nullptr) return true;
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) return true;
    *out = a * b;
    return false;
}

bool AddOverflow(std::size_t a, std::size_t b, std::size_t* out) {
    if (out == nullptr) return true;
    if (b > std::numeric_limits<std::size_t>::max() - a) return true;
    *out = a + b;
    return false;
}

bool FiniteBounded(float value, float abs_limit) {
    return std::isfinite(value) && std::fabs(value) <= abs_limit;
}

AvatarAssetStatus DecodeVertexUnchecked(
    const AvatarAssetView& asset,
    std::size_t vertex_index,
    AvatarVertex* out) {
    const std::uint8_t* ptr =
        asset.blob + asset.vertex_offset + vertex_index * kVertexStride;
    for (int i = 0; i < 3; ++i) out->position[i] = ReadF32(ptr + i * 4);
    for (int i = 0; i < 3; ++i) out->normal[i] = ReadF32(ptr + 12 + i * 4);
    std::copy(ptr + 24, ptr + 28, out->joint_indices);
    std::copy(ptr + 28, ptr + 32, out->joint_weights);
    return AvatarAssetStatus::kOk;
}

AvatarAssetStatus DecodeJointUnchecked(
    const AvatarAssetView& asset,
    std::size_t joint_index,
    AvatarJoint* out) {
    const std::uint8_t* ptr =
        asset.blob + asset.joint_offset + joint_index * kJointStride;
    out->parent = ReadI32(ptr);
    for (int i = 0; i < 3; ++i) out->translation[i] = ReadF32(ptr + 4 + i * 4);
    for (int i = 0; i < 4; ++i) out->rotation_xyzw[i] = ReadF32(ptr + 16 + i * 4);
    for (int i = 0; i < 3; ++i) out->scale[i] = ReadF32(ptr + 32 + i * 4);
    return AvatarAssetStatus::kOk;
}

}  // namespace

AvatarAssetStatus ParseAvatarAsset(
    const std::uint8_t* blob,
    std::size_t blob_size,
    AvatarAssetView* out) {
    if (blob == nullptr || out == nullptr) return AvatarAssetStatus::kNullArgument;
    *out = AvatarAssetView{};
    if (blob_size < kHeaderSize) return AvatarAssetStatus::kBlobTooShort;
    if (!std::equal(kMagic.begin(), kMagic.end(), blob)) return AvatarAssetStatus::kBadMagic;
    if (ReadU16(blob + 8) != kVersion) return AvatarAssetStatus::kUnsupportedVersion;
    if (ReadU16(blob + 10) != kHeaderSize) return AvatarAssetStatus::kInvalidHeader;
    if (ReadU32(blob + 12) != 0 || ReadU32(blob + 36) != 0) {
        return AvatarAssetStatus::kInvalidHeader;
    }
    if (ReadU32(blob + 28) != kVertexStride || ReadU32(blob + 32) != kJointStride) {
        return AvatarAssetStatus::kInvalidHeader;
    }
    if (ReadU32(blob + 76) != Crc32(blob, 76)) {
        return AvatarAssetStatus::kChecksumMismatch;
    }

    const std::uint32_t vertex_count = ReadU32(blob + 16);
    const std::uint32_t index_count = ReadU32(blob + 20);
    const std::uint32_t joint_count = ReadU32(blob + 24);
    if (vertex_count < 3 || vertex_count > kMaxVertices ||
        index_count < 3 || index_count > kMaxIndices || index_count % 3 != 0 ||
        joint_count > kMaxJoints) {
        return AvatarAssetStatus::kInvalidGeometry;
    }

    std::size_t vertex_bytes = 0;
    std::size_t index_bytes = 0;
    std::size_t joint_bytes = 0;
    if (MulOverflow(vertex_count, kVertexStride, &vertex_bytes) ||
        MulOverflow(index_count, sizeof(std::uint32_t), &index_bytes) ||
        MulOverflow(joint_count, kJointStride, &joint_bytes)) {
        return AvatarAssetStatus::kSizeOverflow;
    }
    std::size_t expected_index_offset = 0;
    std::size_t expected_joint_offset = 0;
    std::size_t expected_total = 0;
    if (AddOverflow(kHeaderSize, vertex_bytes, &expected_index_offset) ||
        AddOverflow(expected_index_offset, index_bytes, &expected_joint_offset) ||
        AddOverflow(expected_joint_offset, joint_bytes, &expected_total)) {
        return AvatarAssetStatus::kSizeOverflow;
    }

    const std::uint64_t vertex_offset_u64 = ReadU64(blob + 40);
    const std::uint64_t index_offset_u64 = ReadU64(blob + 48);
    const std::uint64_t joint_offset_u64 = ReadU64(blob + 56);
    const std::uint64_t total_size_u64 = ReadU64(blob + 64);
    if (vertex_offset_u64 != kHeaderSize ||
        index_offset_u64 != expected_index_offset ||
        joint_offset_u64 != expected_joint_offset ||
        total_size_u64 != expected_total ||
        expected_total != blob_size) {
        return AvatarAssetStatus::kInvalidHeader;
    }
    if (ReadU32(blob + 72) != Crc32(blob + kHeaderSize, blob_size - kHeaderSize)) {
        return AvatarAssetStatus::kChecksumMismatch;
    }

    AvatarAssetView view;
    view.blob = blob;
    view.blob_size = blob_size;
    view.vertex_count = vertex_count;
    view.index_count = index_count;
    view.joint_count = joint_count;
    view.vertex_offset = kHeaderSize;
    view.index_offset = expected_index_offset;
    view.joint_offset = expected_joint_offset;

    for (std::size_t i = 0; i < vertex_count; ++i) {
        AvatarVertex vertex;
        DecodeVertexUnchecked(view, i, &vertex);
        float normal_len2 = 0.0f;
        for (int c = 0; c < 3; ++c) {
            if (!FiniteBounded(vertex.position[c], kMaxCoordinate) ||
                !FiniteBounded(vertex.normal[c], 2.0f)) {
                return AvatarAssetStatus::kNonFinite;
            }
            normal_len2 += vertex.normal[c] * vertex.normal[c];
        }
        if (!std::isfinite(normal_len2) || normal_len2 < 0.25f || normal_len2 > 2.25f) {
            return AvatarAssetStatus::kInvalidGeometry;
        }
        std::uint32_t weight_sum = 0;
        for (int slot = 0; slot < 4; ++slot) {
            weight_sum += vertex.joint_weights[slot];
            if (vertex.joint_weights[slot] != 0 &&
                vertex.joint_indices[slot] >= joint_count) {
                return AvatarAssetStatus::kInvalidRig;
            }
        }
        if ((joint_count == 0 && weight_sum != 0) ||
            (joint_count != 0 && weight_sum != 255)) {
            return AvatarAssetStatus::kInvalidRig;
        }
    }

    for (std::size_t i = 0; i < index_count; ++i) {
        const std::uint32_t index = ReadU32(blob + view.index_offset + i * 4);
        if (index >= vertex_count) return AvatarAssetStatus::kIndexOutOfRange;
    }

    bool has_root = joint_count == 0;
    for (std::size_t i = 0; i < joint_count; ++i) {
        const std::uint8_t* ptr = blob + view.joint_offset + i * kJointStride;
        AvatarJoint joint;
        DecodeJointUnchecked(view, i, &joint);
        if (ReadU32(ptr + 44) != 0) return AvatarAssetStatus::kInvalidRig;
        if (joint.parent == -1) {
            has_root = true;
        } else if (joint.parent < 0 || static_cast<std::size_t>(joint.parent) >= i) {
            return AvatarAssetStatus::kInvalidRig;
        }
        float quat_len2 = 0.0f;
        for (float value : joint.translation) {
            if (!FiniteBounded(value, kMaxCoordinate)) return AvatarAssetStatus::kNonFinite;
        }
        for (float value : joint.rotation_xyzw) {
            if (!std::isfinite(value)) return AvatarAssetStatus::kNonFinite;
            quat_len2 += value * value;
        }
        if (!std::isfinite(quat_len2) || quat_len2 < 0.98f || quat_len2 > 1.02f) {
            return AvatarAssetStatus::kInvalidRig;
        }
        for (float value : joint.scale) {
            if (!std::isfinite(value)) return AvatarAssetStatus::kNonFinite;
            if (value <= 0.0f || value > kMaxScale) return AvatarAssetStatus::kInvalidRig;
        }
    }
    if (!has_root) return AvatarAssetStatus::kInvalidRig;

    *out = view;
    return AvatarAssetStatus::kOk;
}

AvatarAssetStatus ReadAvatarVertex(
    const AvatarAssetView& asset,
    std::size_t vertex_index,
    AvatarVertex* out) {
    if (out == nullptr || asset.blob == nullptr) return AvatarAssetStatus::kNullArgument;
    if (vertex_index >= asset.vertex_count) return AvatarAssetStatus::kIndexOutOfRange;
    return DecodeVertexUnchecked(asset, vertex_index, out);
}

AvatarAssetStatus ReadAvatarIndex(
    const AvatarAssetView& asset,
    std::size_t index_index,
    std::uint32_t* out) {
    if (out == nullptr || asset.blob == nullptr) return AvatarAssetStatus::kNullArgument;
    if (index_index >= asset.index_count) return AvatarAssetStatus::kIndexOutOfRange;
    *out = ReadU32(asset.blob + asset.index_offset + index_index * 4);
    return AvatarAssetStatus::kOk;
}

AvatarAssetStatus ReadAvatarJoint(
    const AvatarAssetView& asset,
    std::size_t joint_index,
    AvatarJoint* out) {
    if (out == nullptr || asset.blob == nullptr) return AvatarAssetStatus::kNullArgument;
    if (joint_index >= asset.joint_count) return AvatarAssetStatus::kIndexOutOfRange;
    return DecodeJointUnchecked(asset, joint_index, out);
}

}  // namespace vn97

extern "C" {

int vn97_avatar_asset_parse(
    const std::uint8_t* blob,
    std::size_t blob_size,
    std::uint32_t* vertex_count,
    std::uint32_t* index_count,
    std::uint32_t* joint_count) {
    if (vertex_count == nullptr || index_count == nullptr || joint_count == nullptr) {
        return static_cast<int>(vn97::AvatarAssetStatus::kNullArgument);
    }
    vn97::AvatarAssetView view;
    const auto status = vn97::ParseAvatarAsset(blob, blob_size, &view);
    if (status == vn97::AvatarAssetStatus::kOk) {
        *vertex_count = view.vertex_count;
        *index_count = view.index_count;
        *joint_count = view.joint_count;
    }
    return static_cast<int>(status);
}

}
