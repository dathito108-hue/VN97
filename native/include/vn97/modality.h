#pragma once

#include "vn97/packed_ternary.h"

#include <cstddef>
#include <cstdint>

namespace vn97 {

enum class ModalityStatus {
    kOk = 0,
    kNullArgument,
    kInvalidShape,
    kInvalidParameter,
    kSizeOverflow,
    kOutputTooSmall,
    kInvalidModel,
    kBackendUnavailable,
    kNonFinite,
};

struct AudioProjectionView {
    std::uint32_t frame_size = 0;
    std::uint32_t d_model = 0;
    float rms_eps = 0.0f;
    PackedTernaryView projection;
    const float* norm_weight = nullptr;
};

std::size_t AudioFrameCount(
    std::size_t sample_count,
    std::size_t frame_size,
    std::size_t hop_size);

ModalityStatus PrepareAudioFramesF32(
    const float* waveform,
    float* output,
    std::size_t output_capacity,
    std::size_t batch,
    std::size_t sample_count,
    std::size_t frame_size,
    std::size_t hop_size,
    float eps);

ModalityStatus ValidateAudioProjection(
    const AudioProjectionView& view);

ModalityStatus ProjectAudioFramesF32(
    const AudioProjectionView& view,
    const float* frames,
    std::size_t frame_value_count,
    std::size_t frame_count,
    float* embeddings,
    std::size_t embedding_capacity,
    float* workspace,
    std::size_t workspace_count,
    PackedTernaryBackend backend = PackedTernaryBackend::kAuto);

std::size_t VisionPatchCount(
    std::size_t height,
    std::size_t width,
    std::size_t patch_size);

ModalityStatus PrepareVisionPatchesF32(
    const float* image_nchw,
    float* output,
    std::size_t output_capacity,
    std::size_t batch,
    std::size_t channels,
    std::size_t height,
    std::size_t width,
    std::size_t patch_size,
    float eps);

}  // namespace vn97

extern "C" {

std::size_t vn97_audio_frame_count(
    std::size_t sample_count,
    std::size_t frame_size,
    std::size_t hop_size);

int vn97_prepare_audio_frames_f32(
    const float* waveform,
    float* output,
    std::size_t output_capacity,
    std::size_t batch,
    std::size_t sample_count,
    std::size_t frame_size,
    std::size_t hop_size,
    float eps);

std::size_t vn97_vision_patch_count(
    std::size_t height,
    std::size_t width,
    std::size_t patch_size);

int vn97_prepare_vision_patches_f32(
    const float* image_nchw,
    float* output,
    std::size_t output_capacity,
    std::size_t batch,
    std::size_t channels,
    std::size_t height,
    std::size_t width,
    std::size_t patch_size,
    float eps);

}
