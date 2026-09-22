#pragma once

#include <cstddef>

namespace vn97 {

enum class ModalityStatus {
    kOk = 0,
    kNullArgument,
    kInvalidShape,
    kInvalidParameter,
    kSizeOverflow,
    kOutputTooSmall,
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
