#include "vn97/modality.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace vn97 {
namespace {

bool MulOverflows(std::size_t a, std::size_t b) {
    return b != 0 && a > std::numeric_limits<std::size_t>::max() / b;
}

bool ValidEps(float eps) {
    return std::isfinite(eps) && eps > 0.0f;
}

float Coord(std::size_t index, std::size_t count) {
    if (count <= 1) return 0.0f;
    return -1.0f + 2.0f * static_cast<float>(index) /
        static_cast<float>(count - 1);
}

bool FiniteArray(const float* values, std::size_t count) {
    if (values == nullptr) return false;
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i])) return false;
    }
    return true;
}

bool MatrixShape(
    const PackedTernaryView& matrix,
    std::uint32_t rows,
    std::uint32_t cols) {
    return matrix.rows == rows &&
        matrix.cols == cols &&
        matrix.rows != 0 &&
        matrix.cols != 0 &&
        matrix.scale_bytes != nullptr &&
        matrix.packed_data != nullptr &&
        matrix.tile_rows != 0 &&
        matrix.tile_cols != 0 &&
        matrix.padded_rows >= matrix.rows &&
        matrix.padded_cols >= matrix.cols &&
        matrix.padded_rows % matrix.tile_rows == 0 &&
        matrix.padded_cols % matrix.tile_cols == 0;
}

void RmsNorm(
    const float* input,
    const float* weight,
    std::size_t count,
    float eps,
    float* output) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < count; ++i) {
        sum += input[i] * input[i];
    }
    const float inv =
        1.0f / std::sqrt(sum / static_cast<float>(count) + eps);
    for (std::size_t i = 0; i < count; ++i) {
        output[i] = weight[i] * input[i] * inv;
    }
}

}  // namespace

std::size_t AudioFrameCount(
    std::size_t sample_count,
    std::size_t frame_size,
    std::size_t hop_size) {
    if (frame_size == 0 || hop_size == 0 || sample_count < frame_size) {
        return 0;
    }
    return 1 + (sample_count - frame_size) / hop_size;
}

ModalityStatus PrepareAudioFramesF32(
    const float* waveform,
    float* output,
    std::size_t output_capacity,
    std::size_t batch,
    std::size_t sample_count,
    std::size_t frame_size,
    std::size_t hop_size,
    float eps) {
    if (waveform == nullptr || output == nullptr) {
        return ModalityStatus::kNullArgument;
    }
    if (batch == 0 || sample_count == 0 || frame_size == 0 || hop_size == 0) {
        return ModalityStatus::kInvalidShape;
    }
    if (!ValidEps(eps)) {
        return ModalityStatus::kInvalidParameter;
    }

    const std::size_t frame_count =
        AudioFrameCount(sample_count, frame_size, hop_size);
    if (frame_count == 0) {
        return ModalityStatus::kInvalidShape;
    }
    if (MulOverflows(batch, frame_count) ||
        MulOverflows(batch * frame_count, frame_size)) {
        return ModalityStatus::kSizeOverflow;
    }
    const std::size_t required =
        batch * frame_count * frame_size;
    if (output_capacity < required) {
        return ModalityStatus::kOutputTooSmall;
    }

    for (std::size_t b = 0; b < batch; ++b) {
        const float* batch_input =
            waveform + b * sample_count;
        for (std::size_t t = 0; t < frame_count; ++t) {
            const float* frame =
                batch_input + t * hop_size;
            float* out =
                output + (b * frame_count + t) * frame_size;

            float mean = 0.0f;
            for (std::size_t i = 0; i < frame_size; ++i) {
                mean += frame[i];
            }
            mean /= static_cast<float>(frame_size);

            float square_sum = 0.0f;
            for (std::size_t i = 0; i < frame_size; ++i) {
                const float centered = frame[i] - mean;
                out[i] = centered;
                square_sum += centered * centered;
            }
            const float mean_square =
                square_sum / static_cast<float>(frame_size);
            const float inv_rms =
                1.0f / std::sqrt(mean_square + eps);
            for (std::size_t i = 0; i < frame_size; ++i) {
                out[i] *= inv_rms;
            }
        }
    }
    return ModalityStatus::kOk;
}

ModalityStatus ValidateAudioProjection(
    const AudioProjectionView& view) {
    if (
        view.frame_size == 0 ||
        view.d_model == 0 ||
        !ValidEps(view.rms_eps) ||
        view.norm_weight == nullptr ||
        !MatrixShape(
            view.projection,
            view.d_model,
            view.frame_size)
    ) {
        return ModalityStatus::kInvalidModel;
    }
    if (!FiniteArray(view.norm_weight, view.d_model)) {
        return ModalityStatus::kNonFinite;
    }
    return ModalityStatus::kOk;
}

ModalityStatus ProjectAudioFramesF32(
    const AudioProjectionView& view,
    const float* frames,
    std::size_t frame_value_count,
    std::size_t frame_count,
    float* embeddings,
    std::size_t embedding_capacity,
    float* workspace,
    std::size_t workspace_count,
    PackedTernaryBackend backend) {
    if (
        frames == nullptr ||
        embeddings == nullptr ||
        workspace == nullptr
    ) {
        return ModalityStatus::kNullArgument;
    }
    const auto model_status = ValidateAudioProjection(view);
    if (model_status != ModalityStatus::kOk) return model_status;
    if (frame_count == 0) return ModalityStatus::kInvalidShape;

    if (
        MulOverflows(frame_count, view.frame_size) ||
        MulOverflows(frame_count, view.d_model)
    ) {
        return ModalityStatus::kSizeOverflow;
    }
    const std::size_t required_input =
        frame_count * view.frame_size;
    const std::size_t required_output =
        frame_count * view.d_model;
    if (
        frame_value_count != required_input ||
        embedding_capacity < required_output ||
        workspace_count < view.d_model
    ) {
        return ModalityStatus::kOutputTooSmall;
    }
    if (!FiniteArray(frames, required_input)) {
        return ModalityStatus::kNonFinite;
    }

    const auto resolved = ResolvePackedTernaryBackend(backend);
    if (!PackedTernaryBackendAvailable(resolved)) {
        return ModalityStatus::kBackendUnavailable;
    }

    for (std::size_t frame = 0; frame < frame_count; ++frame) {
        const float* input =
            frames + frame * view.frame_size;
        const auto status = PackedTernaryMatVecF32WithBackend(
            view.projection,
            input,
            nullptr,
            workspace,
            resolved);
        if (status == PackedTernaryStatus::kBackendUnavailable) {
            return ModalityStatus::kBackendUnavailable;
        }
        if (status != PackedTernaryStatus::kOk) {
            return ModalityStatus::kInvalidModel;
        }
        RmsNorm(
            workspace,
            view.norm_weight,
            view.d_model,
            view.rms_eps,
            embeddings + frame * view.d_model);
    }
    return ModalityStatus::kOk;
}

std::size_t VisionPatchCount(
    std::size_t height,
    std::size_t width,
    std::size_t patch_size) {
    if (
        patch_size == 0 ||
        height < patch_size ||
        width < patch_size ||
        height % patch_size != 0 ||
        width % patch_size != 0
    ) {
        return 0;
    }
    const std::size_t rows = height / patch_size;
    const std::size_t cols = width / patch_size;
    if (MulOverflows(rows, cols)) {
        return 0;
    }
    return rows * cols;
}

ModalityStatus PrepareVisionPatchesF32(
    const float* image_nchw,
    float* output,
    std::size_t output_capacity,
    std::size_t batch,
    std::size_t channels,
    std::size_t height,
    std::size_t width,
    std::size_t patch_size,
    float eps) {
    if (image_nchw == nullptr || output == nullptr) {
        return ModalityStatus::kNullArgument;
    }
    if (
        batch == 0 ||
        channels == 0 ||
        height == 0 ||
        width == 0 ||
        patch_size == 0
    ) {
        return ModalityStatus::kInvalidShape;
    }
    if (!ValidEps(eps)) {
        return ModalityStatus::kInvalidParameter;
    }

    const std::size_t patch_count =
        VisionPatchCount(height, width, patch_size);
    if (patch_count == 0) {
        return ModalityStatus::kInvalidShape;
    }
    if (
        MulOverflows(patch_size, patch_size) ||
        MulOverflows(channels, patch_size * patch_size)
    ) {
        return ModalityStatus::kSizeOverflow;
    }
    const std::size_t patch_values =
        channels * patch_size * patch_size;
    if (
        patch_values >
        std::numeric_limits<std::size_t>::max() - 2
    ) {
        return ModalityStatus::kSizeOverflow;
    }
    const std::size_t features =
        patch_values + 2;

    if (
        MulOverflows(batch, patch_count) ||
        MulOverflows(batch * patch_count, features) ||
        MulOverflows(height, width) ||
        MulOverflows(channels, height * width) ||
        MulOverflows(batch, channels * height * width)
    ) {
        return ModalityStatus::kSizeOverflow;
    }

    const std::size_t required =
        batch * patch_count * features;
    if (output_capacity < required) {
        return ModalityStatus::kOutputTooSmall;
    }

    const std::size_t rows =
        height / patch_size;
    const std::size_t cols =
        width / patch_size;
    const std::size_t image_stride =
        channels * height * width;
    const std::size_t channel_stride =
        height * width;

    for (std::size_t b = 0; b < batch; ++b) {
        const float* image =
            image_nchw + b * image_stride;
        for (std::size_t pr = 0; pr < rows; ++pr) {
            for (std::size_t pc = 0; pc < cols; ++pc) {
                const std::size_t patch_index =
                    pr * cols + pc;
                float* out =
                    output +
                    (b * patch_count + patch_index) *
                        features;

                float mean = 0.0f;
                std::size_t out_index = 0;
                for (std::size_t c = 0; c < channels; ++c) {
                    const float* channel =
                        image + c * channel_stride;
                    for (std::size_t py = 0; py < patch_size; ++py) {
                        const std::size_t y =
                            pr * patch_size + py;
                        for (std::size_t px = 0; px < patch_size; ++px) {
                            const std::size_t x =
                                pc * patch_size + px;
                            const float value =
                                channel[y * width + x];
                            out[out_index++] = value;
                            mean += value;
                        }
                    }
                }

                mean /= static_cast<float>(patch_values);

                float square_sum = 0.0f;
                for (std::size_t i = 0; i < patch_values; ++i) {
                    const float centered =
                        out[i] - mean;
                    out[i] = centered;
                    square_sum += centered * centered;
                }

                const float mean_square =
                    square_sum /
                    static_cast<float>(patch_values);
                const float inv_rms =
                    1.0f / std::sqrt(mean_square + eps);
                for (std::size_t i = 0; i < patch_values; ++i) {
                    out[i] *= inv_rms;
                }

                out[patch_values] =
                    Coord(pr, rows);
                out[patch_values + 1] =
                    Coord(pc, cols);
            }
        }
    }
    return ModalityStatus::kOk;
}

}  // namespace vn97

extern "C" {

std::size_t vn97_audio_frame_count(
    std::size_t sample_count,
    std::size_t frame_size,
    std::size_t hop_size) {
    return vn97::AudioFrameCount(
        sample_count, frame_size, hop_size);
}

int vn97_prepare_audio_frames_f32(
    const float* waveform,
    float* output,
    std::size_t output_capacity,
    std::size_t batch,
    std::size_t sample_count,
    std::size_t frame_size,
    std::size_t hop_size,
    float eps) {
    return static_cast<int>(
        vn97::PrepareAudioFramesF32(
            waveform,
            output,
            output_capacity,
            batch,
            sample_count,
            frame_size,
            hop_size,
            eps
        )
    );
}

std::size_t vn97_vision_patch_count(
    std::size_t height,
    std::size_t width,
    std::size_t patch_size) {
    return vn97::VisionPatchCount(
        height, width, patch_size);
}

int vn97_prepare_vision_patches_f32(
    const float* image_nchw,
    float* output,
    std::size_t output_capacity,
    std::size_t batch,
    std::size_t channels,
    std::size_t height,
    std::size_t width,
    std::size_t patch_size,
    float eps) {
    return static_cast<int>(
        vn97::PrepareVisionPatchesF32(
            image_nchw,
            output,
            output_capacity,
            batch,
            channels,
            height,
            width,
            patch_size,
            eps
        )
    );
}

}
