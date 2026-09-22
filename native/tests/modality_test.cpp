#include "vn97/modality.h"

#include <cassert>
#include <cmath>
#include <vector>

namespace {

void AssertNear(
    float actual,
    float expected,
    float tolerance = 2e-5f) {
    assert(
        std::fabs(actual - expected)
        <= tolerance
    );
}

}  // namespace

int main() {
    assert(
        vn97::AudioFrameCount(
            40, 16, 8
        ) == 4
    );
    assert(
        vn97::AudioFrameCount(
            8, 16, 8
        ) == 0
    );

    std::vector<float> waveform(40);
    for (
        std::size_t i = 0;
        i < waveform.size();
        ++i
    ) {
        waveform[i] =
            static_cast<float>(i)
            * 0.25f
            - 3.0f;
    }

    std::vector<float> audio(
        4 * 16
    );
    assert(
        vn97::PrepareAudioFramesF32(
            waveform.data(),
            audio.data(),
            audio.size(),
            1,
            40,
            16,
            8,
            1e-5f
        ) ==
        vn97::ModalityStatus::kOk
    );

    for (
        std::size_t frame = 0;
        frame < 4;
        ++frame
    ) {
        float mean = 0.0f;
        float mean_square = 0.0f;
        for (
            std::size_t i = 0;
            i < 16;
            ++i
        ) {
            const float value =
                audio[
                    frame * 16 + i
                ];
            mean += value;
            mean_square +=
                value * value;
        }
        mean /= 16.0f;
        mean_square /= 16.0f;
        AssertNear(
            mean,
            0.0f,
            2e-6f
        );
        assert(
            mean_square > 0.99f
            && mean_square <= 1.00001f
        );
    }

    assert(
        vn97::VisionPatchCount(
            4, 4, 2
        ) == 4
    );
    assert(
        vn97::VisionPatchCount(
            5, 4, 2
        ) == 0
    );

    std::vector<float> image(16);
    for (
        std::size_t i = 0;
        i < image.size();
        ++i
    ) {
        image[i] =
            static_cast<float>(i);
    }

    std::vector<float> vision(
        4 * 6
    );
    assert(
        vn97::PrepareVisionPatchesF32(
            image.data(),
            vision.data(),
            vision.size(),
            1,
            1,
            4,
            4,
            2,
            1e-5f
        ) ==
        vn97::ModalityStatus::kOk
    );

    const float coords[8] = {
        -1, -1,
        -1, 1,
        1, -1,
        1, 1
    };

    for (
        std::size_t patch = 0;
        patch < 4;
        ++patch
    ) {
        float mean = 0.0f;
        for (
            std::size_t i = 0;
            i < 4;
            ++i
        ) {
            mean +=
                vision[
                    patch * 6 + i
                ];
        }
        AssertNear(
            mean / 4.0f,
            0.0f,
            2e-6f
        );
        AssertNear(
            vision[
                patch * 6 + 4
            ],
            coords[
                patch * 2
            ]
        );
        AssertNear(
            vision[
                patch * 6 + 5
            ],
            coords[
                patch * 2 + 1
            ]
        );
    }

    std::vector<float> tiny(1);
    assert(
        vn97::PrepareAudioFramesF32(
            waveform.data(),
            tiny.data(),
            tiny.size(),
            1,
            40,
            16,
            8,
            1e-5f
        ) ==
        vn97::ModalityStatus::kOutputTooSmall
    );
    assert(
        vn97::PrepareVisionPatchesF32(
            image.data(),
            tiny.data(),
            tiny.size(),
            1,
            1,
            4,
            4,
            2,
            1e-5f
        ) ==
        vn97::ModalityStatus::kOutputTooSmall
    );
    assert(
        vn97::PrepareAudioFramesF32(
            waveform.data(),
            audio.data(),
            audio.size(),
            1,
            40,
            16,
            8,
            0.0f
        ) ==
        vn97::ModalityStatus::kInvalidParameter
    );

    return 0;
}
