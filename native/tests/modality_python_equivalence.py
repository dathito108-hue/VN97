from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import torch

from vn97.modality import (
    prepare_audio_frames,
    prepare_vision_patches,
)


FLOAT_P = ctypes.POINTER(
    ctypes.c_float
)


def ptr(
    tensor: torch.Tensor,
) -> FLOAT_P:
    assert (
        tensor.dtype
        == torch.float32
    )
    assert (
        tensor.device.type
        == "cpu"
    )
    assert tensor.is_contiguous()
    return ctypes.cast(
        tensor.data_ptr(),
        FLOAT_P,
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: modality_python_equivalence.py <shared-library>"
        )

    lib = ctypes.CDLL(
        str(
            Path(
                sys.argv[1]
            ).resolve()
        )
    )

    lib.vn97_audio_frame_count.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.vn97_audio_frame_count.restype = (
        ctypes.c_size_t
    )
    lib.vn97_prepare_audio_frames_f32.argtypes = [
        FLOAT_P,
        FLOAT_P,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_float,
    ]
    lib.vn97_prepare_audio_frames_f32.restype = (
        ctypes.c_int
    )

    lib.vn97_vision_patch_count.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.vn97_vision_patch_count.restype = (
        ctypes.c_size_t
    )
    lib.vn97_prepare_vision_patches_f32.argtypes = [
        FLOAT_P,
        FLOAT_P,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_float,
    ]
    lib.vn97_prepare_vision_patches_f32.restype = (
        ctypes.c_int
    )

    torch.manual_seed(1973)

    waveform = torch.randn(
        2,
        73,
        dtype=torch.float32,
    ).contiguous()
    expected_audio = (
        prepare_audio_frames(
            waveform,
            frame_size=16,
            hop_size=7,
            eps=1e-5,
        ).contiguous()
    )

    assert (
        lib.vn97_audio_frame_count(
            73, 16, 7
        )
        == expected_audio.shape[1]
    )

    actual_audio = torch.empty_like(
        expected_audio
    )

    status = (
        lib.vn97_prepare_audio_frames_f32(
            ptr(waveform),
            ptr(actual_audio),
            actual_audio.numel(),
            waveform.shape[0],
            waveform.shape[1],
            16,
            7,
            1e-5,
        )
    )
    assert status == 0

    torch.testing.assert_close(
        actual_audio,
        expected_audio,
        rtol=3e-5,
        atol=3e-6,
    )

    image = torch.randn(
        2,
        3,
        6,
        8,
        dtype=torch.float32,
    ).contiguous()

    expected_vision = (
        prepare_vision_patches(
            image,
            channels=3,
            patch_size=2,
            eps=1e-5,
        ).contiguous()
    )

    assert (
        lib.vn97_vision_patch_count(
            6, 8, 2
        )
        == expected_vision.shape[1]
    )

    actual_vision = torch.empty_like(
        expected_vision
    )

    status = (
        lib.vn97_prepare_vision_patches_f32(
            ptr(image),
            ptr(actual_vision),
            actual_vision.numel(),
            image.shape[0],
            image.shape[1],
            image.shape[2],
            image.shape[3],
            2,
            1e-5,
        )
    )
    assert status == 0

    torch.testing.assert_close(
        actual_vision,
        expected_vision,
        rtol=3e-5,
        atol=3e-6,
    )

    print(
        "M3B native modality preprocessing equivalence PASS"
    )


if __name__ == "__main__":
    main()
