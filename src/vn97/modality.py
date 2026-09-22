from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantization import TernaryLinear
from .ssm import RMSNorm


class Modality(IntEnum):
    TEXT = 3
    AUDIO = 4
    VISION = 5


@dataclass(frozen=True)
class AudioAdapterConfig:
    frame_size: int = 320
    hop_size: int = 320
    eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.frame_size <= 0 or self.hop_size <= 0:
            raise ValueError("frame_size and hop_size must be positive")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")


@dataclass(frozen=True)
class VisionAdapterConfig:
    channels: int = 3
    patch_size: int = 16
    eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.channels <= 0 or self.patch_size <= 0:
            raise ValueError("channels and patch_size must be positive")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")


def prepare_audio_frames(
    waveform: torch.Tensor,
    *,
    frame_size: int,
    hop_size: int,
    eps: float,
) -> torch.Tensor:
    """Frame mono PCM [B,S], remove DC and RMS-normalize each frame."""
    if waveform.ndim != 2:
        raise ValueError("waveform must have shape [batch, samples]")
    if frame_size <= 0 or hop_size <= 0 or eps <= 0.0:
        raise ValueError("frame_size, hop_size and eps must be positive")
    if waveform.shape[1] < frame_size:
        raise ValueError("waveform is shorter than one audio frame")
    frames = waveform.unfold(1, frame_size, hop_size)
    centered = frames - frames.mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(centered.square().mean(dim=-1, keepdim=True) + eps)
    return centered * inv_rms


def _patch_coordinates(
    rows: int,
    cols: int,
    *,
    device,
    dtype,
) -> torch.Tensor:
    y = torch.zeros(rows, device=device, dtype=dtype)
    x = torch.zeros(cols, device=device, dtype=dtype)
    if rows > 1:
        y = torch.linspace(-1.0, 1.0, rows, device=device, dtype=dtype)
    if cols > 1:
        x = torch.linspace(-1.0, 1.0, cols, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((yy.reshape(-1), xx.reshape(-1)), dim=-1)


def prepare_vision_patches(
    image: torch.Tensor,
    *,
    channels: int,
    patch_size: int,
    eps: float,
) -> torch.Tensor:
    """Create normalized non-overlap NCHW patches with explicit 2-D coordinates."""
    if image.ndim != 4:
        raise ValueError("image must have shape [batch, channels, height, width]")
    if channels <= 0 or patch_size <= 0 or eps <= 0.0:
        raise ValueError("channels, patch_size and eps must be positive")
    if image.shape[1] != channels:
        raise ValueError(f"expected {channels} image channels, got {image.shape[1]}")
    height, width = image.shape[-2:]
    if height < patch_size or width < patch_size:
        raise ValueError("image is smaller than one vision patch")
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("height and width must be divisible by patch_size")

    patches = F.unfold(
        image,
        kernel_size=patch_size,
        stride=patch_size,
    ).transpose(1, 2)
    centered = patches - patches.mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(centered.square().mean(dim=-1, keepdim=True) + eps)
    normalized = centered * inv_rms

    rows = height // patch_size
    cols = width // patch_size
    coords = _patch_coordinates(
        rows,
        cols,
        device=image.device,
        dtype=image.dtype,
    ).unsqueeze(0).expand(image.shape[0], -1, -1)
    return torch.cat((normalized, coords), dim=-1)


class AudioFrameAdapter(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        ternary_threshold: float = 0.5,
        config: AudioAdapterConfig | None = None,
        rms_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.config = config or AudioAdapterConfig()
        self.projection = TernaryLinear(
            self.config.frame_size,
            d_model,
            bias=False,
            threshold=ternary_threshold,
        )
        self.norm = RMSNorm(d_model, eps=rms_eps)

    @property
    def input_features(self) -> int:
        return self.config.frame_size

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        frames = prepare_audio_frames(
            waveform,
            frame_size=self.config.frame_size,
            hop_size=self.config.hop_size,
            eps=self.config.eps,
        )
        return self.norm(self.projection(frames))


class VisionPatchAdapter(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        ternary_threshold: float = 0.5,
        config: VisionAdapterConfig | None = None,
        rms_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.config = config or VisionAdapterConfig()
        patch_values = (
            self.config.channels
            * self.config.patch_size
            * self.config.patch_size
        )
        self.projection = TernaryLinear(
            patch_values + 2,
            d_model,
            bias=False,
            threshold=ternary_threshold,
        )
        self.norm = RMSNorm(d_model, eps=rms_eps)

    @property
    def input_features(self) -> int:
        return self.projection.in_features

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        patches = prepare_vision_patches(
            image,
            channels=self.config.channels,
            patch_size=self.config.patch_size,
            eps=self.config.eps,
        )
        return self.norm(self.projection(patches))


def prepend_modality_identity(
    language_core: nn.Module,
    embeddings: torch.Tensor,
    modality: Modality,
) -> torch.Tensor:
    if embeddings.ndim != 3:
        raise ValueError("embeddings must have shape [batch, sequence, d_model]")
    config = getattr(language_core, "config", None)
    embedding = getattr(language_core, "embedding", None)
    if config is None or embedding is None:
        raise TypeError("language_core must expose config and embedding")
    if embeddings.shape[-1] != config.d_model:
        raise ValueError(
            f"expected embedding width {config.d_model}, got {embeddings.shape[-1]}"
        )
    ids = torch.full(
        (embeddings.shape[0], 1),
        int(modality),
        dtype=torch.long,
        device=embeddings.device,
    )
    prefix = embedding(ids)
    if prefix.dtype != embeddings.dtype:
        prefix = prefix.to(dtype=embeddings.dtype)
    return torch.cat((prefix, embeddings), dim=1)
