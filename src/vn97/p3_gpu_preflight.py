from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


class VN97P3GPUEnvironmentError(RuntimeError):
    pass


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_sha256(value: str, *, label: str) -> str:
    if (
        len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise VN97P3GPUEnvironmentError(
            f"{label} must be lowercase SHA-256"
        )
    return value


def _require_commit(value: str) -> str:
    if (
        len(value) != 40
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise VN97P3GPUEnvironmentError(
            "repository commit must be 40 lowercase hex"
        )
    return value


def _bounded_text(
    value: str,
    *,
    label: str,
    max_bytes: int = 256,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > max_bytes
        or any(ord(ch) < 0x20 for ch in value)
    ):
        raise VN97P3GPUEnvironmentError(
            f"{label} is invalid"
        )
    return value


@dataclass(frozen=True)
class VN97P3GPUEnvironmentReceipt:
    repository_commit: str
    corpus_manifest_id: str
    corpus_manifest_sha256: str
    p3_profile_sha256: str
    python_version: str
    torch_version: str
    cuda_runtime_version: str
    selected_device: str
    gpu_name: str
    compute_capability_major: int
    compute_capability_minor: int
    total_vram_bytes: int
    free_vram_bytes: int
    cuda_device_count: int
    cudnn_version: int | None
    bf16_supported: bool

    def __post_init__(self) -> None:
        _require_commit(self.repository_commit)
        for value, label in (
            (self.corpus_manifest_id, "corpus manifest ID"),
            (
                self.corpus_manifest_sha256,
                "corpus manifest SHA-256",
            ),
            (self.p3_profile_sha256, "P3 profile SHA-256"),
        ):
            _require_sha256(value, label=label)

        for value, label in (
            (self.python_version, "Python version"),
            (self.torch_version, "Torch version"),
            (self.cuda_runtime_version, "CUDA runtime version"),
            (self.selected_device, "selected device"),
            (self.gpu_name, "GPU name"),
        ):
            _bounded_text(value, label=label)

        if not self.selected_device.startswith("cuda"):
            raise VN97P3GPUEnvironmentError(
                "selected device must be CUDA"
            )
        if (
            type(self.compute_capability_major) is not int
            or type(self.compute_capability_minor) is not int
            or self.compute_capability_major < 0
            or self.compute_capability_minor < 0
        ):
            raise VN97P3GPUEnvironmentError(
                "compute capability is invalid"
            )
        if (
            type(self.total_vram_bytes) is not int
            or type(self.free_vram_bytes) is not int
            or self.total_vram_bytes <= 0
            or self.free_vram_bytes <= 0
            or self.free_vram_bytes > self.total_vram_bytes
        ):
            raise VN97P3GPUEnvironmentError(
                "VRAM values are invalid"
            )
        if (
            type(self.cuda_device_count) is not int
            or self.cuda_device_count <= 0
        ):
            raise VN97P3GPUEnvironmentError(
                "CUDA device count must be positive"
            )
        if (
            self.cudnn_version is not None
            and (
                type(self.cudnn_version) is not int
                or self.cudnn_version <= 0
            )
        ):
            raise VN97P3GPUEnvironmentError(
                "cuDNN version is invalid"
            )
        if type(self.bf16_supported) is not bool:
            raise VN97P3GPUEnvironmentError(
                "bf16_supported must be boolean"
            )

    def canonical_object(self) -> dict[str, object]:
        body: dict[str, object] = {
            "bf16_supported": self.bf16_supported,
            "compute_capability_major":
                self.compute_capability_major,
            "compute_capability_minor":
                self.compute_capability_minor,
            "corpus_manifest_id": self.corpus_manifest_id,
            "corpus_manifest_sha256":
                self.corpus_manifest_sha256,
            "cuda_device_count": self.cuda_device_count,
            "cuda_runtime_version": self.cuda_runtime_version,
            "cudnn_version": self.cudnn_version,
            "free_vram_bytes": self.free_vram_bytes,
            "gpu_name": self.gpu_name,
            "p3_profile_sha256": self.p3_profile_sha256,
            "python_version": self.python_version,
            "repository_commit": self.repository_commit,
            "schema": "VN97GPUENV1",
            "selected_device": self.selected_device,
            "torch_version": self.torch_version,
            "total_vram_bytes": self.total_vram_bytes,
        }
        body["environment_id"] = hashlib.sha256(
            b"VN97GPUENV1\0" + _canonical_json(body)
        ).hexdigest()
        return body

    def to_bytes(self) -> bytes:
        return _canonical_json(
            self.canonical_object()
        ) + b"\n"


def require_memory_floor(
    *,
    total_vram_bytes: int,
    free_vram_bytes: int,
    min_total_vram_bytes: int,
    min_free_vram_bytes: int,
) -> None:
    for value, label in (
        (min_total_vram_bytes, "minimum total VRAM"),
        (min_free_vram_bytes, "minimum free VRAM"),
    ):
        if type(value) is not int or value < 0:
            raise VN97P3GPUEnvironmentError(
                f"{label} must be a non-negative integer"
            )
    if total_vram_bytes < min_total_vram_bytes:
        raise VN97P3GPUEnvironmentError(
            "GPU total VRAM is below requested floor"
        )
    if free_vram_bytes < min_free_vram_bytes:
        raise VN97P3GPUEnvironmentError(
            "GPU free VRAM is below requested floor"
        )
