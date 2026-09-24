from __future__ import annotations

import argparse
from pathlib import Path
import platform
import sys

import torch

from .p3_gpu_preflight import (
    VN97P3GPUEnvironmentError,
    VN97P3GPUEnvironmentReceipt,
    require_memory_floor,
)
from .p3_language_campaign import profile_sha256
from .p3_language_campaign_cli import _validate_corpus
from .training_cli import _atomic_write


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a CUDA host and sealed P3 corpus before the "
            "VN97 production-language campaign."
        )
    )
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--repository-commit", required=True)
    parser.add_argument(
        "--device",
        default="cuda:0",
    )
    parser.add_argument(
        "--min-total-vram-bytes",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--min-free-vram-bytes",
        type=int,
        default=0,
    )
    parser.add_argument("--output", required=True)
    return parser


def _device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise VN97P3GPUEnvironmentError(
            "P3 GPU preflight requires a CUDA device"
        )
    if device.index is not None:
        return int(device.index)
    current = int(torch.cuda.current_device())
    if current < 0:
        raise VN97P3GPUEnvironmentError(
            "CUDA current device index is invalid"
        )
    return current


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise VN97P3GPUEnvironmentError(
            "CUDA is not available on this host"
        )

    device = torch.device(args.device)
    index = _device_index(device)
    if not 0 <= index < torch.cuda.device_count():
        raise VN97P3GPUEnvironmentError(
            "selected CUDA device index is unavailable"
        )
    torch.cuda.set_device(index)

    (
        corpus_manifest_id,
        corpus_manifest_sha256,
    ) = _validate_corpus(Path(args.corpus_dir))

    properties = torch.cuda.get_device_properties(index)
    free_vram, total_vram = torch.cuda.mem_get_info(index)
    require_memory_floor(
        total_vram_bytes=int(total_vram),
        free_vram_bytes=int(free_vram),
        min_total_vram_bytes=args.min_total_vram_bytes,
        min_free_vram_bytes=args.min_free_vram_bytes,
    )

    cuda_runtime = torch.version.cuda
    if not isinstance(cuda_runtime, str) or not cuda_runtime:
        raise VN97P3GPUEnvironmentError(
            "PyTorch does not expose a CUDA runtime version"
        )

    cudnn = torch.backends.cudnn.version()
    receipt = VN97P3GPUEnvironmentReceipt(
        repository_commit=args.repository_commit,
        corpus_manifest_id=corpus_manifest_id,
        corpus_manifest_sha256=corpus_manifest_sha256,
        p3_profile_sha256=profile_sha256(),
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        cuda_runtime_version=cuda_runtime,
        selected_device=f"cuda:{index}",
        gpu_name=str(properties.name),
        compute_capability_major=int(properties.major),
        compute_capability_minor=int(properties.minor),
        total_vram_bytes=int(total_vram),
        free_vram_bytes=int(free_vram),
        cuda_device_count=int(torch.cuda.device_count()),
        cudnn_version=(
            None if cudnn is None else int(cudnn)
        ),
        bf16_supported=bool(
            torch.cuda.is_bf16_supported()
        ),
    )
    _atomic_write(
        Path(args.output),
        receipt.to_bytes(),
    )
    payload = receipt.canonical_object()
    print(
        "VN97GPUENV1 "
        f"id={payload['environment_id']} "
        f"device={receipt.selected_device} "
        f"gpu={receipt.gpu_name} "
        f"free_vram={receipt.free_vram_bytes}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            f"vn97-p3-gpu-preflight: {exc}",
            file=sys.stderr,
        )
        raise
