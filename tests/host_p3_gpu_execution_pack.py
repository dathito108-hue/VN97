from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "vn97"

pkg = types.ModuleType("vn97")
pkg.__path__ = [str(SRC)]
sys.modules["vn97"] = pkg

spec = importlib.util.spec_from_file_location(
    "vn97.p3_gpu_preflight",
    SRC / "p3_gpu_preflight.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load p3_gpu_preflight.py")
module = importlib.util.module_from_spec(spec)
sys.modules["vn97.p3_gpu_preflight"] = module
spec.loader.exec_module(module)


def expect_failure(label: str, fn) -> None:
    try:
        fn()
    except Exception:
        return
    raise AssertionError(f"expected failure: {label}")


def main() -> None:
    receipt = module.VN97P3GPUEnvironmentReceipt(
        repository_commit="a" * 40,
        corpus_manifest_id="1" * 64,
        corpus_manifest_sha256="2" * 64,
        p3_profile_sha256="3" * 64,
        python_version="3.12.14",
        torch_version="2.8.0+cu128",
        cuda_runtime_version="12.8",
        selected_device="cuda:0",
        gpu_name="Example NVIDIA GPU",
        compute_capability_major=8,
        compute_capability_minor=9,
        total_vram_bytes=24 * 1024**3,
        free_vram_bytes=22 * 1024**3,
        cuda_device_count=1,
        cudnn_version=90501,
        bf16_supported=True,
    )
    payload = json.loads(
        receipt.to_bytes().decode("utf-8")
    )
    assert payload["schema"] == "VN97GPUENV1"
    assert payload["selected_device"] == "cuda:0"
    assert payload["gpu_name"] == "Example NVIDIA GPU"
    assert len(payload["environment_id"]) == 64

    module.require_memory_floor(
        total_vram_bytes=24 * 1024**3,
        free_vram_bytes=22 * 1024**3,
        min_total_vram_bytes=16 * 1024**3,
        min_free_vram_bytes=8 * 1024**3,
    )
    expect_failure(
        "total VRAM floor",
        lambda: module.require_memory_floor(
            total_vram_bytes=8 * 1024**3,
            free_vram_bytes=7 * 1024**3,
            min_total_vram_bytes=16 * 1024**3,
            min_free_vram_bytes=0,
        ),
    )
    expect_failure(
        "free VRAM floor",
        lambda: module.require_memory_floor(
            total_vram_bytes=24 * 1024**3,
            free_vram_bytes=2 * 1024**3,
            min_total_vram_bytes=0,
            min_free_vram_bytes=8 * 1024**3,
        ),
    )
    expect_failure(
        "CPU device receipt",
        lambda: module.VN97P3GPUEnvironmentReceipt(
            repository_commit="a" * 40,
            corpus_manifest_id="1" * 64,
            corpus_manifest_sha256="2" * 64,
            p3_profile_sha256="3" * 64,
            python_version="3.12.14",
            torch_version="2.8.0",
            cuda_runtime_version="12.8",
            selected_device="cpu",
            gpu_name="CPU",
            compute_capability_major=0,
            compute_capability_minor=0,
            total_vram_bytes=1,
            free_vram_bytes=1,
            cuda_device_count=1,
            cudnn_version=None,
            bf16_supported=False,
        ),
    )

    launcher = (
        ROOT / "tools" / "run_p3_gpu.sh"
    ).read_text(encoding="utf-8")
    for required in (
        "VN97_EXPECTED_REPOSITORY_COMMIT",
        "vn97-p3-gpu-preflight",
        "vn97-p3-language-campaign",
        "gpu-environment.vn97gpuenv1.json",
        "p3-run.vn97p3run1.json",
        "SHA256SUMS",
    ):
        assert required in launcher

    print(
        "VN97 P3 GPU execution pack host contract PASS "
        f"environment={payload['environment_id']}"
    )


if __name__ == "__main__":
    main()
