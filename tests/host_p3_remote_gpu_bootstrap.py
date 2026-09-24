from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    script = (
        ROOT / "tools" / "bootstrap_p3_remote_gpu.sh"
    ).read_text(encoding="utf-8")

    required = (
        "b68428ea9e16f8fd0b42bdd6a76b633456695201",
        "35992987070",
        "VN97-P3-Corpus-1623f2c187d043918d681a7dc9e9cf025a58d26f",
        "77e9142054f82b78c8ca0ff108c5e5297d43292841f9fe1ea69c5dc97a7c303f",
        "gh auth status",
        "gh run download",
        "sha256sum -c SHA256SUMS",
        "VN97_EXPECTED_REPOSITORY_COMMIT",
        "tools/run_p3_gpu.sh",
        "p3-run.vn97p3run1.json",
    )
    for value in required:
        assert value in script, value

    assert 'git -C "$REPO_DIR" checkout --detach' in script
    assert 'git -C "$REPO_DIR" diff --quiet' in script
    assert "training commit mismatch after checkout" in script
    assert "VN97CORPUS1 identity mismatch" in script
    assert "output directory must be new or empty" in script

    docs = (
        ROOT / "docs" / "P3_REMOTE_GPU_BOOTSTRAP.md"
    ).read_text(encoding="utf-8")
    assert "2026-10-01T11:26:15Z" in docs
    assert "Do not paste access tokens into ChatGPT." in docs
    assert "VN97GPUENV1" in docs

    print("VN97 P3 remote GPU bootstrap host contract PASS")


if __name__ == "__main__":
    main()
