from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "vn97" / "release_candidate.py"
SPEC = importlib.util.spec_from_file_location(
    "vn97_release_candidate_standalone",
    MODULE_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load release_candidate.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

Device = MODULE.VN97ReleaseCandidateDeviceEvidence
Manifest = MODULE.VN97ReleaseCandidateManifest
Error = MODULE.VN97ReleaseCandidateError
parse = MODULE.parse_release_candidate_manifest
MAX = MODULE.MAX_MANIFEST_BYTES


def expect_failure(label: str, fn) -> None:
    try:
        fn()
    except Exception:
        return
    raise AssertionError(f"expected failure: {label}")


def evidence(
    digest: str = "22" * 32,
) -> object:
    return Device(
        evidence_sha256=digest,
        manufacturer="fixture",
        model="phone",
        sdk_int=37,
        abi="arm64-v8a",
        runs=5,
        text_prefill_p95_ms=12.0,
        text_decode_p95_ms_per_token=6.0,
        speech_prefill_p95_ms=14.0,
        peak_pss_kib=120000,
        thermal_status_max=2,
        battery_energy_counter_delta_nwh=100,
    )


def manifest(
    items: tuple[object, ...] | None = None,
) -> object:
    return Manifest(
        selected_candidate_id="0123456789abcdef",
        checkpoint_sha256="11" * 32,
        checkpoint_bytes=4096,
        tokenizer_sha256="33" * 32,
        tokenizer_bytes=2048,
        production_campaign_report_sha256="44" * 32,
        model_image_sha256="55" * 32,
        tile_rows=16,
        tile_cols=16,
        speech_enabled=True,
        vision_enabled=False,
        speech_training_report_sha256="66" * 32,
        vision_training_report_sha256=None,
        device_evidence=(
            evidence(),
        )
        if items is None
        else items,
    )


def main() -> None:
    value = manifest()
    encoded = value.to_bytes()
    parsed = parse(encoded)
    assert parsed == value
    assert parsed.to_bytes() == encoded

    expect_failure(
        "duplicate device evidence",
        lambda: manifest(
            (
                evidence(),
                evidence(),
            )
        ),
    )
    expect_failure(
        "missing mobile evidence",
        lambda: manifest(tuple()),
    )
    expect_failure(
        "speech report missing",
        lambda: Manifest(
            selected_candidate_id="0123456789abcdef",
            checkpoint_sha256="11" * 32,
            checkpoint_bytes=4096,
            tokenizer_sha256="33" * 32,
            tokenizer_bytes=2048,
            production_campaign_report_sha256="44" * 32,
            model_image_sha256="55" * 32,
            tile_rows=16,
            tile_cols=16,
            speech_enabled=True,
            vision_enabled=False,
            speech_training_report_sha256=None,
            vision_training_report_sha256=None,
            device_evidence=(evidence(),),
        ),
    )
    expect_failure(
        "below min sdk",
        lambda: Device(
            evidence_sha256="77" * 32,
            manufacturer="fixture",
            model="old-phone",
            sdk_int=25,
            abi="arm64-v8a",
            runs=5,
            text_prefill_p95_ms=1.0,
            text_decode_p95_ms_per_token=1.0,
            speech_prefill_p95_ms=None,
            peak_pss_kib=1,
            thermal_status_max=0,
            battery_energy_counter_delta_nwh=None,
        ),
    )
    expect_failure(
        "trailing data",
        lambda: parse(encoded + b"x"),
    )
    expect_failure(
        "oversized manifest",
        lambda: parse(b"x" * (MAX + 1)),
    )

    print("M19B VN97RC1 host contracts: PASS")


if __name__ == "__main__":
    main()
