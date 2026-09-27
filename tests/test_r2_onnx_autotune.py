from __future__ import annotations

import json
from pathlib import Path

import pytest

from vn97.r2.onnx_autotune import (
    compile_r2e4_profile,
    verify_r2e4_profile,
)
from vn97.r2.onnx_profiling import (
    R2E3Measurement,
    build_r2e3_receipt,
)


BUNDLE = "a" * 64
ARCH = "b" * 64


def _manifest() -> dict[str, object]:
    return {
        "bundle_id": BUNDLE,
        "architecture_fingerprint": ARCH,
        "profile": "deep",
        "graphs": [
            {
                "filename": "step.onnx",
                "kind": "step",
                "sequence_length": 1,
            },
            {
                "filename": "chunk-8.onnx",
                "kind": "chunk",
                "sequence_length": 8,
            },
            {
                "filename": "chunk-32.onnx",
                "kind": "chunk",
                "sequence_length": 32,
            },
        ],
    }


def _m(
    filename: str,
    sequence_length: int,
    provider: str,
    *,
    p50_ns: int,
    p95_ns: int,
) -> R2E3Measurement:
    return R2E3Measurement(
        graph_filename=filename,
        kind="step" if sequence_length == 1 else "chunk",
        sequence_length=sequence_length,
        requested_provider=provider,
        actual_provider=provider,
        warmup_iterations=3,
        steady_iterations=5,
        session_create_ns=20_000_000,
        steady_latencies_ns=(
            p50_ns - 2,
            p50_ns - 1,
            p50_ns,
            p95_ns,
            p95_ns,
        ),
        memory_before_bytes=2_000_000_000,
        memory_after_bytes=1_900_000_000,
        thermal_before=1,
        thermal_after=2,
        fallback_used=False,
    )


def _receipt(
    *,
    include_cpu: bool = True,
) -> dict[str, object]:
    measurements = [
        _m(
            "step.onnx",
            1,
            "XNNPACK",
            p50_ns=1_000_000,
            p95_ns=1_200_000,
        ),
        _m(
            "chunk-8.onnx",
            8,
            "NNAPI",
            p50_ns=3_000_000,
            p95_ns=3_500_000,
        ),
        _m(
            "chunk-32.onnx",
            32,
            "NNAPI",
            p50_ns=8_000_000,
            p95_ns=9_000_000,
        ),
    ]
    if include_cpu:
        measurements.extend(
            [
                _m(
                    "step.onnx",
                    1,
                    "CPU",
                    p50_ns=1_500_000,
                    p95_ns=1_700_000,
                ),
                _m(
                    "chunk-8.onnx",
                    8,
                    "CPU",
                    p50_ns=4_000_000,
                    p95_ns=4_500_000,
                ),
                _m(
                    "chunk-32.onnx",
                    32,
                    "CPU",
                    p50_ns=12_000_000,
                    p95_ns=13_000_000,
                ),
            ]
        )
    return build_r2e3_receipt(
        bundle_id=BUNDLE,
        architecture_fingerprint=ARCH,
        profile="deep",
        device={
            "sdk_int": 36,
            "logical_cores": 8,
            "available_memory_bytes": 2_000_000_000,
            "thermal_status": 1,
            "hardware": "exynos2100",
            "soc_manufacturer": "samsung",
            "soc_model": "Exynos 2100",
        },
        measurements=tuple(measurements),
    )


def test_compile_profile_uses_measured_graph_specific_providers() -> None:
    profile = compile_r2e4_profile(
        _manifest(),
        _receipt(),
    )
    policies = {
        item["graph_filename"]: item
        for item in profile["graph_policies"]
    }
    assert policies["step.onnx"]["preferred_provider"] == "XNNPACK"
    assert policies["step.onnx"]["provider_order"] == [
        "XNNPACK",
        "CPU",
    ]
    assert policies["chunk-32.onnx"]["preferred_provider"] == "NNAPI"
    assert policies["chunk-32.onnx"]["provider_order"] == [
        "NNAPI",
        "CPU",
    ]
    # chunk-32: 32 / 8ms = 4000 tok/s, chunk-8: 8 / 3ms ~= 2666.
    assert profile["preferred_chunk_sizes"] == [32, 8]
    assert profile["same_weights_semantics"] is True
    assert profile["quantization_used"] is False


def test_compile_profile_honors_measured_cpu_win() -> None:
    receipt = _receipt()
    step = next(
        item
        for item in receipt["graph_results"]
        if item["graph_filename"] == "step.onnx"
    )
    for item in step["ranked_providers"]:
        if item["requested_provider"] == "CPU":
            item["latency_p50_ns"] = 500_000
            item["latency_p95_ns"] = 600_000
            item["tokens_per_second_milli_p50"] = 2_000_000
    # Rebuild the receipt identity after the deliberate test mutation.
    measurements = (
        _m(
            "step.onnx",
            1,
            "CPU",
            p50_ns=500_000,
            p95_ns=600_000,
        ),
        _m(
            "step.onnx",
            1,
            "XNNPACK",
            p50_ns=1_000_000,
            p95_ns=1_200_000,
        ),
        _m(
            "chunk-8.onnx",
            8,
            "NNAPI",
            p50_ns=3_000_000,
            p95_ns=3_500_000,
        ),
        _m(
            "chunk-8.onnx",
            8,
            "CPU",
            p50_ns=4_000_000,
            p95_ns=4_500_000,
        ),
        _m(
            "chunk-32.onnx",
            32,
            "NNAPI",
            p50_ns=8_000_000,
            p95_ns=9_000_000,
        ),
        _m(
            "chunk-32.onnx",
            32,
            "CPU",
            p50_ns=12_000_000,
            p95_ns=13_000_000,
        ),
    )
    receipt = build_r2e3_receipt(
        bundle_id=BUNDLE,
        architecture_fingerprint=ARCH,
        profile="deep",
        device=_receipt()["device"],
        measurements=measurements,
    )
    profile = compile_r2e4_profile(_manifest(), receipt)
    step_policy = next(
        item
        for item in profile["graph_policies"]
        if item["graph_filename"] == "step.onnx"
    )
    assert step_policy["preferred_provider"] == "CPU"
    assert step_policy["provider_order"] == ["CPU"]


def test_compile_requires_cpu_fallback_measurement() -> None:
    with pytest.raises(
        ValueError,
        match="successful CPU fallback measurement",
    ):
        compile_r2e4_profile(
            _manifest(),
            _receipt(include_cpu=False),
        )


def test_compile_requires_exact_e2_graph_coverage() -> None:
    receipt = _receipt()
    receipt["graph_results"] = [
        item
        for item in receipt["graph_results"]
        if item["graph_filename"] != "chunk-8.onnx"
    ]
    # The compiler checks semantics after E3 verification has happened.
    with pytest.raises(ValueError, match="exact E2 graph coverage"):
        compile_r2e4_profile(_manifest(), receipt)


def test_profile_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    profile = compile_r2e4_profile(
        _manifest(),
        _receipt(),
    )
    path = tmp_path / "tuning.json"
    path.write_text(
        json.dumps(
            profile,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    verified = verify_r2e4_profile(
        path,
        expected_bundle_id=BUNDLE,
    )
    assert verified == profile

    profile["preferred_chunk_sizes"] = [8, 32]
    path.write_text(
        json.dumps(profile, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_r2e4_profile(path)


def test_profile_control_policy_has_hysteresis() -> None:
    profile = compile_r2e4_profile(
        _manifest(),
        _receipt(),
    )
    control = profile["control_policy"]
    assert control["thermal_hot_exit"] < control["thermal_hot_enter"]
    assert (
        control["thermal_hot_enter"]
        < control["thermal_critical_enter"]
    )
    assert (
        control["memory_pressure_enter_ppm"]
        < control["memory_pressure_exit_ppm"]
    )
    assert (
        control["latency_slow_exit_ppm"]
        < control["latency_slow_enter_ppm"]
    )
    assert control["quarantine_failure_threshold"] == 2
    assert control["quarantine_cooldown_decisions"] == 16
