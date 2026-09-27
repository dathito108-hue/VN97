from __future__ import annotations

import json
from pathlib import Path

import pytest

from vn97.r2.onnx_profiling import (
    R2E3Measurement,
    build_r2e3_receipt,
    percentile_nearest_rank,
    rank_measurements,
    verify_r2e3_receipt,
)


BUNDLE = "a" * 64
ARCH = "b" * 64


def _measurement(
    provider: str,
    *,
    actual: str | None = None,
    p50_base: float = 10.0,
    fallback: bool = False,
    sequence_length: int = 32,
    filename: str = "chunk-32.onnx",
) -> R2E3Measurement:
    return R2E3Measurement(
        graph_filename=filename,
        kind="step" if sequence_length == 1 else "chunk",
        sequence_length=sequence_length,
        requested_provider=provider,
        actual_provider=actual or provider,
        warmup_iterations=3,
        steady_iterations=5,
        session_create_ns=25_000_000,
        steady_latencies_ns=tuple(
            int(value * 1_000_000)
            for value in (
                p50_base - 1.0,
                p50_base - 0.5,
                p50_base,
                p50_base + 0.5,
                p50_base + 1.0,
            )
        ),
        memory_before_bytes=2_000_000_000,
        memory_after_bytes=1_900_000_000,
        thermal_before=1,
        thermal_after=2,
        fallback_used=fallback,
    )


def _device() -> dict[str, object]:
    return {
        "sdk_int": 36,
        "logical_cores": 8,
        "available_memory_bytes": 2_000_000_000,
        "thermal_status": 1,
        "hardware": "exynos2100",
        "soc_manufacturer": "samsung",
        "soc_model": "Exynos 2100",
    }


def test_percentile_nearest_rank_is_deterministic() -> None:
    values = (1, 2, 3, 4, 5)
    assert percentile_nearest_rank(values, 0.5) == 3
    assert percentile_nearest_rank(values, 0.95) == 5
    assert percentile_nearest_rank(values, 1.0) == 5


def test_rank_prefers_real_provider_over_faster_fallback() -> None:
    nnapi_fallback = _measurement(
        "NNAPI",
        actual="CPU",
        p50_base=4.0,
        fallback=True,
    )
    xnnpack = _measurement(
        "XNNPACK",
        p50_base=6.0,
    )
    cpu = _measurement(
        "CPU",
        p50_base=8.0,
    )
    ranked = rank_measurements(
        (nnapi_fallback, cpu, xnnpack)
    )
    assert ranked[0]["requested_provider"] == "XNNPACK"
    assert ranked[0]["actual_provider"] == "XNNPACK"
    assert ranked[-1]["fallback_used"] is True


def test_receipt_groups_graphs_and_recommends_per_graph() -> None:
    receipt = build_r2e3_receipt(
        bundle_id=BUNDLE,
        architecture_fingerprint=ARCH,
        profile="deep",
        device=_device(),
        measurements=(
            _measurement("NNAPI", p50_base=5.0),
            _measurement("XNNPACK", p50_base=7.0),
            _measurement(
                "NNAPI",
                p50_base=1.5,
                sequence_length=1,
                filename="step.onnx",
            ),
            _measurement(
                "XNNPACK",
                p50_base=1.1,
                sequence_length=1,
                filename="step.onnx",
            ),
        ),
        failures=(
            {
                "graph_filename": "chunk-32.onnx",
                "requested_provider": "QNN",
                "error_class": "provider_not_available",
            },
        ),
    )
    assert receipt["schema"] == "VN97R2E3PROFILE1"
    assert receipt["device_measured"] is True
    assert receipt["synthetic"] is False
    results = {
        item["graph_filename"]: item
        for item in receipt["graph_results"]
    }
    assert (
        results["chunk-32.onnx"]["recommended_provider"]
        == "NNAPI"
    )
    assert results["step.onnx"]["recommended_provider"] == "XNNPACK"


def test_receipt_roundtrip_and_bundle_binding(tmp_path: Path) -> None:
    receipt = build_r2e3_receipt(
        bundle_id=BUNDLE,
        architecture_fingerprint=ARCH,
        profile="fast",
        device=_device(),
        measurements=(
            _measurement("NNAPI", p50_base=5.0),
            _measurement("CPU", p50_base=9.0),
        ),
    )
    path = tmp_path / "receipt.json"
    path.write_text(
        json.dumps(
            receipt,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    verified = verify_r2e3_receipt(
        path,
        expected_bundle_id=BUNDLE,
    )
    assert verified == receipt

    with pytest.raises(ValueError, match="another ONNX bundle"):
        verify_r2e3_receipt(
            path,
            expected_bundle_id="c" * 64,
        )


def test_receipt_tamper_is_rejected(tmp_path: Path) -> None:
    receipt = build_r2e3_receipt(
        bundle_id=BUNDLE,
        architecture_fingerprint=ARCH,
        profile="deep",
        device=_device(),
        measurements=(
            _measurement("NNAPI"),
        ),
    )
    receipt["graph_results"][0]["recommended_provider"] = "CPU"
    path = tmp_path / "receipt.json"
    path.write_text(
        json.dumps(receipt, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_r2e3_receipt(path)


def test_provider_substitution_requires_fallback_flag() -> None:
    with pytest.raises(ValueError, match="fallback_used"):
        _measurement(
            "NNAPI",
            actual="CPU",
            fallback=False,
        )


def test_measurement_throughput_uses_sequence_length() -> None:
    measurement = _measurement(
        "NNAPI",
        p50_base=8.0,
        sequence_length=32,
    )
    summary = measurement.summary()
    assert summary["latency_p50_ns"] == 8_000_000
    assert summary["tokens_per_second_milli_p50"] == 4_000_000
