from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vn97.r2.onnx_hardening import (
    R2E5_HARDENING_SCHEMA,
    R2E5_RUN_SCHEMA,
    compile_r2e5_hardening,
    is_s21_fe_device,
    verify_r2e5_hardening,
    verify_r2e5_run,
)


BUNDLE = "a" * 64
ARCH = "b" * 64
E3 = "c" * 64
E4 = "d" * 64


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _device(
    model: str = "SM-G990B",
) -> dict[str, object]:
    return {
        "manufacturer": "samsung",
        "model": model,
        "sdk_int": 36,
        "logical_cores": 8,
        "hardware": "exynos2100",
        "soc_manufacturer": "samsung",
        "soc_model": "Exynos 2100",
    }


def _phase(
    name: str,
    kind: str,
    *,
    iterations: int,
    tokens_per_iteration: int,
    latency_ns: int,
    thermal_before: int = 1,
    thermal_after: int = 2,
    failure_count: int = 0,
) -> dict[str, object]:
    return {
        "name": name,
        "kind": kind,
        "iterations": iterations,
        "tokens_per_iteration": tokens_per_iteration,
        "latencies_ns": [latency_ns] * iterations,
        "provider_counts": {"CPU": iterations},
        "graph_counts": {
            "step.onnx" if tokens_per_iteration == 1
            else "chunk-32.onnx": iterations
        },
        "thermal_before": thermal_before,
        "thermal_after": thermal_after,
        "memory_before_bytes": 2_000_000_000,
        "memory_after_bytes": 1_800_000_000,
        "failure_count": failure_count,
    }


def _run(
    *,
    model: str = "SM-G990B",
    controls_pass: bool = True,
    sustained_latency_ns: int = 320_000_000,
    failure_count: int = 0,
    recovery_thermal_after: int = 2,
) -> dict[str, object]:
    phases = [
        _phase(
            "cold_start",
            "cold",
            iterations=1,
            tokens_per_iteration=1,
            latency_ns=20_000_000,
        ),
        _phase(
            "warm_step",
            "warm_step",
            iterations=8,
            tokens_per_iteration=1,
            latency_ns=10_000_000,
        ),
        _phase(
            "prefill",
            "prefill",
            iterations=4,
            tokens_per_iteration=64,
            latency_ns=350_000_000,
        ),
        _phase(
            "sustained",
            "sustained",
            iterations=8,
            tokens_per_iteration=64,
            latency_ns=sustained_latency_ns,
            thermal_before=2,
            thermal_after=4,
            failure_count=failure_count,
        ),
        _phase(
            "recovery",
            "recovery",
            iterations=4,
            tokens_per_iteration=1,
            latency_ns=12_000_000,
            thermal_before=4,
            thermal_after=recovery_thermal_after,
        ),
    ]
    controls = {
        "thermal_hysteresis_passed": controls_pass,
        "memory_pressure_passed": controls_pass,
        "provider_quarantine_passed": controls_pass,
        "provider_recovery_passed": controls_pass,
        "cpu_fallback_passed": controls_pass,
    }
    body = {
        "schema": R2E5_RUN_SCHEMA,
        "bundle_id": BUNDLE,
        "e3_receipt_id": E3,
        "tuning_id": E4,
        "device": _device(model),
        "phases": phases,
        "control_tests": controls,
        "device_measured": True,
        "synthetic": False,
    }
    run = dict(body)
    run["run_id"] = hashlib.sha256(
        b"VN97R2E5RUN1\0" + _canonical(body)
    ).hexdigest()
    return run


def _manifest() -> dict[str, object]:
    return {
        "bundle_id": BUNDLE,
        "architecture_fingerprint": ARCH,
    }


def _e3() -> dict[str, object]:
    return {
        "bundle_id": BUNDLE,
        "receipt_id": E3,
    }


def _e4() -> dict[str, object]:
    return {
        "bundle_id": BUNDLE,
        "e3_receipt_id": E3,
        "tuning_id": E4,
        "preferred_chunk_sizes": [32, 8, 16],
        "graph_policies": [
            {"graph_filename": "step.onnx"},
            {"graph_filename": "chunk-8.onnx"},
            {"graph_filename": "chunk-16.onnx"},
            {"graph_filename": "chunk-32.onnx"},
        ],
    }


def test_s21_fe_identity_accepts_g990_family() -> None:
    assert is_s21_fe_device(_device("SM-G990B"))
    assert is_s21_fe_device(_device("SM-G990U1"))
    assert not is_s21_fe_device(_device("SM-G991B"))
    assert not is_s21_fe_device(
        {
            **_device("SM-G990B"),
            "manufacturer": "other",
        }
    )


def test_device_run_roundtrip_and_bindings(tmp_path: Path) -> None:
    run = _run()
    path = tmp_path / "run.json"
    path.write_bytes(_canonical(run) + b"\n")
    verified = verify_r2e5_run(
        path,
        expected_bundle_id=BUNDLE,
        expected_e3_receipt_id=E3,
        expected_tuning_id=E4,
    )
    assert verified == run

    with pytest.raises(ValueError, match="another E2 bundle"):
        verify_r2e5_run(
            path,
            expected_bundle_id="e" * 64,
        )


def test_device_run_tamper_is_rejected(tmp_path: Path) -> None:
    run = _run()
    run["phases"][1]["latencies_ns"][0] += 1
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_r2e5_run(path)


def test_compile_hardening_normalizes_latency_by_tokens() -> None:
    sealed = compile_r2e5_hardening(
        _manifest(),
        _e3(),
        _e4(),
        _run(sustained_latency_ns=320_000_000),
    )
    # warm = 10 ms/token; sustained = 320 ms / 64 = 5 ms/token.
    assert sealed["warm_step_per_token_p95_ns"] == 10_000_000
    assert sealed["sustained_per_token_p95_ns"] == 5_000_000
    assert sealed["sustained_p95_over_warm_ppm"] == 500_000
    assert sealed["sustained_guard_passed"] is True
    assert sealed["hardening_passed"] is True
    assert sealed["minimal_apk_graphs"] == [
        "step.onnx",
        "chunk-32.onnx",
        "chunk-8.onnx",
    ]


def test_compile_hardening_fails_sustained_degradation() -> None:
    sealed = compile_r2e5_hardening(
        _manifest(),
        _e3(),
        _e4(),
        _run(sustained_latency_ns=1_600_000_000),
    )
    # 1.6s / 64 = 25ms/token = 2.5x the warm-step baseline.
    assert sealed["sustained_p95_over_warm_ppm"] == 2_500_000
    assert sealed["sustained_guard_passed"] is False
    assert sealed["hardening_passed"] is False


def test_compile_hardening_fails_control_or_execution_failure() -> None:
    control_failed = compile_r2e5_hardening(
        _manifest(),
        _e3(),
        _e4(),
        _run(controls_pass=False),
    )
    assert control_failed["control_tests_passed"] is False
    assert control_failed["hardening_passed"] is False

    execution_failed = compile_r2e5_hardening(
        _manifest(),
        _e3(),
        _e4(),
        _run(failure_count=1),
    )
    assert execution_failed["failure_count"] == 1
    assert execution_failed["failure_guard_passed"] is False
    assert execution_failed["hardening_passed"] is False


def test_compile_requires_s21_fe_by_default() -> None:
    with pytest.raises(ValueError, match="not from a Galaxy S21 FE"):
        compile_r2e5_hardening(
            _manifest(),
            _e3(),
            _e4(),
            _run(model="SM-G991B"),
        )

    sealed = compile_r2e5_hardening(
        _manifest(),
        _e3(),
        _e4(),
        _run(model="SM-G991B"),
        require_s21_fe=False,
    )
    assert sealed["target_device_match"] is False


def test_hardening_seal_roundtrip_and_tamper(tmp_path: Path) -> None:
    sealed = compile_r2e5_hardening(
        _manifest(),
        _e3(),
        _e4(),
        _run(),
    )
    assert sealed["schema"] == R2E5_HARDENING_SCHEMA
    path = tmp_path / "hardening.json"
    path.write_bytes(_canonical(sealed) + b"\n")
    verified = verify_r2e5_hardening(
        path,
        expected_bundle_id=BUNDLE,
    )
    assert verified == sealed

    sealed["hardening_passed"] = False
    path.write_text(json.dumps(sealed), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_r2e5_hardening(path)


def test_recovery_thermal_guard_is_explicit() -> None:
    sealed = compile_r2e5_hardening(
        _manifest(),
        _e3(),
        _e4(),
        _run(recovery_thermal_after=5),
    )
    assert sealed["recovery_thermal_passed"] is False
    assert sealed["hardening_passed"] is False
