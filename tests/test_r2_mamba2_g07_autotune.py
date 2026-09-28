import hashlib
import json

import pytest

from vn97.r2.mamba2_autotune import (
    VN97_MAMBA2_G07_PROFILE_SCHEMA,
    VN97_MAMBA2_G07_TUNE_SCHEMA,
    compile_g07_tuning,
    verify_g07_profile,
    verify_g07_tuning,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _measurement(
    provider: str,
    *,
    p50: int,
    p95: int,
    actual: str | None = None,
    fallback: bool = False,
) -> dict[str, object]:
    return {
        "requested_provider": provider,
        "actual_provider": actual or provider,
        "fallback_used": fallback,
        "warmup_iterations": 3,
        "steady_iterations": 10,
        "session_create_ns": 2_000_000,
        "latency_p50_ns": p50,
        "latency_p95_ns": p95,
        "memory_before_bytes": 2_000_000_000,
        "memory_after_bytes": 1_900_000_000,
        "thermal_before": 1,
        "thermal_after": 2,
    }


def _profile() -> dict[str, object]:
    workloads = []
    for valid_length in (1, 8, 16, 32):
        base = valid_length * 1_000_000
        workloads.append(
            {
                "valid_length": valid_length,
                "measurements": [
                    _measurement(
                        "QNN",
                        p50=base + 100_000,
                        p95=base + 200_000,
                    ),
                    _measurement(
                        "XNNPACK",
                        p50=base + 200_000,
                        p95=base + 350_000,
                    ),
                    _measurement(
                        "CPU",
                        p50=base + 400_000,
                        p95=base + 600_000,
                    ),
                ],
            }
        )
    body: dict[str, object] = {
        "schema": VN97_MAMBA2_G07_PROFILE_SCHEMA,
        "runtime_id": "a" * 64,
        "graph_filename": "recurrent-32.onnx",
        "max_chunk_size": 32,
        "device": {
            "sdk_int": 36,
            "logical_cores": 8,
            "hardware": "qcom",
            "soc_manufacturer": "Qualcomm",
            "soc_model": "SM8350",
            "profiled_available_memory_bytes": 3_000_000_000,
        },
        "workloads": workloads,
        "device_measured": True,
        "synthetic": False,
        "same_weights_semantics": True,
        "production_activation_authorized": False,
    }
    return {
        **body,
        "receipt_id": hashlib.sha256(
            b"VN97M2G07PROFILE1\0" + _canonical(body)
        ).hexdigest(),
    }


def _reseal(profile: dict[str, object]) -> dict[str, object]:
    body = dict(profile)
    body.pop("receipt_id", None)
    body["receipt_id"] = hashlib.sha256(
        b"VN97M2G07PROFILE1\0" + _canonical(body)
    ).hexdigest()
    return body


def test_g07_profile_requires_exact_device_measured_workload_coverage() -> None:
    profile = _profile()
    verified = verify_g07_profile(
        profile,
        expected_runtime_id="a" * 64,
    )
    assert verified["device_measured"] is True
    assert [item["valid_length"] for item in verified["workloads"]] == [
        1,
        8,
        16,
        32,
    ]


def test_g07_tuning_ranks_direct_measurements_and_keeps_cpu_final() -> None:
    tuning = compile_g07_tuning(
        _profile(),
        expected_runtime_id="a" * 64,
    )
    assert tuning["schema"] == VN97_MAMBA2_G07_TUNE_SCHEMA
    assert tuning["runtime_id"] == "a" * 64
    assert tuning["profile_is_device_measured"] is True
    assert tuning["production_activation_authorized"] is False
    policies = tuning["workload_policies"]
    assert [item["valid_length"] for item in policies] == [1, 8, 16, 32]
    assert all(
        item["provider_order"] == ["QNN", "XNNPACK", "CPU"]
        for item in policies
    )
    assert all(item["preferred_provider"] == "QNN" for item in policies)
    verify_g07_tuning(
        tuning,
        expected_runtime_id="a" * 64,
    )


def test_g07_excludes_measurement_that_used_provider_fallback() -> None:
    profile = _profile()
    first = profile["workloads"][0]["measurements"][0]
    first["actual_provider"] = "CPU"
    first["fallback_used"] = True
    profile = _reseal(profile)

    tuning = compile_g07_tuning(profile)
    first_policy = tuning["workload_policies"][0]
    assert first_policy["provider_order"] == ["XNNPACK", "CPU"]
    assert first_policy["preferred_provider"] == "XNNPACK"


def test_g07_rejects_synthetic_profile_even_if_hash_is_valid() -> None:
    profile = _profile()
    profile["synthetic"] = True
    profile = _reseal(profile)
    with pytest.raises(ValueError, match="synthetic"):
        compile_g07_tuning(profile)


def test_g07_rejects_missing_decode_profile() -> None:
    profile = _profile()
    profile["workloads"] = [
        item
        for item in profile["workloads"]
        if item["valid_length"] != 1
    ]
    profile = _reseal(profile)
    with pytest.raises(ValueError, match="coverage"):
        verify_g07_profile(profile)
