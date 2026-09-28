from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


VN97_MAMBA2_G07_PROFILE_SCHEMA = "VN97M2G07PROFILE1"
VN97_MAMBA2_G07_TUNE_SCHEMA = "VN97M2G07TUNE1"
VN97_MAMBA2_G07_TUNE_FILENAME = "tuning.vn97m2g07.json"
_ALLOWED_PROVIDERS = ("QNN", "NNAPI", "XNNPACK", "CPU")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _required_valid_lengths(max_chunk_size: int) -> tuple[int, ...]:
    if max_chunk_size not in (8, 16, 32):
        raise ValueError("G0.7 max chunk size must be 8, 16, or 32")
    return tuple(
        value
        for value in (1, 8, 16, 32)
        if value == 1 or value <= max_chunk_size
    )


def _device_identity(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("G0.7 device identity is missing")
    expected = {
        "sdk_int",
        "logical_cores",
        "hardware",
        "soc_manufacturer",
        "soc_model",
        "profiled_available_memory_bytes",
    }
    if set(value) != expected:
        raise ValueError("G0.7 device identity fields mismatch")
    sdk = value["sdk_int"]
    cores = value["logical_cores"]
    memory = value["profiled_available_memory_bytes"]
    if not isinstance(sdk, int) or sdk < 26:
        raise ValueError("G0.7 sdk_int must be Android API 26+")
    if not isinstance(cores, int) or cores <= 0:
        raise ValueError("G0.7 logical_cores must be positive")
    if not isinstance(memory, int) or memory <= 0:
        raise ValueError(
            "G0.7 profiled_available_memory_bytes must be positive"
        )
    output: dict[str, object] = {
        "sdk_int": sdk,
        "logical_cores": cores,
        "profiled_available_memory_bytes": memory,
    }
    for field in ("hardware", "soc_manufacturer", "soc_model"):
        item = value[field]
        if not isinstance(item, str):
            raise ValueError(f"G0.7 device {field} must be string")
        output[field] = item
    return output


def verify_g07_profile(
    payload: Mapping[str, object],
    *,
    expected_runtime_id: str | None = None,
) -> dict[str, object]:
    if payload.get("schema") != VN97_MAMBA2_G07_PROFILE_SCHEMA:
        raise ValueError("G0.7 profile schema mismatch")
    receipt_id = _require_sha256(
        payload.get("receipt_id"),
        "G0.7 profile receipt ID",
    )
    body = dict(payload)
    body.pop("receipt_id", None)
    expected_receipt = _sha256_bytes(
        b"VN97M2G07PROFILE1\0" + _canonical_json(body)
    )
    if receipt_id != expected_receipt:
        raise ValueError("G0.7 profile receipt identity mismatch")

    runtime_id = _require_sha256(
        payload.get("runtime_id"),
        "G0.7 runtime ID",
    )
    if expected_runtime_id is not None and runtime_id != expected_runtime_id:
        raise ValueError("G0.7 profile belongs to another runtime")
    graph = payload.get("graph_filename")
    if not isinstance(graph, str) or not graph.startswith("recurrent-"):
        raise ValueError("G0.7 graph filename invalid")
    max_chunk = payload.get("max_chunk_size")
    if not isinstance(max_chunk, int):
        raise ValueError("G0.7 max chunk size missing")
    expected_graph = f"recurrent-{max_chunk}.onnx"
    if graph != expected_graph:
        raise ValueError("G0.7 graph filename/max chunk mismatch")

    _device_identity(payload.get("device"))
    if payload.get("device_measured") is not True:
        raise ValueError("G0.7 profile must be device measured")
    if payload.get("synthetic") is not False:
        raise ValueError("G0.7 synthetic profile cannot tune production")
    if payload.get("same_weights_semantics") is not True:
        raise ValueError("G0.7 same-weight semantics are not locked")
    if payload.get("production_activation_authorized") is not False:
        raise ValueError("G0.7 profile cannot self-authorize production")

    workloads = payload.get("workloads")
    if not isinstance(workloads, list):
        raise ValueError("G0.7 workloads are missing")
    by_length: dict[int, dict[str, object]] = {}
    for workload in workloads:
        if not isinstance(workload, Mapping):
            raise ValueError("G0.7 workload record must be object")
        if set(workload) != {"valid_length", "measurements"}:
            raise ValueError("G0.7 workload fields mismatch")
        valid_length = workload.get("valid_length")
        if (
            not isinstance(valid_length, int)
            or valid_length <= 0
            or valid_length > max_chunk
            or valid_length in by_length
        ):
            raise ValueError("G0.7 workload valid_length invalid")
        measurements = workload.get("measurements")
        if not isinstance(measurements, list) or not measurements:
            raise ValueError("G0.7 workload measurements are missing")
        seen_provider: set[str] = set()
        normalized: list[dict[str, object]] = []
        cpu_usable = False
        for measurement in measurements:
            if not isinstance(measurement, Mapping):
                raise ValueError("G0.7 measurement must be object")
            expected_fields = {
                "requested_provider",
                "actual_provider",
                "fallback_used",
                "warmup_iterations",
                "steady_iterations",
                "session_create_ns",
                "latency_p50_ns",
                "latency_p95_ns",
                "memory_before_bytes",
                "memory_after_bytes",
                "thermal_before",
                "thermal_after",
            }
            if set(measurement) != expected_fields:
                raise ValueError("G0.7 measurement fields mismatch")
            requested = measurement.get("requested_provider")
            actual = measurement.get("actual_provider")
            if requested not in _ALLOWED_PROVIDERS:
                raise ValueError("G0.7 requested provider invalid")
            if actual not in _ALLOWED_PROVIDERS:
                raise ValueError("G0.7 actual provider invalid")
            if requested in seen_provider:
                raise ValueError("G0.7 duplicate requested provider")
            seen_provider.add(str(requested))
            fallback = measurement.get("fallback_used")
            if not isinstance(fallback, bool):
                raise ValueError("G0.7 fallback_used must be boolean")
            warmup = measurement.get("warmup_iterations")
            steady = measurement.get("steady_iterations")
            if not isinstance(warmup, int) or warmup < 1:
                raise ValueError("G0.7 warmup_iterations invalid")
            if not isinstance(steady, int) or steady < 3:
                raise ValueError("G0.7 steady_iterations invalid")
            for field in (
                "session_create_ns",
                "latency_p50_ns",
                "latency_p95_ns",
                "memory_before_bytes",
                "memory_after_bytes",
            ):
                value = measurement.get(field)
                if not isinstance(value, int) or value <= 0:
                    raise ValueError(f"G0.7 {field} must be positive")
            if int(measurement["latency_p95_ns"]) < int(
                measurement["latency_p50_ns"]
            ):
                raise ValueError("G0.7 p95 latency cannot be below p50")
            for field in ("thermal_before", "thermal_after"):
                value = measurement.get(field)
                if not isinstance(value, int) or value not in range(0, 7):
                    raise ValueError(f"G0.7 {field} must be in [0,6]")
            if (
                requested == "CPU"
                and actual == "CPU"
                and fallback is False
            ):
                cpu_usable = True
            normalized.append(dict(measurement))
        if not cpu_usable:
            raise ValueError(
                "G0.7 each workload requires direct CPU measurement"
            )
        by_length[valid_length] = {
            "valid_length": valid_length,
            "measurements": normalized,
        }

    required = _required_valid_lengths(max_chunk)
    if set(by_length) != set(required):
        raise ValueError(
            f"G0.7 workload coverage must be exactly {required}"
        )
    return dict(payload)


def _usable_measurements(
    workload: Mapping[str, object],
) -> list[dict[str, object]]:
    measurements = workload["measurements"]
    assert isinstance(measurements, list)
    usable = [
        dict(item)
        for item in measurements
        if isinstance(item, Mapping)
        and item.get("fallback_used") is False
        and item.get("requested_provider") == item.get("actual_provider")
    ]
    if not any(item["requested_provider"] == "CPU" for item in usable):
        raise ValueError("G0.7 usable CPU measurement missing")
    return usable


def compile_g07_tuning(
    profile: Mapping[str, object],
    *,
    expected_runtime_id: str | None = None,
) -> dict[str, object]:
    verified = verify_g07_profile(
        profile,
        expected_runtime_id=expected_runtime_id,
    )
    runtime_id = str(verified["runtime_id"])
    receipt_id = str(verified["receipt_id"])
    max_chunk = int(verified["max_chunk_size"])
    device = _device_identity(verified["device"])

    policies: list[dict[str, object]] = []
    workloads = verified["workloads"]
    assert isinstance(workloads, list)
    for workload in sorted(
        workloads,
        key=lambda item: int(item["valid_length"]),
    ):
        valid_length = int(workload["valid_length"])
        usable = _usable_measurements(workload)
        ranked = sorted(
            usable,
            key=lambda item: (
                int(item["latency_p95_ns"]),
                int(item["latency_p50_ns"]),
                str(item["requested_provider"]),
            ),
        )
        provider_order = [
            str(item["requested_provider"])
            for item in ranked
            if item["requested_provider"] != "CPU"
        ]
        provider_order.append("CPU")
        preferred = provider_order[0]
        preferred_measurement = next(
            item
            for item in ranked
            if item["requested_provider"] == preferred
        )
        policies.append(
            {
                "valid_length": valid_length,
                "provider_order": provider_order,
                "preferred_provider": preferred,
                "latency_p50_ns": int(
                    preferred_measurement["latency_p50_ns"]
                ),
                "latency_p95_ns": int(
                    preferred_measurement["latency_p95_ns"]
                ),
            }
        )

    cores = int(device["logical_cores"])
    body: dict[str, object] = {
        "schema": VN97_MAMBA2_G07_TUNE_SCHEMA,
        "runtime_id": runtime_id,
        "profile_receipt_id": receipt_id,
        "graph_filename": verified["graph_filename"],
        "max_chunk_size": max_chunk,
        "device": device,
        "workload_policies": policies,
        "control_policy": {
            "thermal_hot_enter": 4,
            "thermal_critical_enter": 5,
            "xnnpack_threads_normal": min(4, cores),
            "xnnpack_threads_hot": min(2, cores),
            "xnnpack_threads_critical": 1,
        },
        "profile_is_device_measured": True,
        "same_weights_semantics": True,
        "quantization_used": False,
        "production_activation_authorized": False,
    }
    result = dict(body)
    result["tuning_id"] = _sha256_bytes(
        b"VN97M2G07TUNE1\0" + _canonical_json(body)
    )
    return result


def verify_g07_tuning(
    payload: Mapping[str, object],
    *,
    expected_runtime_id: str | None = None,
) -> dict[str, object]:
    if payload.get("schema") != VN97_MAMBA2_G07_TUNE_SCHEMA:
        raise ValueError("G0.7 tuning schema mismatch")
    tuning_id = _require_sha256(
        payload.get("tuning_id"),
        "G0.7 tuning ID",
    )
    body = dict(payload)
    body.pop("tuning_id", None)
    expected = _sha256_bytes(
        b"VN97M2G07TUNE1\0" + _canonical_json(body)
    )
    if tuning_id != expected:
        raise ValueError("G0.7 tuning identity mismatch")
    runtime_id = _require_sha256(
        payload.get("runtime_id"),
        "G0.7 tuning runtime ID",
    )
    if expected_runtime_id is not None and runtime_id != expected_runtime_id:
        raise ValueError("G0.7 tuning belongs to another runtime")
    _require_sha256(
        payload.get("profile_receipt_id"),
        "G0.7 profile receipt ID",
    )
    max_chunk = payload.get("max_chunk_size")
    if not isinstance(max_chunk, int):
        raise ValueError("G0.7 tuning max chunk missing")
    if payload.get("graph_filename") != f"recurrent-{max_chunk}.onnx":
        raise ValueError("G0.7 tuning graph mismatch")
    _device_identity(payload.get("device"))
    policies = payload.get("workload_policies")
    if not isinstance(policies, list):
        raise ValueError("G0.7 tuning policies missing")
    lengths: list[int] = []
    for policy in policies:
        if not isinstance(policy, Mapping):
            raise ValueError("G0.7 tuning policy invalid")
        if set(policy) != {
            "valid_length",
            "provider_order",
            "preferred_provider",
            "latency_p50_ns",
            "latency_p95_ns",
        }:
            raise ValueError("G0.7 tuning policy fields mismatch")
        valid_length = policy["valid_length"]
        if not isinstance(valid_length, int):
            raise ValueError("G0.7 policy valid_length invalid")
        lengths.append(valid_length)
        order = policy["provider_order"]
        if (
            not isinstance(order, list)
            or not order
            or len(set(order)) != len(order)
            or any(item not in _ALLOWED_PROVIDERS for item in order)
            or order[-1] != "CPU"
        ):
            raise ValueError("G0.7 provider order invalid")
        if policy["preferred_provider"] != order[0]:
            raise ValueError("G0.7 preferred provider mismatch")
        p50 = policy["latency_p50_ns"]
        p95 = policy["latency_p95_ns"]
        if (
            not isinstance(p50, int)
            or not isinstance(p95, int)
            or p50 <= 0
            or p95 < p50
        ):
            raise ValueError("G0.7 tuning latency invalid")
    if tuple(sorted(lengths)) != _required_valid_lengths(max_chunk):
        raise ValueError("G0.7 tuning workload coverage mismatch")
    control = payload.get("control_policy")
    if not isinstance(control, Mapping):
        raise ValueError("G0.7 control policy missing")
    if set(control) != {
        "thermal_hot_enter",
        "thermal_critical_enter",
        "xnnpack_threads_normal",
        "xnnpack_threads_hot",
        "xnnpack_threads_critical",
    }:
        raise ValueError("G0.7 control policy fields mismatch")
    if not (
        0
        <= int(control["thermal_hot_enter"])
        < int(control["thermal_critical_enter"])
        <= 6
    ):
        raise ValueError("G0.7 thermal policy invalid")
    for field in (
        "xnnpack_threads_normal",
        "xnnpack_threads_hot",
        "xnnpack_threads_critical",
    ):
        if not isinstance(control[field], int) or int(control[field]) <= 0:
            raise ValueError("G0.7 XNNPACK thread policy invalid")
    if payload.get("profile_is_device_measured") is not True:
        raise ValueError("G0.7 tuning requires device measured evidence")
    if payload.get("same_weights_semantics") is not True:
        raise ValueError("G0.7 tuning same-weight semantics missing")
    if payload.get("quantization_used") is not False:
        raise ValueError("G0.7 tuning cannot introduce quantization")
    if payload.get("production_activation_authorized") is not False:
        raise ValueError("G0.7 tuning cannot self-authorize production")
    return dict(payload)


def build_g07_tuning_file(
    *,
    profile_path: Path,
    output_path: Path,
    expected_runtime_id: str | None = None,
) -> dict[str, object]:
    if profile_path.is_symlink() or not profile_path.is_file():
        raise ValueError("G0.7 profile path must be regular file")
    profile = json.loads(profile_path.read_text(encoding="ascii"))
    if not isinstance(profile, dict):
        raise ValueError("G0.7 profile must be JSON object")
    tuning = compile_g07_tuning(
        profile,
        expected_runtime_id=expected_runtime_id,
    )
    verify_g07_tuning(
        tuning,
        expected_runtime_id=expected_runtime_id,
    )
    if output_path.exists() or output_path.is_symlink():
        raise ValueError("G0.7 tuning output must not already exist")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_name(output_path.name + ".tmp")
    temp.write_bytes(_canonical_json(tuning) + b"\n")
    temp.replace(output_path)
    return tuning
