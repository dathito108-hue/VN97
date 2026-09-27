from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from .onnx_export import verify_r2_onnx_bundle
from .onnx_profiling import (
    R2E3_ALLOWED_PROVIDERS,
    verify_r2e3_receipt,
)


R2E4_PROFILE_SCHEMA = "VN97R2E4TUNE1"
R2E4_THERMAL_HOT_ENTER = 4
R2E4_THERMAL_HOT_EXIT = 2
R2E4_THERMAL_CRITICAL_ENTER = 5
R2E4_THERMAL_CRITICAL_EXIT = 3
R2E4_MEMORY_PRESSURE_ENTER_PPM = 300_000
R2E4_MEMORY_PRESSURE_EXIT_PPM = 450_000
R2E4_LATENCY_SLOW_ENTER_PPM = 1_250_000
R2E4_LATENCY_SLOW_EXIT_PPM = 1_100_000
R2E4_QUARANTINE_FAILURE_THRESHOLD = 2
R2E4_QUARANTINE_COOLDOWN_DECISIONS = 16


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _device_key(device: Mapping[str, object]) -> str:
    stable = {
        "sdk_int": int(device["sdk_int"]),
        "logical_cores": int(device["logical_cores"]),
        "hardware": str(device["hardware"]),
        "soc_manufacturer": str(device["soc_manufacturer"]),
        "soc_model": str(device["soc_model"]),
    }
    return _sha256_bytes(
        b"VN97R2E4DEVICE1\0" + _canonical_json(stable)
    )


def _graph_records(
    manifest: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    raw = manifest.get("graphs")
    if not isinstance(raw, list) or not raw:
        raise ValueError("E4 requires E2 graph records")
    records: dict[str, dict[str, object]] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("E4 graph record must be an object")
        filename = item.get("filename")
        kind = item.get("kind")
        sequence_length = item.get("sequence_length")
        if (
            not isinstance(filename, str)
            or not filename
            or kind not in {"step", "chunk"}
            or not isinstance(sequence_length, int)
            or sequence_length <= 0
        ):
            raise ValueError("E4 graph record is invalid")
        if filename in records:
            raise ValueError("E4 graph filenames must be unique")
        records[filename] = item
    if "step.onnx" not in records:
        raise ValueError("E4 requires step.onnx")
    return records


def _receipt_results(
    receipt: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    raw = receipt.get("graph_results")
    if not isinstance(raw, list) or not raw:
        raise ValueError("E4 requires E3 graph results")
    results: dict[str, dict[str, object]] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("E4 E3 graph result must be an object")
        filename = item.get("graph_filename")
        if not isinstance(filename, str) or not filename:
            raise ValueError("E4 E3 graph filename is invalid")
        if filename in results:
            raise ValueError("E4 E3 graph results must be unique")
        results[filename] = item
    return results


def _provider_policy(
    graph: Mapping[str, object],
    result: Mapping[str, object],
) -> dict[str, object]:
    ranked = result.get("ranked_providers")
    if not isinstance(ranked, list) or not ranked:
        raise ValueError("E4 provider measurements are missing")

    usable: list[dict[str, object]] = []
    cpu_seen = False
    for item in ranked:
        if not isinstance(item, dict):
            raise ValueError("E4 provider measurement is invalid")
        requested = item.get("requested_provider")
        actual = item.get("actual_provider")
        fallback = item.get("fallback_used")
        if requested not in R2E3_ALLOWED_PROVIDERS:
            raise ValueError("E4 provider name is invalid")
        if actual not in R2E3_ALLOWED_PROVIDERS:
            raise ValueError("E4 actual provider name is invalid")
        if not isinstance(fallback, bool):
            raise ValueError("E4 fallback flag is invalid")
        if fallback or requested != actual:
            continue
        for field in (
            "latency_p50_ns",
            "latency_p95_ns",
            "tokens_per_second_milli_p50",
        ):
            value = item.get(field)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"E4 provider measurement {field} is invalid"
                )
        normalized = {
            "provider": requested,
            "latency_p50_ns": int(item["latency_p50_ns"]),
            "latency_p95_ns": int(item["latency_p95_ns"]),
            "tokens_per_second_milli_p50": int(
                item["tokens_per_second_milli_p50"]
            ),
            "thermal_before": int(item["thermal_before"]),
            "thermal_after": int(item["thermal_after"]),
        }
        usable.append(normalized)
        if requested == "CPU":
            cpu_seen = True

    if not usable:
        raise ValueError(
            "E4 graph has no successful non-fallback provider"
        )
    if not cpu_seen:
        raise ValueError(
            "E4 graph requires a successful CPU fallback measurement"
        )

    usable.sort(
        key=lambda item: (
            int(item["latency_p50_ns"]),
            int(item["latency_p95_ns"]),
            str(item["provider"]),
        )
    )
    ranked_order = [str(item["provider"]) for item in usable]
    if ranked_order[0] == "CPU":
        provider_order = ["CPU"]
    else:
        provider_order = [
            provider
            for provider in ranked_order
            if provider != "CPU"
        ] + ["CPU"]

    return {
        "graph_filename": str(graph["filename"]),
        "kind": str(graph["kind"]),
        "sequence_length": int(graph["sequence_length"]),
        "provider_order": provider_order,
        "preferred_provider": provider_order[0],
        "measurements": usable,
    }


def compile_r2e4_profile(
    manifest: Mapping[str, object],
    receipt: Mapping[str, object],
) -> dict[str, object]:
    bundle_id = _require_sha256(
        manifest.get("bundle_id"),
        label="E4 bundle ID",
    )
    if receipt.get("bundle_id") != bundle_id:
        raise ValueError("E4 receipt belongs to another E2 bundle")
    if receipt.get("architecture_fingerprint") != manifest.get(
        "architecture_fingerprint"
    ):
        raise ValueError("E4 architecture fingerprint mismatch")
    if receipt.get("profile") != manifest.get("profile"):
        raise ValueError("E4 fast/deep profile mismatch")
    if receipt.get("device_measured") is not True:
        raise ValueError("E4 requires device-measured E3 evidence")
    if receipt.get("synthetic") is not False:
        raise ValueError("E4 refuses synthetic E3 evidence")

    receipt_id = _require_sha256(
        receipt.get("receipt_id"),
        label="E4 E3 receipt ID",
    )
    graphs = _graph_records(manifest)
    results = _receipt_results(receipt)
    if set(results) != set(graphs):
        missing = sorted(set(graphs) - set(results))
        extra = sorted(set(results) - set(graphs))
        raise ValueError(
            "E4 requires exact E2 graph coverage; "
            f"missing={missing} extra={extra}"
        )

    policies = [
        _provider_policy(graphs[name], results[name])
        for name in sorted(graphs)
    ]

    chunk_policies = [
        item
        for item in policies
        if item["kind"] == "chunk"
    ]
    if not chunk_policies:
        raise ValueError("E4 requires at least one chunk graph")

    def preferred_measurement(
        policy: Mapping[str, object],
    ) -> Mapping[str, object]:
        preferred = policy["preferred_provider"]
        measurements = policy["measurements"]
        assert isinstance(measurements, list)
        return next(
            item
            for item in measurements
            if item["provider"] == preferred
        )

    chunk_policies.sort(
        key=lambda item: (
            -int(
                preferred_measurement(item)[
                    "tokens_per_second_milli_p50"
                ]
            ),
            -int(item["sequence_length"]),
            str(item["graph_filename"]),
        )
    )
    preferred_chunks = [
        int(item["sequence_length"])
        for item in chunk_policies
    ]

    device = receipt.get("device")
    if not isinstance(device, dict):
        raise ValueError("E4 device evidence is missing")
    baseline_memory = device.get("available_memory_bytes")
    logical_cores = device.get("logical_cores")
    if (
        not isinstance(baseline_memory, int)
        or baseline_memory <= 0
        or not isinstance(logical_cores, int)
        or logical_cores <= 0
    ):
        raise ValueError("E4 device memory/core evidence is invalid")

    control_policy = {
        "thermal_hot_enter": R2E4_THERMAL_HOT_ENTER,
        "thermal_hot_exit": R2E4_THERMAL_HOT_EXIT,
        "thermal_critical_enter": R2E4_THERMAL_CRITICAL_ENTER,
        "thermal_critical_exit": R2E4_THERMAL_CRITICAL_EXIT,
        "memory_pressure_enter_ppm": (
            R2E4_MEMORY_PRESSURE_ENTER_PPM
        ),
        "memory_pressure_exit_ppm": (
            R2E4_MEMORY_PRESSURE_EXIT_PPM
        ),
        "latency_slow_enter_ppm": R2E4_LATENCY_SLOW_ENTER_PPM,
        "latency_slow_exit_ppm": R2E4_LATENCY_SLOW_EXIT_PPM,
        "quarantine_failure_threshold": (
            R2E4_QUARANTINE_FAILURE_THRESHOLD
        ),
        "quarantine_cooldown_decisions": (
            R2E4_QUARANTINE_COOLDOWN_DECISIONS
        ),
        "xnnpack_threads_normal": min(4, logical_cores),
        "xnnpack_threads_hot": min(2, logical_cores),
        "xnnpack_threads_critical": 1,
    }

    body = {
        "schema": R2E4_PROFILE_SCHEMA,
        "bundle_id": bundle_id,
        "architecture_fingerprint": _require_sha256(
            manifest.get("architecture_fingerprint"),
            label="E4 architecture fingerprint",
        ),
        "profile": manifest.get("profile"),
        "e3_receipt_id": receipt_id,
        "device_key": _device_key(device),
        "device": {
            "sdk_int": int(device["sdk_int"]),
            "logical_cores": logical_cores,
            "hardware": str(device["hardware"]),
            "soc_manufacturer": str(device["soc_manufacturer"]),
            "soc_model": str(device["soc_model"]),
            "profiled_available_memory_bytes": baseline_memory,
        },
        "graph_policies": policies,
        "preferred_chunk_sizes": preferred_chunks,
        "control_policy": control_policy,
        "same_weights_semantics": True,
        "quantization_used": False,
    }
    profile = dict(body)
    profile["tuning_id"] = _sha256_bytes(
        b"VN97R2E4TUNE1\0" + _canonical_json(body)
    )
    return profile


def verify_r2e4_profile(
    path: Path,
    *,
    expected_bundle_id: str | None = None,
) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("E4 profile must be a regular non-symlink file")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8", errors="strict"),
            parse_constant=lambda raw: (
                _ for _ in ()
            ).throw(ValueError(raw)),
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise ValueError("E4 profile must be strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("E4 profile must be an object")
    if payload.get("schema") != R2E4_PROFILE_SCHEMA:
        raise ValueError("E4 profile schema mismatch")
    tuning_id = _require_sha256(
        payload.get("tuning_id"),
        label="E4 tuning ID",
    )
    body = dict(payload)
    body.pop("tuning_id", None)
    if tuning_id != _sha256_bytes(
        b"VN97R2E4TUNE1\0" + _canonical_json(body)
    ):
        raise ValueError("E4 tuning profile identity mismatch")

    bundle_id = _require_sha256(
        payload.get("bundle_id"),
        label="E4 bundle ID",
    )
    if expected_bundle_id is not None and bundle_id != expected_bundle_id:
        raise ValueError("E4 profile belongs to another E2 bundle")

    if payload.get("same_weights_semantics") is not True:
        raise ValueError("E4 same-weights semantics are not locked")
    if payload.get("quantization_used") is not False:
        raise ValueError("E4 must not introduce quantization")

    device = payload.get("device")
    if not isinstance(device, dict):
        raise ValueError("E4 device identity is missing")
    required_device = {
        "sdk_int",
        "logical_cores",
        "hardware",
        "soc_manufacturer",
        "soc_model",
        "profiled_available_memory_bytes",
    }
    if set(device) != required_device:
        raise ValueError("E4 device identity fields mismatch")
    stable_device = {
        key: device[key]
        for key in (
            "sdk_int",
            "logical_cores",
            "hardware",
            "soc_manufacturer",
            "soc_model",
        )
    }
    if payload.get("device_key") != _device_key(stable_device):
        raise ValueError("E4 device key mismatch")

    policies = payload.get("graph_policies")
    if not isinstance(policies, list) or not policies:
        raise ValueError("E4 graph policies are missing")
    names: set[str] = set()
    step_seen = False
    cpu_fallback_all = True
    chunk_sizes: list[int] = []
    for item in policies:
        if not isinstance(item, dict):
            raise ValueError("E4 graph policy is invalid")
        name = item.get("graph_filename")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("E4 graph policy filename is invalid")
        names.add(name)
        kind = item.get("kind")
        sequence_length = item.get("sequence_length")
        if kind == "step":
            step_seen = True
        elif kind == "chunk":
            if not isinstance(sequence_length, int):
                raise ValueError("E4 chunk size is invalid")
            chunk_sizes.append(sequence_length)
        else:
            raise ValueError("E4 graph policy kind is invalid")
        order = item.get("provider_order")
        if (
            not isinstance(order, list)
            or not order
            or any(
                provider not in R2E3_ALLOWED_PROVIDERS
                for provider in order
            )
            or len(order) != len(set(order))
        ):
            raise ValueError("E4 provider order is invalid")
        cpu_fallback_all = (
            cpu_fallback_all and order[-1] == "CPU"
        )
        if item.get("preferred_provider") != order[0]:
            raise ValueError("E4 preferred provider mismatch")
    if not step_seen or not chunk_sizes:
        raise ValueError("E4 requires step and chunk policies")
    if not cpu_fallback_all:
        raise ValueError("E4 CPU must remain final fallback")

    preferred = payload.get("preferred_chunk_sizes")
    if (
        not isinstance(preferred, list)
        or sorted(preferred) != sorted(chunk_sizes)
        or len(preferred) != len(set(preferred))
    ):
        raise ValueError("E4 preferred chunk set mismatch")

    control = payload.get("control_policy")
    if not isinstance(control, dict):
        raise ValueError("E4 control policy is missing")
    required_control = {
        "thermal_hot_enter",
        "thermal_hot_exit",
        "thermal_critical_enter",
        "thermal_critical_exit",
        "memory_pressure_enter_ppm",
        "memory_pressure_exit_ppm",
        "latency_slow_enter_ppm",
        "latency_slow_exit_ppm",
        "quarantine_failure_threshold",
        "quarantine_cooldown_decisions",
        "xnnpack_threads_normal",
        "xnnpack_threads_hot",
        "xnnpack_threads_critical",
    }
    if set(control) != required_control:
        raise ValueError("E4 control policy fields mismatch")
    if not (
        int(control["thermal_hot_exit"])
        < int(control["thermal_hot_enter"])
        < int(control["thermal_critical_enter"])
    ):
        raise ValueError("E4 thermal hysteresis is invalid")
    if not (
        int(control["memory_pressure_enter_ppm"])
        < int(control["memory_pressure_exit_ppm"])
    ):
        raise ValueError("E4 memory hysteresis is invalid")
    if not (
        int(control["latency_slow_exit_ppm"])
        < int(control["latency_slow_enter_ppm"])
    ):
        raise ValueError("E4 latency hysteresis is invalid")
    return payload


def build_r2e4_profile_from_files(
    *,
    bundle_dir: Path,
    receipt_path: Path,
    output_path: Path,
) -> dict[str, object]:
    manifest = verify_r2_onnx_bundle(bundle_dir)
    receipt = verify_r2e3_receipt(
        receipt_path,
        expected_bundle_id=str(manifest["bundle_id"]),
    )
    profile = compile_r2e4_profile(manifest, receipt)
    if output_path.exists() or output_path.is_symlink():
        raise ValueError("E4 output profile must not already exist")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_name(output_path.name + ".tmp")
    temp.write_bytes(_canonical_json(profile) + b"\n")
    temp.replace(output_path)
    verify_r2e4_profile(
        output_path,
        expected_bundle_id=str(manifest["bundle_id"]),
    )
    return profile
