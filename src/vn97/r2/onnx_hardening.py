from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from .onnx_autotune import verify_r2e4_profile
from .onnx_export import verify_r2_onnx_bundle
from .onnx_profiling import verify_r2e3_receipt


R2E5_RUN_SCHEMA = "VN97R2E5RUN1"
R2E5_HARDENING_SCHEMA = "VN97R2E5HARDEN1"
R2E5_TARGET_FAMILY = "galaxy_s21_fe"
R2E5_S21FE_MODEL_PREFIX = "SM-G990"
R2E5_MAX_SUSTAINED_P95_OVER_WARM_PPM = 2_000_000


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


def _strict_json(path: Path, *, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    try:
        value = json.loads(
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
        raise ValueError(f"{label} must be strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be one JSON object")
    return value


def percentile_nearest_rank_int(
    values: Sequence[int],
    q: float,
) -> int:
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 < q <= 1.0:
        raise ValueError("q must be in (0, 1]")
    ordered = sorted(int(value) for value in values)
    if any(value <= 0 for value in ordered):
        raise ValueError("values must be positive")
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[rank - 1]


def _validate_device(device: Mapping[str, object]) -> None:
    required = {
        "manufacturer",
        "model",
        "sdk_int",
        "logical_cores",
        "hardware",
        "soc_manufacturer",
        "soc_model",
    }
    if set(device) != required:
        raise ValueError("E5 device identity fields mismatch")
    if not isinstance(device["manufacturer"], str):
        raise ValueError("E5 manufacturer must be a string")
    if not isinstance(device["model"], str):
        raise ValueError("E5 model must be a string")
    if (
        not isinstance(device["sdk_int"], int)
        or int(device["sdk_int"]) < 26
    ):
        raise ValueError("E5 sdk_int is invalid")
    if (
        not isinstance(device["logical_cores"], int)
        or int(device["logical_cores"]) <= 0
    ):
        raise ValueError("E5 logical_cores is invalid")


def is_s21_fe_device(device: Mapping[str, object]) -> bool:
    _validate_device(device)
    manufacturer = str(device["manufacturer"]).strip().lower()
    model = str(device["model"]).strip().upper()
    return (
        "samsung" in manufacturer
        and model.startswith(R2E5_S21FE_MODEL_PREFIX)
    )


def _validate_phase(phase: Mapping[str, object]) -> None:
    required = {
        "name",
        "kind",
        "iterations",
        "tokens_per_iteration",
        "latencies_ns",
        "provider_counts",
        "graph_counts",
        "thermal_before",
        "thermal_after",
        "memory_before_bytes",
        "memory_after_bytes",
        "failure_count",
    }
    if set(phase) != required:
        raise ValueError("E5 phase fields mismatch")
    if phase["kind"] not in {
        "cold",
        "warm_step",
        "prefill",
        "sustained",
        "recovery",
    }:
        raise ValueError("E5 phase kind is invalid")
    iterations = phase["iterations"]
    tokens_per_iteration = phase["tokens_per_iteration"]
    latencies = phase["latencies_ns"]
    if (
        not isinstance(iterations, int)
        or iterations <= 0
        or not isinstance(tokens_per_iteration, int)
        or tokens_per_iteration <= 0
        or not isinstance(latencies, list)
        or len(latencies) != iterations
        or any(
            not isinstance(item, int) or item <= 0
            for item in latencies
        )
    ):
        raise ValueError("E5 phase latency evidence is invalid")
    for field in ("provider_counts", "graph_counts"):
        counts = phase[field]
        if (
            not isinstance(counts, dict)
            or not counts
            or sum(
                value
                for value in counts.values()
                if isinstance(value, int)
            ) <= 0
            or any(
                not isinstance(key, str)
                or not key
                or not isinstance(value, int)
                or value < 0
                for key, value in counts.items()
            )
        ):
            raise ValueError(f"E5 {field} is invalid")
    for field in ("thermal_before", "thermal_after"):
        value = phase[field]
        if not isinstance(value, int) or not 0 <= value <= 6:
            raise ValueError(f"E5 {field} is invalid")
    for field in ("memory_before_bytes", "memory_after_bytes"):
        value = phase[field]
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"E5 {field} is invalid")
    if (
        not isinstance(phase["failure_count"], int)
        or int(phase["failure_count"]) < 0
    ):
        raise ValueError("E5 failure_count is invalid")


def verify_r2e5_run(
    path: Path,
    *,
    expected_bundle_id: str | None = None,
    expected_e3_receipt_id: str | None = None,
    expected_tuning_id: str | None = None,
) -> dict[str, object]:
    payload = _strict_json(path, label="E5 device run")
    if payload.get("schema") != R2E5_RUN_SCHEMA:
        raise ValueError("E5 run schema mismatch")
    run_id = _require_sha256(
        payload.get("run_id"),
        label="E5 run ID",
    )
    body = dict(payload)
    body.pop("run_id", None)
    if run_id != _sha256_bytes(
        b"VN97R2E5RUN1\0" + _canonical_json(body)
    ):
        raise ValueError("E5 run identity mismatch")

    bundle_id = _require_sha256(
        payload.get("bundle_id"),
        label="E5 bundle ID",
    )
    e3_id = _require_sha256(
        payload.get("e3_receipt_id"),
        label="E5 E3 receipt ID",
    )
    tuning_id = _require_sha256(
        payload.get("tuning_id"),
        label="E5 tuning ID",
    )
    if expected_bundle_id is not None and bundle_id != expected_bundle_id:
        raise ValueError("E5 run belongs to another E2 bundle")
    if (
        expected_e3_receipt_id is not None
        and e3_id != expected_e3_receipt_id
    ):
        raise ValueError("E5 run belongs to another E3 receipt")
    if (
        expected_tuning_id is not None
        and tuning_id != expected_tuning_id
    ):
        raise ValueError("E5 run belongs to another E4 tuning profile")

    device = payload.get("device")
    if not isinstance(device, dict):
        raise ValueError("E5 device evidence is missing")
    _validate_device(device)

    if payload.get("device_measured") is not True:
        raise ValueError("E5 run is not device-measured")
    if payload.get("synthetic") is not False:
        raise ValueError("E5 run must not be synthetic")

    phases = payload.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("E5 phases are missing")
    names: set[str] = set()
    kinds: set[str] = set()
    for phase in phases:
        if not isinstance(phase, dict):
            raise ValueError("E5 phase must be an object")
        _validate_phase(phase)
        name = str(phase["name"])
        if name in names:
            raise ValueError("E5 phase names must be unique")
        names.add(name)
        kinds.add(str(phase["kind"]))
    required_kinds = {
        "cold",
        "warm_step",
        "prefill",
        "sustained",
        "recovery",
    }
    if not required_kinds.issubset(kinds):
        raise ValueError("E5 run is missing required hardening phases")

    controls = payload.get("control_tests")
    if not isinstance(controls, dict):
        raise ValueError("E5 control tests are missing")
    required_controls = {
        "thermal_hysteresis_passed",
        "memory_pressure_passed",
        "provider_quarantine_passed",
        "provider_recovery_passed",
        "cpu_fallback_passed",
    }
    if set(controls) != required_controls:
        raise ValueError("E5 control test fields mismatch")
    if any(not isinstance(value, bool) for value in controls.values()):
        raise ValueError("E5 control tests must be booleans")
    return payload


def _phase_by_kind(
    run: Mapping[str, object],
    kind: str,
) -> Mapping[str, object]:
    phases = run["phases"]
    assert isinstance(phases, list)
    matches = [
        phase
        for phase in phases
        if isinstance(phase, dict) and phase.get("kind") == kind
    ]
    if len(matches) != 1:
        raise ValueError(
            f"E5 requires exactly one canonical {kind} phase"
        )
    return matches[0]


def _minimal_graph_set(
    tuning: Mapping[str, object],
) -> list[str]:
    policies = tuning.get("graph_policies")
    preferred = tuning.get("preferred_chunk_sizes")
    if (
        not isinstance(policies, list)
        or not isinstance(preferred, list)
        or not preferred
    ):
        raise ValueError("E5 tuning graph policy is invalid")
    chunk_sizes = sorted(
        int(item)
        for item in preferred
    )
    fastest = int(preferred[0])
    smallest = chunk_sizes[0]
    names = ["step.onnx", f"chunk-{fastest}.onnx"]
    if smallest != fastest:
        names.append(f"chunk-{smallest}.onnx")
    available = {
        str(item["graph_filename"])
        for item in policies
        if isinstance(item, dict)
    }
    if any(name not in available for name in names):
        raise ValueError("E5 minimal graph set is not present in E4")
    return names


def compile_r2e5_hardening(
    manifest: Mapping[str, object],
    e3: Mapping[str, object],
    e4: Mapping[str, object],
    run: Mapping[str, object],
    *,
    require_s21_fe: bool = True,
) -> dict[str, object]:
    bundle_id = _require_sha256(
        manifest.get("bundle_id"),
        label="E5 bundle ID",
    )
    if e3.get("bundle_id") != bundle_id:
        raise ValueError("E5 E3 receipt belongs to another E2 bundle")
    if e4.get("bundle_id") != bundle_id:
        raise ValueError("E5 E4 profile belongs to another E2 bundle")
    if run.get("bundle_id") != bundle_id:
        raise ValueError("E5 run belongs to another E2 bundle")
    if e4.get("e3_receipt_id") != e3.get("receipt_id"):
        raise ValueError("E5 E4 profile is not bound to supplied E3")
    if run.get("e3_receipt_id") != e3.get("receipt_id"):
        raise ValueError("E5 run is not bound to supplied E3")
    if run.get("tuning_id") != e4.get("tuning_id"):
        raise ValueError("E5 run is not bound to supplied E4 tuning")

    device = run.get("device")
    if not isinstance(device, dict):
        raise ValueError("E5 run device evidence is missing")
    target_match = is_s21_fe_device(device)
    if require_s21_fe and not target_match:
        raise ValueError("E5 run is not from a Galaxy S21 FE device")

    controls = run.get("control_tests")
    if not isinstance(controls, dict):
        raise ValueError("E5 run control evidence is missing")
    control_passed = all(bool(value) for value in controls.values())

    warm = _phase_by_kind(run, "warm_step")
    sustained = _phase_by_kind(run, "sustained")
    recovery = _phase_by_kind(run, "recovery")
    warm_p95 = percentile_nearest_rank_int(
        warm["latencies_ns"],
        0.95,
    )
    sustained_p95 = percentile_nearest_rank_int(
        sustained["latencies_ns"],
        0.95,
    )
    warm_per_token_p95 = (
        warm_p95 // int(warm["tokens_per_iteration"])
    )
    sustained_per_token_p95 = (
        sustained_p95 // int(sustained["tokens_per_iteration"])
    )
    if warm_per_token_p95 <= 0 or sustained_per_token_p95 <= 0:
        raise ValueError("E5 normalized latency evidence is invalid")
    sustained_ratio_ppm = (
        sustained_per_token_p95 * 1_000_000 // warm_per_token_p95
    )
    sustained_guard_passed = (
        sustained_ratio_ppm
        <= R2E5_MAX_SUSTAINED_P95_OVER_WARM_PPM
    )
    phases = run.get("phases")
    if not isinstance(phases, list):
        raise ValueError("E5 run phases are missing")
    failure_count = sum(
        int(phase["failure_count"])
        for phase in phases
        if isinstance(phase, dict)
    )
    failure_guard_passed = failure_count == 0
    recovery_thermal_passed = (
        int(recovery["thermal_after"])
        <= int(sustained["thermal_after"])
    )
    hardening_passed = (
        target_match
        and control_passed
        and sustained_guard_passed
        and failure_guard_passed
        and recovery_thermal_passed
    )

    body = {
        "schema": R2E5_HARDENING_SCHEMA,
        "target_family": R2E5_TARGET_FAMILY,
        "target_device_match": target_match,
        "bundle_id": bundle_id,
        "architecture_fingerprint": _require_sha256(
            manifest.get("architecture_fingerprint"),
            label="E5 architecture fingerprint",
        ),
        "e3_receipt_id": _require_sha256(
            e3.get("receipt_id"),
            label="E5 E3 receipt ID",
        ),
        "e4_tuning_id": _require_sha256(
            e4.get("tuning_id"),
            label="E5 E4 tuning ID",
        ),
        "run_id": _require_sha256(
            run.get("run_id"),
            label="E5 run ID",
        ),
        "device": device,
        "warm_step_p95_ns": warm_p95,
        "sustained_p95_ns": sustained_p95,
        "warm_step_per_token_p95_ns": warm_per_token_p95,
        "sustained_per_token_p95_ns": sustained_per_token_p95,
        "sustained_p95_over_warm_ppm": sustained_ratio_ppm,
        "max_sustained_p95_over_warm_ppm": (
            R2E5_MAX_SUSTAINED_P95_OVER_WARM_PPM
        ),
        "control_tests_passed": control_passed,
        "sustained_guard_passed": sustained_guard_passed,
        "failure_guard_passed": failure_guard_passed,
        "recovery_thermal_passed": recovery_thermal_passed,
        "failure_count": failure_count,
        "minimal_apk_graphs": _minimal_graph_set(e4),
        "hardening_passed": hardening_passed,
        "device_measured": True,
        "synthetic": False,
        "quantization_used": False,
        "same_weights_semantics": True,
    }
    sealed = dict(body)
    sealed["hardening_id"] = _sha256_bytes(
        b"VN97R2E5HARDEN1\0" + _canonical_json(body)
    )
    return sealed


def seal_r2e5_hardening(
    *,
    bundle_dir: Path,
    e3_receipt_path: Path,
    e4_profile_path: Path,
    run_path: Path,
    output_path: Path,
    require_s21_fe: bool = True,
) -> dict[str, object]:
    manifest = verify_r2_onnx_bundle(bundle_dir)
    bundle_id = str(manifest["bundle_id"])
    e3 = verify_r2e3_receipt(
        e3_receipt_path,
        expected_bundle_id=bundle_id,
    )
    e4 = verify_r2e4_profile(
        e4_profile_path,
        expected_bundle_id=bundle_id,
    )
    run = verify_r2e5_run(
        run_path,
        expected_bundle_id=bundle_id,
        expected_e3_receipt_id=str(e3["receipt_id"]),
        expected_tuning_id=str(e4["tuning_id"]),
    )
    sealed = compile_r2e5_hardening(
        manifest,
        e3,
        e4,
        run,
        require_s21_fe=require_s21_fe,
    )

    if output_path.exists() or output_path.is_symlink():
        raise ValueError("E5 hardening output must not already exist")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_name(output_path.name + ".tmp")
    temp.write_bytes(_canonical_json(sealed) + b"\n")
    temp.replace(output_path)
    verify_r2e5_hardening(
        output_path,
        expected_bundle_id=bundle_id,
    )
    return sealed


def verify_r2e5_hardening(
    path: Path,
    *,
    expected_bundle_id: str | None = None,
) -> dict[str, object]:
    payload = _strict_json(path, label="E5 hardening seal")
    if payload.get("schema") != R2E5_HARDENING_SCHEMA:
        raise ValueError("E5 hardening schema mismatch")
    hardening_id = _require_sha256(
        payload.get("hardening_id"),
        label="E5 hardening ID",
    )
    body = dict(payload)
    body.pop("hardening_id", None)
    if hardening_id != _sha256_bytes(
        b"VN97R2E5HARDEN1\0" + _canonical_json(body)
    ):
        raise ValueError("E5 hardening identity mismatch")
    bundle_id = _require_sha256(
        payload.get("bundle_id"),
        label="E5 bundle ID",
    )
    if expected_bundle_id is not None and bundle_id != expected_bundle_id:
        raise ValueError("E5 hardening belongs to another E2 bundle")
    if payload.get("target_family") != R2E5_TARGET_FAMILY:
        raise ValueError("E5 target family mismatch")
    if payload.get("device_measured") is not True:
        raise ValueError("E5 hardening is not device-measured")
    if payload.get("synthetic") is not False:
        raise ValueError("E5 hardening must not be synthetic")
    if payload.get("quantization_used") is not False:
        raise ValueError("E5 must not introduce quantization")
    if payload.get("same_weights_semantics") is not True:
        raise ValueError("E5 same-weights semantics are not locked")
    graphs = payload.get("minimal_apk_graphs")
    if (
        not isinstance(graphs, list)
        or not graphs
        or graphs[0] != "step.onnx"
        or len(graphs) != len(set(graphs))
    ):
        raise ValueError("E5 minimal APK graph set is invalid")
    return payload
