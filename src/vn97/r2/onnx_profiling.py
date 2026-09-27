from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence


R2E3_RECEIPT_SCHEMA = "VN97R2E3PROFILE1"
R2E3_ALLOWED_PROVIDERS = ("QNN", "NNAPI", "XNNPACK", "CPU")
R2E3_ALLOWED_KINDS = ("step", "chunk")


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


def percentile_nearest_rank(
    values: Sequence[int],
    q: float,
) -> int:
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 < q <= 1.0:
        raise ValueError("q must be in (0, 1]")
    ordered = sorted(int(value) for value in values)
    if any(value <= 0 for value in ordered):
        raise ValueError("latencies must be positive")
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[rank - 1]


@dataclass(frozen=True)
class R2E3Measurement:
    graph_filename: str
    kind: str
    sequence_length: int
    requested_provider: str
    actual_provider: str
    warmup_iterations: int
    steady_iterations: int
    session_create_ns: int
    steady_latencies_ns: tuple[int, ...]
    memory_before_bytes: int
    memory_after_bytes: int
    thermal_before: int
    thermal_after: int
    fallback_used: bool = False

    def __post_init__(self) -> None:
        if self.kind not in R2E3_ALLOWED_KINDS:
            raise ValueError("invalid graph kind")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.requested_provider not in R2E3_ALLOWED_PROVIDERS:
            raise ValueError("invalid requested provider")
        if self.actual_provider not in R2E3_ALLOWED_PROVIDERS:
            raise ValueError("invalid actual provider")
        if self.warmup_iterations < 1 or self.steady_iterations < 3:
            raise ValueError("profiling requires warmup>=1 and steady>=3")
        if len(self.steady_latencies_ns) != self.steady_iterations:
            raise ValueError("steady latency count mismatch")
        if self.session_create_ns <= 0:
            raise ValueError("session_create_ns must be positive")
        if self.memory_before_bytes <= 0 or self.memory_after_bytes <= 0:
            raise ValueError("memory readings must be positive")
        if not 0 <= self.thermal_before <= 6:
            raise ValueError("thermal_before must be in [0, 6]")
        if not 0 <= self.thermal_after <= 6:
            raise ValueError("thermal_after must be in [0, 6]")
        # A provider substitution is fallback evidence regardless of caller flag.
        if (
            self.requested_provider != self.actual_provider
            and not self.fallback_used
        ):
            raise ValueError("provider substitution must mark fallback_used")
        for value in self.steady_latencies_ns:
            if value <= 0:
                raise ValueError("steady latencies must be positive")

    def summary(self) -> dict[str, object]:
        p50 = percentile_nearest_rank(
            self.steady_latencies_ns,
            0.50,
        )
        p95 = percentile_nearest_rank(
            self.steady_latencies_ns,
            0.95,
        )
        tokens_per_second_milli = (
            self.sequence_length * 1_000_000_000_000 // p50
        )
        return {
            "graph_filename": self.graph_filename,
            "kind": self.kind,
            "sequence_length": self.sequence_length,
            "requested_provider": self.requested_provider,
            "actual_provider": self.actual_provider,
            "fallback_used": self.fallback_used,
            "warmup_iterations": self.warmup_iterations,
            "steady_iterations": self.steady_iterations,
            "session_create_ns": self.session_create_ns,
            "latency_p50_ns": p50,
            "latency_p95_ns": p95,
            "latency_min_ns": min(self.steady_latencies_ns),
            "latency_max_ns": max(self.steady_latencies_ns),
            "tokens_per_second_milli_p50": (
                tokens_per_second_milli
            ),
            "memory_before_bytes": self.memory_before_bytes,
            "memory_after_bytes": self.memory_after_bytes,
            "memory_delta_bytes": (
                self.memory_after_bytes - self.memory_before_bytes
            ),
            "thermal_before": self.thermal_before,
            "thermal_after": self.thermal_after,
        }


def rank_measurements(
    measurements: Iterable[R2E3Measurement],
) -> tuple[dict[str, object], ...]:
    summaries = [item.summary() for item in measurements]
    if not summaries:
        raise ValueError("at least one successful measurement is required")
    # Prefer no fallback, then lower steady-state p50, then lower p95.
    summaries.sort(
        key=lambda item: (
            bool(item["fallback_used"]),
            int(item["latency_p50_ns"]),
            int(item["latency_p95_ns"]),
            str(item["requested_provider"]),
        )
    )
    return tuple(summaries)


def build_r2e3_receipt(
    *,
    bundle_id: str,
    architecture_fingerprint: str,
    profile: str,
    device: Mapping[str, object],
    measurements: Sequence[R2E3Measurement],
    failures: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    _require_sha256(bundle_id, label="bundle_id")
    _require_sha256(
        architecture_fingerprint,
        label="architecture_fingerprint",
    )
    if profile not in {"fast", "deep"} and not (
        profile.startswith("layers-") and profile[7:].isdigit()
    ):
        raise ValueError("invalid ONNX profile")
    required_device = {
        "sdk_int",
        "logical_cores",
        "available_memory_bytes",
        "thermal_status",
        "hardware",
        "soc_manufacturer",
        "soc_model",
    }
    if set(device) != required_device:
        raise ValueError("device evidence fields mismatch")

    groups: dict[str, list[R2E3Measurement]] = {}
    for measurement in measurements:
        groups.setdefault(
            measurement.graph_filename,
            [],
        ).append(measurement)
    if not groups:
        raise ValueError("receipt requires successful measurements")

    graph_results = []
    for filename in sorted(groups):
        ranked = rank_measurements(groups[filename])
        graph_results.append(
            {
                "graph_filename": filename,
                "ranked_providers": list(ranked),
                "recommended_provider": ranked[0][
                    "requested_provider"
                ],
            }
        )

    normalized_failures = []
    for raw in failures:
        if set(raw) != {
            "graph_filename",
            "requested_provider",
            "error_class",
        }:
            raise ValueError("failure evidence fields mismatch")
        if raw["requested_provider"] not in R2E3_ALLOWED_PROVIDERS:
            raise ValueError("failure provider is invalid")
        normalized_failures.append(dict(raw))
    normalized_failures.sort(
        key=lambda item: (
            str(item["graph_filename"]),
            str(item["requested_provider"]),
            str(item["error_class"]),
        )
    )

    body = {
        "schema": R2E3_RECEIPT_SCHEMA,
        "bundle_id": bundle_id,
        "architecture_fingerprint": architecture_fingerprint,
        "profile": profile,
        "device": dict(device),
        "graph_results": graph_results,
        "failures": normalized_failures,
        "device_measured": True,
        "synthetic": False,
    }
    receipt = dict(body)
    receipt["receipt_id"] = _sha256_bytes(
        b"VN97R2E3PROFILE1\0" + _canonical_json(body)
    )
    return receipt


def verify_r2e3_receipt(
    path: Path,
    *,
    expected_bundle_id: str | None = None,
) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("E3 receipt must be a regular non-symlink file")
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
        raise ValueError("E3 receipt must be strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("E3 receipt must be an object")
    if payload.get("schema") != R2E3_RECEIPT_SCHEMA:
        raise ValueError("E3 receipt schema mismatch")
    receipt_id = _require_sha256(
        payload.get("receipt_id"),
        label="receipt_id",
    )
    body = dict(payload)
    body.pop("receipt_id", None)
    if receipt_id != _sha256_bytes(
        b"VN97R2E3PROFILE1\0" + _canonical_json(body)
    ):
        raise ValueError("E3 receipt identity mismatch")
    bundle_id = _require_sha256(
        payload.get("bundle_id"),
        label="bundle_id",
    )
    if expected_bundle_id is not None and bundle_id != expected_bundle_id:
        raise ValueError("E3 receipt belongs to another ONNX bundle")
    if payload.get("device_measured") is not True:
        raise ValueError("E3 receipt is not device-measured")
    if payload.get("synthetic") is not False:
        raise ValueError("E3 receipt must not be synthetic")
    results = payload.get("graph_results")
    if not isinstance(results, list) or not results:
        raise ValueError("E3 graph results are missing")
    for graph in results:
        if not isinstance(graph, dict):
            raise ValueError("E3 graph result is invalid")
        ranked = graph.get("ranked_providers")
        if not isinstance(ranked, list) or not ranked:
            raise ValueError("E3 provider ranking is missing")
        if graph.get("recommended_provider") != ranked[0].get(
            "requested_provider"
        ):
            raise ValueError("E3 recommendation/ranking mismatch")
    return payload
