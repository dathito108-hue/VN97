from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any


class VN97DeviceEvidenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class VN97LatencyPercentiles:
    p50_ms: float
    p95_ms: float

    def __post_init__(self) -> None:
        for name, value in (("p50_ms", self.p50_ms), ("p95_ms", self.p95_ms)):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.p95_ms < self.p50_ms:
            raise ValueError("p95_ms must be >= p50_ms")


@dataclass(frozen=True)
class VN97DeviceEvidence:
    model_image_sha256: str
    manufacturer: str
    model: str
    sdk_int: int
    abi: str
    runs: int
    text_prefill: VN97LatencyPercentiles
    text_decode_per_token: VN97LatencyPercentiles
    speech_prefill: VN97LatencyPercentiles | None
    peak_pss_kib: int
    thermal_status_max: int
    battery_energy_counter_delta_nwh: int | None
    evidence_sha256: str

    def __post_init__(self) -> None:
        if (
            len(self.model_image_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in self.model_image_sha256)
        ):
            raise ValueError("model_image_sha256 must be lowercase SHA-256")
        if not self.manufacturer or not self.model or not self.abi:
            raise ValueError("device identity strings must be non-empty")
        if self.sdk_int <= 0 or self.runs <= 0:
            raise ValueError("sdk_int and runs must be positive")
        if self.peak_pss_kib <= 0:
            raise ValueError("peak_pss_kib must be positive")
        if not 0 <= self.thermal_status_max <= 6:
            raise ValueError("thermal_status_max must be in [0, 6]")
        if (
            len(self.evidence_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in self.evidence_sha256)
        ):
            raise ValueError("evidence_sha256 must be lowercase SHA-256")


@dataclass(frozen=True)
class VN97DeviceEvidenceCriteria:
    min_runs: int = 5
    max_text_prefill_p95_ms: float = 10_000.0
    max_text_decode_p95_ms_per_token: float = 2_000.0
    max_peak_pss_kib: int = 2 * 1024 * 1024
    max_thermal_status: int = 5
    max_speech_prefill_p95_ms: float | None = None
    require_energy_counter: bool = False
    max_abs_battery_energy_counter_delta_nwh: int | None = None

    def __post_init__(self) -> None:
        if self.min_runs <= 0:
            raise ValueError("min_runs must be positive")
        for name, value in (
            ("max_text_prefill_p95_ms", self.max_text_prefill_p95_ms),
            (
                "max_text_decode_p95_ms_per_token",
                self.max_text_decode_p95_ms_per_token,
            ),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_peak_pss_kib <= 0:
            raise ValueError("max_peak_pss_kib must be positive")
        if not 0 <= self.max_thermal_status <= 6:
            raise ValueError("max_thermal_status must be in [0, 6]")
        if self.max_speech_prefill_p95_ms is not None and (
            not math.isfinite(self.max_speech_prefill_p95_ms)
            or self.max_speech_prefill_p95_ms <= 0.0
        ):
            raise ValueError(
                "max_speech_prefill_p95_ms must be finite and positive"
            )
        if (
            self.max_abs_battery_energy_counter_delta_nwh is not None
            and self.max_abs_battery_energy_counter_delta_nwh <= 0
        ):
            raise ValueError(
                "max_abs_battery_energy_counter_delta_nwh must be positive"
            )


def _strict_object(data: bytes) -> dict[str, Any]:
    duplicates: list[str] = []

    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                duplicates.append(key)
            output[key] = value
        return output

    try:
        text = data.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=hook,
            parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise VN97DeviceEvidenceError(
            "device evidence must be strict UTF-8 JSON"
        ) from exc
    if duplicates or not isinstance(value, dict):
        raise VN97DeviceEvidenceError(
            "device evidence must be one object without duplicate keys"
        )
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if canonical != data:
        raise VN97DeviceEvidenceError(
            "device evidence must use canonical JSON"
        )
    return value


def _latency(raw: object, label: str) -> VN97LatencyPercentiles:
    if (
        not isinstance(raw, dict)
        or set(raw) != {"p50_ms", "p95_ms"}
    ):
        raise VN97DeviceEvidenceError(f"{label} latency keys are invalid")
    values: dict[str, float] = {}
    for key in ("p50_ms", "p95_ms"):
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise VN97DeviceEvidenceError(f"{label} {key} must be numeric")
        values[key] = float(value)
    try:
        return VN97LatencyPercentiles(**values)
    except ValueError as exc:
        raise VN97DeviceEvidenceError(str(exc)) from exc


def parse_device_evidence(data: bytes) -> VN97DeviceEvidence:
    root = _strict_object(data)
    expected = {
        "battery_energy_counter_delta_nwh",
        "device",
        "model_image_sha256",
        "peak_pss_kib",
        "runs",
        "schema",
        "speech_prefill",
        "text_decode_per_token",
        "text_prefill",
        "thermal_status_max",
    }
    if set(root) != expected or root["schema"] != "VN97MOBEVID1":
        raise VN97DeviceEvidenceError(
            "device evidence schema/keys are invalid"
        )

    device = root["device"]
    if (
        not isinstance(device, dict)
        or set(device) != {"abi", "manufacturer", "model", "sdk_int"}
    ):
        raise VN97DeviceEvidenceError("device identity keys are invalid")
    for key in ("abi", "manufacturer", "model"):
        if not isinstance(device[key], str) or not device[key]:
            raise VN97DeviceEvidenceError(
                f"device {key} must be non-empty text"
            )
    for key in ("sdk_int",):
        if type(device[key]) is not int:
            raise VN97DeviceEvidenceError(f"device {key} must be integer")

    for key in ("runs", "peak_pss_kib", "thermal_status_max"):
        if type(root[key]) is not int:
            raise VN97DeviceEvidenceError(f"{key} must be integer")

    energy = root["battery_energy_counter_delta_nwh"]
    if energy is not None and type(energy) is not int:
        raise VN97DeviceEvidenceError(
            "battery_energy_counter_delta_nwh must be integer or null"
        )

    speech = (
        None
        if root["speech_prefill"] is None
        else _latency(root["speech_prefill"], "speech_prefill")
    )
    digest = hashlib.sha256(data).hexdigest()
    try:
        return VN97DeviceEvidence(
            model_image_sha256=root["model_image_sha256"],
            manufacturer=device["manufacturer"],
            model=device["model"],
            sdk_int=device["sdk_int"],
            abi=device["abi"],
            runs=root["runs"],
            text_prefill=_latency(root["text_prefill"], "text_prefill"),
            text_decode_per_token=_latency(
                root["text_decode_per_token"],
                "text_decode_per_token",
            ),
            speech_prefill=speech,
            peak_pss_kib=root["peak_pss_kib"],
            thermal_status_max=root["thermal_status_max"],
            battery_energy_counter_delta_nwh=energy,
            evidence_sha256=digest,
        )
    except (TypeError, ValueError) as exc:
        raise VN97DeviceEvidenceError(str(exc)) from exc


def load_device_evidence(path: str | Path) -> VN97DeviceEvidence:
    target = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(target, flags)
    except OSError as exc:
        raise VN97DeviceEvidenceError(
            "device evidence could not be opened safely"
        ) from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise VN97DeviceEvidenceError(
                "device evidence must be a regular file"
            )
        if not 0 < info.st_size <= 1024 * 1024:
            raise VN97DeviceEvidenceError(
                "device evidence byte size is outside bounds"
            )
        out = bytearray()
        while len(out) < info.st_size:
            chunk = os.read(
                fd,
                min(256 * 1024, info.st_size - len(out)),
            )
            if not chunk:
                break
            out.extend(chunk)
        after = os.fstat(fd)
        if (
            len(out) != info.st_size
            or after.st_size != info.st_size
            or after.st_ino != info.st_ino
            or after.st_dev != info.st_dev
        ):
            raise VN97DeviceEvidenceError(
                "device evidence changed while being read"
            )
        return parse_device_evidence(bytes(out))
    finally:
        os.close(fd)


def require_device_evidence(
    evidence: VN97DeviceEvidence,
    criteria: VN97DeviceEvidenceCriteria,
    *,
    expected_model_image_sha256: str,
    speech_enabled: bool,
) -> None:
    failures: list[str] = []
    if evidence.model_image_sha256 != expected_model_image_sha256:
        failures.append("model image identity mismatch")
    if evidence.runs < criteria.min_runs:
        failures.append(
            f"runs {evidence.runs} < {criteria.min_runs}"
        )
    if evidence.text_prefill.p95_ms > criteria.max_text_prefill_p95_ms:
        failures.append(
            "text prefill p95 exceeds mobile evidence limit"
        )
    if (
        evidence.text_decode_per_token.p95_ms
        > criteria.max_text_decode_p95_ms_per_token
    ):
        failures.append(
            "text decode/token p95 exceeds mobile evidence limit"
        )
    if evidence.peak_pss_kib > criteria.max_peak_pss_kib:
        failures.append("peak PSS exceeds mobile evidence limit")
    if evidence.thermal_status_max > criteria.max_thermal_status:
        failures.append("thermal status exceeds mobile evidence limit")

    if speech_enabled:
        if evidence.speech_prefill is None:
            failures.append(
                "speech-enabled release requires speech prefill evidence"
            )
        elif (
            criteria.max_speech_prefill_p95_ms is not None
            and evidence.speech_prefill.p95_ms
            > criteria.max_speech_prefill_p95_ms
        ):
            failures.append(
                "speech prefill p95 exceeds mobile evidence limit"
            )

    if criteria.require_energy_counter and (
        evidence.battery_energy_counter_delta_nwh is None
    ):
        failures.append("device battery energy counter evidence is unavailable")
    if (
        criteria.max_abs_battery_energy_counter_delta_nwh is not None
        and evidence.battery_energy_counter_delta_nwh is not None
        and abs(evidence.battery_energy_counter_delta_nwh)
        > criteria.max_abs_battery_energy_counter_delta_nwh
    ):
        failures.append(
            "battery energy counter delta exceeds mobile evidence limit"
        )

    if failures:
        raise VN97DeviceEvidenceError(
            "VN97 mobile evidence gate failed: " + "; ".join(failures)
        )
