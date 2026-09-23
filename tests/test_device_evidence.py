import hashlib
import json

import pytest

from vn97 import (
    VN97DeviceEvidenceCriteria,
    VN97DeviceEvidenceError,
    parse_device_evidence,
    require_device_evidence,
)


def _blob(*, model_sha="a" * 64, speech=True):
    value = {
        "battery_energy_counter_delta_nwh": 1234,
        "device": {
            "abi": "arm64-v8a",
            "manufacturer": "VN97",
            "model": "MobileFixture",
            "sdk_int": 37,
        },
        "model_image_sha256": model_sha,
        "peak_pss_kib": 256000,
        "runs": 7,
        "schema": "VN97MOBEVID1",
        "speech_prefill": (
            {"p50_ms": 42.0, "p95_ms": 55.0}
            if speech
            else None
        ),
        "text_decode_per_token": {
            "p50_ms": 8.0,
            "p95_ms": 10.0,
        },
        "text_prefill": {
            "p50_ms": 20.0,
            "p95_ms": 25.0,
        },
        "thermal_status_max": 2,
    }
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def test_mobile_evidence_parses_and_binds_model_identity():
    blob = _blob()
    evidence = parse_device_evidence(blob)
    assert evidence.model_image_sha256 == "a" * 64
    assert evidence.evidence_sha256 == hashlib.sha256(blob).hexdigest()
    assert evidence.text_prefill.p95_ms == 25.0
    assert evidence.text_decode_per_token.p95_ms == 10.0
    assert evidence.speech_prefill is not None

    require_device_evidence(
        evidence,
        VN97DeviceEvidenceCriteria(
            min_runs=5,
            max_text_prefill_p95_ms=30.0,
            max_text_decode_p95_ms_per_token=12.0,
            max_peak_pss_kib=300000,
            max_thermal_status=3,
            max_speech_prefill_p95_ms=60.0,
            require_energy_counter=True,
            max_abs_battery_energy_counter_delta_nwh=2000,
        ),
        expected_model_image_sha256="a" * 64,
        speech_enabled=True,
    )


def test_mobile_evidence_gate_rejects_wrong_model_and_missing_speech():
    evidence = parse_device_evidence(_blob(speech=False))
    with pytest.raises(VN97DeviceEvidenceError, match="model image identity"):
        require_device_evidence(
            evidence,
            VN97DeviceEvidenceCriteria(),
            expected_model_image_sha256="b" * 64,
            speech_enabled=False,
        )

    with pytest.raises(VN97DeviceEvidenceError, match="speech prefill"):
        require_device_evidence(
            evidence,
            VN97DeviceEvidenceCriteria(),
            expected_model_image_sha256="a" * 64,
            speech_enabled=True,
        )


def test_mobile_evidence_requires_canonical_json_and_limits():
    raw = json.dumps(
        json.loads(_blob()),
        indent=2,
    ).encode("utf-8")
    with pytest.raises(VN97DeviceEvidenceError, match="canonical"):
        parse_device_evidence(raw)

    evidence = parse_device_evidence(_blob())
    with pytest.raises(VN97DeviceEvidenceError, match="decode/token"):
        require_device_evidence(
            evidence,
            VN97DeviceEvidenceCriteria(
                max_text_decode_p95_ms_per_token=5.0,
            ),
            expected_model_image_sha256="a" * 64,
            speech_enabled=True,
        )
