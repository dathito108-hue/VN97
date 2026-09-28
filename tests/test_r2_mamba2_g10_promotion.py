import hashlib
import json

import pytest

from vn97.r2.mamba2_promotion import (
    VN97_MAMBA2_G10_MOBILE_QUAL_SCHEMA,
    VN97_MAMBA2_G10_MODEL_PARITY_SCHEMA,
    VN97_MAMBA2_G10_PROMOTION_SCHEMA,
    VN97_MAMBA2_G10_TOKEN_PARITY_SCHEMA,
    compile_g10_promotion,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _seal(schema: str, body: dict[str, object]) -> dict[str, object]:
    return {
        **body,
        "receipt_id": hashlib.sha256(
            schema.encode("ascii") + b"\0" + _canonical(body)
        ).hexdigest(),
    }


def _bridge() -> dict[str, object]:
    return {
        "schema": "VN97M2G09BRIDGE1",
        "runtime_id": "1" * 64,
        "g05_manifest_id": "2" * 64,
        "capsule_id": "3" * 64,
        "source_weight_sha256": "4" * 64,
        "tokenizer_id": "5" * 64,
        "tuning_id": "6" * 64,
        "device_profile_receipt_id": "7" * 64,
        "token_id_space": 50277,
        "runtime_logits_size": 50288,
        "eos_token_id": 0,
        "same_weights_semantics": True,
        "same_token_ids_required": True,
        "padded_logits_must_be_masked": True,
        "device_measured_tuning_required": True,
        "candidate_validation_only": True,
        "production_activation_authorized": False,
        "bridge_id": "8" * 64,
    }


def _model(bridge: dict[str, object]) -> dict[str, object]:
    return _seal(
        VN97_MAMBA2_G10_MODEL_PARITY_SCHEMA,
        {
            "schema": VN97_MAMBA2_G10_MODEL_PARITY_SCHEMA,
            "bridge_id": bridge["bridge_id"],
            "runtime_id": bridge["runtime_id"],
            "capsule_id": bridge["capsule_id"],
            "source_weight_sha256": bridge["source_weight_sha256"],
            "synthetic": False,
            "full_source_weights_verified": True,
            "source_to_vn97_passed": True,
            "vn97_to_ort_passed": True,
            "recurrent_state_passed": True,
            "max_abs_error": 2.0e-5,
            "state_max_abs_error": 1.0e-5,
            "passed": True,
        },
    )


def _token(bridge: dict[str, object]) -> dict[str, object]:
    return _seal(
        VN97_MAMBA2_G10_TOKEN_PARITY_SCHEMA,
        {
            "schema": VN97_MAMBA2_G10_TOKEN_PARITY_SCHEMA,
            "bridge_id": bridge["bridge_id"],
            "tokenizer_id": bridge["tokenizer_id"],
            "reference_corpus_sha256": "9" * 64,
            "cases": 256,
            "synthetic": False,
            "official_gpt_neox_reference": True,
            "token_ids_exact": True,
            "decode_bytes_exact": True,
            "passed": True,
        },
    )


def _mobile(bridge: dict[str, object]) -> dict[str, object]:
    return _seal(
        VN97_MAMBA2_G10_MOBILE_QUAL_SCHEMA,
        {
            "schema": VN97_MAMBA2_G10_MOBILE_QUAL_SCHEMA,
            "bridge_id": bridge["bridge_id"],
            "runtime_id": bridge["runtime_id"],
            "tuning_id": bridge["tuning_id"],
            "device_profile_receipt_id":
                bridge["device_profile_receipt_id"],
            "target_family": "Samsung Galaxy S21 FE",
            "device_measured": True,
            "synthetic": False,
            "zero_execution_failures": True,
            "memory_passed": True,
            "latency_passed": True,
            "thermal_recovery_passed": True,
            "passed": True,
        },
    )


def test_g10_promotes_only_one_exact_lineage() -> None:
    bridge = _bridge()
    promotion = compile_g10_promotion(
        bridge,
        _model(bridge),
        _token(bridge),
        _mobile(bridge),
    )
    assert promotion["schema"] == VN97_MAMBA2_G10_PROMOTION_SCHEMA
    assert promotion["bridge_id"] == bridge["bridge_id"]
    assert promotion["runtime_id"] == bridge["runtime_id"]
    assert promotion["tokenizer_id"] == bridge["tokenizer_id"]
    assert promotion["tuning_id"] == bridge["tuning_id"]
    assert promotion["production_activation_authorized"] is True
    assert promotion["rollback_required"] is True
    assert promotion["real_evidence_required"] is True
    assert len(promotion["promotion_id"]) == 64


@pytest.mark.parametrize(
    "mutator,match",
    [
        (
            lambda b, m, t, q: m.__setitem__("synthetic", True),
            "must be real",
        ),
        (
            lambda b, m, t, q: m.__setitem__(
                "source_to_vn97_passed",
                False,
            ),
            "source→VN97 parity",
        ),
        (
            lambda b, m, t, q: t.__setitem__("token_ids_exact", False),
            "token IDs",
        ),
        (
            lambda b, m, t, q: t.__setitem__("cases", 12),
            "at least 100",
        ),
        (
            lambda b, m, t, q: q.__setitem__(
                "target_family",
                "Other Device",
            ),
            "Galaxy S21 FE",
        ),
        (
            lambda b, m, t, q: q.__setitem__("device_measured", False),
            "device measured",
        ),
        (
            lambda b, m, t, q: q.__setitem__(
                "thermal_recovery_passed",
                False,
            ),
            "thermal recovery",
        ),
    ],
)
def test_g10_rejects_nonproduction_evidence(mutator, match: str) -> None:
    bridge = _bridge()
    model = _model(bridge)
    token = _token(bridge)
    mobile = _mobile(bridge)
    mutator(bridge, model, token, mobile)

    # Reseal mutated evidence only when its content was intentionally changed;
    # the gate must reject the semantic defect, not merely stale receipt hash.
    if model.get("schema") == VN97_MAMBA2_G10_MODEL_PARITY_SCHEMA:
        body = dict(model)
        body.pop("receipt_id")
        model = _seal(VN97_MAMBA2_G10_MODEL_PARITY_SCHEMA, body)
    if token.get("schema") == VN97_MAMBA2_G10_TOKEN_PARITY_SCHEMA:
        body = dict(token)
        body.pop("receipt_id")
        token = _seal(VN97_MAMBA2_G10_TOKEN_PARITY_SCHEMA, body)
    if mobile.get("schema") == VN97_MAMBA2_G10_MOBILE_QUAL_SCHEMA:
        body = dict(mobile)
        body.pop("receipt_id")
        mobile = _seal(VN97_MAMBA2_G10_MOBILE_QUAL_SCHEMA, body)

    with pytest.raises(ValueError, match=match):
        compile_g10_promotion(bridge, model, token, mobile)


def test_g10_rejects_lineage_mismatch() -> None:
    bridge = _bridge()
    model = _model(bridge)
    token = _token(bridge)
    mobile = _mobile(bridge)
    body = dict(mobile)
    body.pop("receipt_id")
    body["runtime_id"] = "a" * 64
    mobile = _seal(VN97_MAMBA2_G10_MOBILE_QUAL_SCHEMA, body)

    with pytest.raises(ValueError, match="runtime_id mismatch"):
        compile_g10_promotion(bridge, model, token, mobile)


def test_g10_rejects_tampered_receipt_identity() -> None:
    bridge = _bridge()
    model = _model(bridge)
    model["max_abs_error"] = 0.5

    with pytest.raises(ValueError, match="receipt identity mismatch"):
        compile_g10_promotion(
            bridge,
            model,
            _token(bridge),
            _mobile(bridge),
        )
