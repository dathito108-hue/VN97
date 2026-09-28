from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


VN97_MAMBA2_G10_MODEL_PARITY_SCHEMA = "VN97M2G10MODELPARITY1"
VN97_MAMBA2_G10_TOKEN_PARITY_SCHEMA = "VN97M2G10TOKENPARITY1"
VN97_MAMBA2_G10_MOBILE_QUAL_SCHEMA = "VN97M2G10MOBILEQUAL1"
VN97_MAMBA2_G10_PROMOTION_SCHEMA = "VN97M2G10PROMOTE1"
VN97_MAMBA2_G10_PROMOTION_FILENAME = "promotion.vn97m2g10.json"


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


def _require_finite_nonnegative(value: object, label: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    resolved = float(value)
    if not (resolved >= 0.0 and resolved < float("inf")):
        raise ValueError(f"{label} must be finite and nonnegative")
    return resolved


def _load_object(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be regular non-symlink file")
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be JSON object")
    return value


def _receipt_id(
    schema: str,
    receipt: Mapping[str, object],
    *,
    id_field: str,
) -> str:
    actual = _require_sha256(receipt.get(id_field), f"{schema} {id_field}")
    body = dict(receipt)
    body.pop(id_field, None)
    expected = _sha256_bytes(
        schema.encode("ascii") + b"\0" + _canonical_json(body)
    )
    if actual != expected:
        raise ValueError(f"{schema} receipt identity mismatch")
    return actual


def compile_g10_promotion(
    bridge: Mapping[str, object],
    model_parity: Mapping[str, object],
    token_parity: Mapping[str, object],
    mobile_qualification: Mapping[str, object],
) -> dict[str, object]:
    if bridge.get("schema") != "VN97M2G09BRIDGE1":
        raise ValueError("G0.10 requires G0.9 bridge binding")
    bridge_id = _require_sha256(bridge.get("bridge_id"), "G0.10 bridge ID")
    runtime_id = _require_sha256(bridge.get("runtime_id"), "G0.10 runtime ID")
    capsule_id = _require_sha256(bridge.get("capsule_id"), "G0.10 capsule ID")
    source_weight = _require_sha256(
        bridge.get("source_weight_sha256"),
        "G0.10 source weight SHA-256",
    )
    tokenizer_id = _require_sha256(
        bridge.get("tokenizer_id"),
        "G0.10 tokenizer ID",
    )
    tuning_id = _require_sha256(
        bridge.get("tuning_id"),
        "G0.10 tuning ID",
    )
    profile_receipt_id = _require_sha256(
        bridge.get("device_profile_receipt_id"),
        "G0.10 profile receipt ID",
    )
    if bridge.get("candidate_validation_only") is not True:
        raise ValueError("G0.10 expects a candidate-only G0.9 bridge")
    if bridge.get("production_activation_authorized") is not False:
        raise ValueError("G0.9 bridge must not self-authorize production")
    if bridge.get("same_weights_semantics") is not True:
        raise ValueError("G0.10 requires same-weight G0.9 semantics")
    if bridge.get("same_token_ids_required") is not True:
        raise ValueError("G0.10 requires exact source token IDs")

    if model_parity.get("schema") != VN97_MAMBA2_G10_MODEL_PARITY_SCHEMA:
        raise ValueError("G0.10 model parity receipt schema mismatch")
    model_receipt_id = _receipt_id(
        VN97_MAMBA2_G10_MODEL_PARITY_SCHEMA,
        model_parity,
        id_field="receipt_id",
    )
    for field, expected in (
        ("bridge_id", bridge_id),
        ("runtime_id", runtime_id),
        ("capsule_id", capsule_id),
        ("source_weight_sha256", source_weight),
    ):
        if model_parity.get(field) != expected:
            raise ValueError(f"G0.10 model parity {field} mismatch")
    if model_parity.get("synthetic") is not False:
        raise ValueError("G0.10 model parity must be real, not synthetic")
    if model_parity.get("full_source_weights_verified") is not True:
        raise ValueError("G0.10 requires full source-weight verification")
    if model_parity.get("source_to_vn97_passed") is not True:
        raise ValueError("G0.10 source→VN97 parity did not pass")
    if model_parity.get("vn97_to_ort_passed") is not True:
        raise ValueError("G0.10 VN97→ORT parity did not pass")
    if model_parity.get("recurrent_state_passed") is not True:
        raise ValueError("G0.10 recurrent-state parity did not pass")
    if model_parity.get("passed") is not True:
        raise ValueError("G0.10 model parity receipt is not passing")
    model_max_abs = _require_finite_nonnegative(
        model_parity.get("max_abs_error"),
        "G0.10 model parity max_abs_error",
    )
    state_max_abs = _require_finite_nonnegative(
        model_parity.get("state_max_abs_error"),
        "G0.10 model parity state_max_abs_error",
    )

    if token_parity.get("schema") != VN97_MAMBA2_G10_TOKEN_PARITY_SCHEMA:
        raise ValueError("G0.10 token parity receipt schema mismatch")
    token_receipt_id = _receipt_id(
        VN97_MAMBA2_G10_TOKEN_PARITY_SCHEMA,
        token_parity,
        id_field="receipt_id",
    )
    if token_parity.get("bridge_id") != bridge_id:
        raise ValueError("G0.10 token parity bridge mismatch")
    if token_parity.get("tokenizer_id") != tokenizer_id:
        raise ValueError("G0.10 token parity tokenizer mismatch")
    _require_sha256(
        token_parity.get("reference_corpus_sha256"),
        "G0.10 token parity corpus SHA-256",
    )
    cases = token_parity.get("cases")
    if not isinstance(cases, int) or cases < 100:
        raise ValueError("G0.10 token parity requires at least 100 cases")
    if token_parity.get("synthetic") is not False:
        raise ValueError("G0.10 token parity must use the official reference")
    if token_parity.get("official_gpt_neox_reference") is not True:
        raise ValueError("G0.10 requires official GPT-NeoX tokenizer reference")
    if token_parity.get("token_ids_exact") is not True:
        raise ValueError("G0.10 token IDs are not exact")
    if token_parity.get("decode_bytes_exact") is not True:
        raise ValueError("G0.10 tokenizer decode bytes are not exact")
    if token_parity.get("passed") is not True:
        raise ValueError("G0.10 token parity receipt is not passing")

    if mobile_qualification.get("schema") != VN97_MAMBA2_G10_MOBILE_QUAL_SCHEMA:
        raise ValueError("G0.10 mobile qualification schema mismatch")
    mobile_receipt_id = _receipt_id(
        VN97_MAMBA2_G10_MOBILE_QUAL_SCHEMA,
        mobile_qualification,
        id_field="receipt_id",
    )
    for field, expected in (
        ("bridge_id", bridge_id),
        ("runtime_id", runtime_id),
        ("tuning_id", tuning_id),
        ("device_profile_receipt_id", profile_receipt_id),
    ):
        if mobile_qualification.get(field) != expected:
            raise ValueError(f"G0.10 mobile qualification {field} mismatch")
    if mobile_qualification.get("device_measured") is not True:
        raise ValueError("G0.10 mobile qualification must be device measured")
    if mobile_qualification.get("synthetic") is not False:
        raise ValueError("G0.10 mobile qualification cannot be synthetic")
    if mobile_qualification.get("target_family") != "Samsung Galaxy S21 FE":
        raise ValueError("G0.10 qualification target must be Galaxy S21 FE")
    if mobile_qualification.get("zero_execution_failures") is not True:
        raise ValueError("G0.10 mobile run contains execution failures")
    if mobile_qualification.get("memory_passed") is not True:
        raise ValueError("G0.10 mobile memory qualification failed")
    if mobile_qualification.get("latency_passed") is not True:
        raise ValueError("G0.10 mobile latency qualification failed")
    if mobile_qualification.get("thermal_recovery_passed") is not True:
        raise ValueError("G0.10 mobile thermal recovery qualification failed")
    if mobile_qualification.get("passed") is not True:
        raise ValueError("G0.10 mobile qualification receipt is not passing")

    body: dict[str, object] = {
        "schema": VN97_MAMBA2_G10_PROMOTION_SCHEMA,
        "bridge_id": bridge_id,
        "runtime_id": runtime_id,
        "capsule_id": capsule_id,
        "source_weight_sha256": source_weight,
        "tokenizer_id": tokenizer_id,
        "tuning_id": tuning_id,
        "device_profile_receipt_id": profile_receipt_id,
        "model_parity_receipt_id": model_receipt_id,
        "token_parity_receipt_id": token_receipt_id,
        "mobile_qualification_receipt_id": mobile_receipt_id,
        "model_max_abs_error_e12": int(round(model_max_abs * 1_000_000_000_000)),
        "state_max_abs_error_e12": int(round(state_max_abs * 1_000_000_000_000)),
        "token_parity_cases": cases,
        "target_family": "Samsung Galaxy S21 FE",
        "same_weights_semantics": True,
        "same_token_ids_required": True,
        "real_evidence_required": True,
        "rollback_required": True,
        "production_activation_authorized": True,
    }
    promotion = dict(body)
    promotion["promotion_id"] = _sha256_bytes(
        b"VN97M2G10PROMOTE1\0" + _canonical_json(body)
    )
    return promotion


def build_g10_promotion(
    *,
    bridge_path: Path,
    model_parity_path: Path,
    token_parity_path: Path,
    mobile_qualification_path: Path,
    output_path: Path,
) -> dict[str, object]:
    promotion = compile_g10_promotion(
        _load_object(bridge_path, "G0.9 bridge binding"),
        _load_object(model_parity_path, "G0.10 model parity receipt"),
        _load_object(token_parity_path, "G0.10 token parity receipt"),
        _load_object(
            mobile_qualification_path,
            "G0.10 mobile qualification receipt",
        ),
    )
    if output_path.exists() or output_path.is_symlink():
        raise ValueError("G0.10 promotion output must not already exist")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_name(output_path.name + ".tmp")
    temp.write_bytes(_canonical_json(promotion) + b"\n")
    temp.replace(output_path)
    return promotion
