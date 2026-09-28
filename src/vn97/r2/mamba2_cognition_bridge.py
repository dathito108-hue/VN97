from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping


VN97_MAMBA2_G09_BRIDGE_SCHEMA = "VN97M2G09BRIDGE1"
VN97_MAMBA2_G09_BRIDGE_FILENAME = "binding.vn97m2g09.json"


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


def _load_object(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be regular non-symlink file")
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be JSON object")
    return value


def compile_g09_bridge_binding(
    runtime: Mapping[str, object],
    tokenizer: Mapping[str, object],
    tuning: Mapping[str, object],
) -> dict[str, object]:
    if runtime.get("schema") != "VN97M2G06RUNTIME1":
        raise ValueError("G0.9 requires G0.6 runtime descriptor")
    if tokenizer.get("schema") != "VN97M2G08TOK1":
        raise ValueError("G0.9 requires G0.8 tokenizer descriptor")
    if tuning.get("schema") != "VN97M2G07TUNE1":
        raise ValueError("G0.9 requires G0.7 tuning descriptor")

    runtime_id = _require_sha256(
        runtime.get("runtime_id"),
        "G0.9 runtime ID",
    )
    tokenizer_id = _require_sha256(
        tokenizer.get("tokenizer_id"),
        "G0.9 tokenizer ID",
    )
    tuning_id = _require_sha256(
        tuning.get("tuning_id"),
        "G0.9 tuning ID",
    )
    capsule_id = _require_sha256(
        runtime.get("capsule_id"),
        "G0.9 capsule ID",
    )
    tokenizer_capsule = _require_sha256(
        tokenizer.get("capsule_id"),
        "G0.9 tokenizer capsule ID",
    )
    if tokenizer_capsule != capsule_id:
        raise ValueError("G0.9 runtime/tokenizer capsule identity mismatch")
    if tuning.get("runtime_id") != runtime_id:
        raise ValueError("G0.9 tuning belongs to another runtime")

    if runtime.get("same_weights_semantics") is not True:
        raise ValueError("G0.9 requires same-weight runtime semantics")
    if runtime.get("quantization_used") is not False:
        raise ValueError("G0.9 dense candidate must not be quantized")
    if tokenizer.get("same_token_ids_required") is not True:
        raise ValueError("G0.9 requires source token-ID semantics")
    if tuning.get("profile_is_device_measured") is not True:
        raise ValueError("G0.9 requires device-measured tuning")
    if any(
        value.get("production_activation_authorized") is not False
        for value in (runtime, tokenizer, tuning)
    ):
        raise ValueError(
            "G0.9 inputs must remain non-production-authorizing evidence"
        )

    token_id_space = tokenizer.get("token_id_space")
    runtime_logits_size = tokenizer.get("runtime_logits_size")
    eos_token_id = tokenizer.get("eos_token_id")
    if token_id_space != 50_277:
        raise ValueError("G0.9 GPT-NeoX token-ID space mismatch")
    if runtime_logits_size != 50_288:
        raise ValueError("G0.9 runtime logits width mismatch")
    if runtime.get("vocab_size") != runtime_logits_size:
        raise ValueError("G0.9 tokenizer/runtime logits width mismatch")
    if (
        not isinstance(eos_token_id, int)
        or not 0 <= eos_token_id < token_id_space
    ):
        raise ValueError("G0.9 EOS token ID invalid")

    source_weight = _require_sha256(
        runtime.get("source_weight_sha256"),
        "G0.9 source weight SHA-256",
    )
    g05_manifest_id = _require_sha256(
        runtime.get("g05_manifest_id"),
        "G0.9 G0.5 manifest ID",
    )
    profile_receipt_id = _require_sha256(
        tuning.get("profile_receipt_id"),
        "G0.9 device profile receipt ID",
    )

    body: dict[str, object] = {
        "schema": VN97_MAMBA2_G09_BRIDGE_SCHEMA,
        "runtime_id": runtime_id,
        "g05_manifest_id": g05_manifest_id,
        "capsule_id": capsule_id,
        "source_weight_sha256": source_weight,
        "tokenizer_id": tokenizer_id,
        "tuning_id": tuning_id,
        "device_profile_receipt_id": profile_receipt_id,
        "token_id_space": token_id_space,
        "runtime_logits_size": runtime_logits_size,
        "eos_token_id": eos_token_id,
        "same_weights_semantics": True,
        "same_token_ids_required": True,
        "padded_logits_must_be_masked": True,
        "device_measured_tuning_required": True,
        "candidate_validation_only": True,
        "production_activation_authorized": False,
    }
    binding = dict(body)
    binding["bridge_id"] = _sha256_bytes(
        b"VN97M2G09BRIDGE1\0" + _canonical_json(body)
    )
    return binding


def build_g09_bridge_binding(
    *,
    runtime_path: Path,
    tokenizer_path: Path,
    tuning_path: Path,
    output_path: Path,
) -> dict[str, object]:
    binding = compile_g09_bridge_binding(
        _load_object(runtime_path, "G0.6 runtime descriptor"),
        _load_object(tokenizer_path, "G0.8 tokenizer descriptor"),
        _load_object(tuning_path, "G0.7 tuning descriptor"),
    )
    if output_path.exists() or output_path.is_symlink():
        raise ValueError("G0.9 bridge output must not already exist")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_name(output_path.name + ".tmp")
    temp.write_bytes(_canonical_json(binding) + b"\n")
    temp.replace(output_path)
    return binding
