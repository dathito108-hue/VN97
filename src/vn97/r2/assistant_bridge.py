from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from .onnx_export import verify_r2_onnx_bundle
from .onnx_runtime_package import (
    R2F1_RUNTIME_FILENAME,
    verify_r2f1_runtime_package,
)

R2F2_BINDING_SCHEMA = "VN97R2F2BIND1"
R2F2_BINDING_FILENAME = "assistant.vn97r2f2.json"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
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


def compile_r2f2_binding(
    manifest: Mapping[str, object],
    runtime: Mapping[str, object],
    *,
    tokenizer_model_sha256: str,
) -> dict[str, object]:
    checkpoint_sha = _require_sha256(
        manifest.get("checkpoint_sha256"),
        label="F2 checkpoint SHA-256",
    )
    tokenizer_sha = _require_sha256(
        tokenizer_model_sha256,
        label="F2 tokenizer model SHA-256",
    )
    bundle_id = _require_sha256(
        manifest.get("bundle_id"),
        label="F2 bundle ID",
    )
    runtime_id = _require_sha256(
        runtime.get("runtime_id"),
        label="F2 runtime ID",
    )
    architecture = _require_sha256(
        manifest.get("architecture_fingerprint"),
        label="F2 architecture fingerprint",
    )
    if runtime.get("bundle_id") != bundle_id:
        raise ValueError("F2 runtime belongs to another E2 bundle")
    if runtime.get("architecture_fingerprint") != architecture:
        raise ValueError("F2 runtime architecture differs from E2")
    if runtime.get("profile") != manifest.get("profile"):
        raise ValueError("F2 runtime profile differs from E2")
    if runtime.get("active_layers") != manifest.get("active_layers"):
        raise ValueError("F2 active-layer count differs from E2")

    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("F2 E2 config is missing")
    required = ("vocab_size", "d_model", "n_layers", "d_state")
    if any(
        not isinstance(config.get(key), int) or int(config[key]) <= 0
        for key in required
    ):
        raise ValueError("F2 E2 geometry is invalid")
    if int(runtime.get("vocab_size", -1)) != int(config["vocab_size"]):
        raise ValueError("F2 runtime vocabulary differs from checkpoint")
    if int(runtime.get("d_state", -1)) != int(config["d_state"]):
        raise ValueError("F2 runtime state geometry differs from checkpoint")

    checkpoint_stage = manifest.get("checkpoint_stage")
    if not isinstance(checkpoint_stage, str) or not checkpoint_stage:
        raise ValueError("F2 requires a named validated checkpoint stage")

    body: dict[str, object] = {
        "schema": R2F2_BINDING_SCHEMA,
        "runtime_id": runtime_id,
        "bundle_id": bundle_id,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_stage": checkpoint_stage,
        "tokenizer_model_sha256": tokenizer_sha,
        "architecture_fingerprint": architecture,
        "profile": manifest.get("profile"),
        "active_layers": int(manifest["active_layers"]),
        "vocab_size": int(config["vocab_size"]),
        "d_model": int(config["d_model"]),
        "n_layers": int(config["n_layers"]),
        "d_state": int(config["d_state"]),
        "same_weights_semantics": True,
        "legacy_inference_fallback": False,
    }
    result = dict(body)
    result["binding_id"] = _sha256_bytes(
        b"VN97R2F2BIND1\0" + _canonical_json(body)
    )
    return result


def verify_r2f2_binding(
    path: Path,
    *,
    expected_runtime_id: str | None = None,
    expected_tokenizer_model_sha256: str | None = None,
) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("F2 binding must be a regular non-symlink file")
    duplicates: list[str] = []

    def hook(pairs):
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                duplicates.append(key)
            output[key] = value
        return output

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8", errors="strict"),
            object_pairs_hook=hook,
            parse_constant=lambda raw: (
                _ for _ in ()
            ).throw(ValueError(raw)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("F2 binding must be strict UTF-8 JSON") from exc
    if duplicates or not isinstance(payload, dict):
        raise ValueError("F2 binding must be one object without duplicate keys")

    fields = {
        "schema",
        "runtime_id",
        "bundle_id",
        "checkpoint_sha256",
        "checkpoint_stage",
        "tokenizer_model_sha256",
        "architecture_fingerprint",
        "profile",
        "active_layers",
        "vocab_size",
        "d_model",
        "n_layers",
        "d_state",
        "same_weights_semantics",
        "legacy_inference_fallback",
        "binding_id",
    }
    if set(payload) != fields:
        raise ValueError("F2 binding fields mismatch")
    if payload.get("schema") != R2F2_BINDING_SCHEMA:
        raise ValueError("F2 binding schema mismatch")

    binding_id = _require_sha256(
        payload.get("binding_id"),
        label="F2 binding ID",
    )
    body = dict(payload)
    body.pop("binding_id", None)
    if binding_id != _sha256_bytes(
        b"VN97R2F2BIND1\0" + _canonical_json(body)
    ):
        raise ValueError("F2 binding identity mismatch")

    for key in (
        "runtime_id",
        "bundle_id",
        "checkpoint_sha256",
        "tokenizer_model_sha256",
        "architecture_fingerprint",
    ):
        _require_sha256(payload.get(key), label=f"F2 {key}")
    for key in (
        "active_layers",
        "vocab_size",
        "d_model",
        "n_layers",
        "d_state",
    ):
        value = payload.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"F2 {key} must be positive integer")
    if payload["active_layers"] > payload["n_layers"]:
        raise ValueError("F2 active layers exceed checkpoint layers")
    if not isinstance(payload.get("checkpoint_stage"), str) or not payload["checkpoint_stage"]:
        raise ValueError("F2 checkpoint stage is missing")
    if payload.get("same_weights_semantics") is not True:
        raise ValueError("F2 same-weight semantic lock is missing")
    if payload.get("legacy_inference_fallback") is not False:
        raise ValueError("F2 legacy inference fallback must remain disabled")
    if expected_runtime_id is not None and payload["runtime_id"] != expected_runtime_id:
        raise ValueError("F2 binding belongs to another runtime")
    if (
        expected_tokenizer_model_sha256 is not None
        and payload["tokenizer_model_sha256"]
        != expected_tokenizer_model_sha256
    ):
        raise ValueError("F2 binding belongs to another tokenizer artifact")
    return payload


def build_r2f2_binding(
    *,
    bundle_dir: Path,
    tokenizer_model_sha256: str,
    output_path: Path | None = None,
) -> dict[str, object]:
    manifest = verify_r2_onnx_bundle(bundle_dir)
    runtime_path = bundle_dir / R2F1_RUNTIME_FILENAME
    runtime = verify_r2f1_runtime_package(
        runtime_path,
        expected_bundle_id=str(manifest["bundle_id"]),
    )
    payload = compile_r2f2_binding(
        manifest,
        runtime,
        tokenizer_model_sha256=tokenizer_model_sha256,
    )
    target = output_path or bundle_dir / R2F2_BINDING_FILENAME
    if target.exists() or target.is_symlink():
        raise ValueError("F2 binding output must not already exist")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    temp.write_bytes(_canonical_json(payload) + b"\n")
    temp.replace(target)
    verify_r2f2_binding(
        target,
        expected_runtime_id=str(runtime["runtime_id"]),
        expected_tokenizer_model_sha256=tokenizer_model_sha256,
    )
    return payload
