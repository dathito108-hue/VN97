"""Dependency-free G0.5 inventory checks; no inference or parity claims."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

VN97_MAMBA2_G05_SCHEMA = "VN97M2G05ONNX1"
VN97_MAMBA2_G05_OPSET = 18
VN97_MAMBA2_G05_INPUTS = (
    "input_ids",
    "valid_length",
    "conv_state",
    "ssm_state",
)
VN97_MAMBA2_G05_OUTPUTS = (
    "logits",
    "next_conv_state",
    "next_ssm_state",
)
VN97_MAMBA2_G05_CHUNKS = (8, 16, 32)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _safe_filename(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and value not in {".", ".."}
        and not any(ch in value for ch in ("/", "\\", ":"))
        and all(ord(ch) >= 32 and ord(ch) != 127 for ch in value)
    )


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"G0.5 duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"G0.5 non-finite JSON value: {value}")


def verify_g05_bundle(root: Path) -> dict[str, object]:
    resolved = root.resolve(strict=True)
    path = resolved / "manifest.vn97m2g05.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("G0.5 manifest must be a regular non-symlink file")
    payload = json.loads(
        path.read_text(encoding="ascii"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError("G0.5 manifest must be an object")
    if payload.get("schema") != VN97_MAMBA2_G05_SCHEMA:
        raise ValueError("G0.5 schema mismatch")
    manifest_id = _require_sha256(
        payload.get("manifest_id"),
        label="G0.5 manifest ID",
    )
    body = dict(payload)
    body.pop("manifest_id", None)
    expected = hashlib.sha256(
        b"VN97M2G05ONNX1\0" + _canonical_json(body)
    ).hexdigest()
    if manifest_id != expected:
        raise ValueError("G0.5 manifest identity mismatch")
    for field in (
        "capsule_id",
        "capsule_manifest_sha256",
        "source_weight_sha256",
    ):
        _require_sha256(payload.get(field), label=f"G0.5 {field}")
    if payload.get("single_weight_graph") is not True:
        raise ValueError("G0.5 must use one shared-weight recurrent graph")
    if payload.get("decode_via_valid_length_one") is not True:
        raise ValueError("G0.5 decode contract mismatch")
    if payload.get("parallel_prefill_ready") is not True:
        raise ValueError("G0.5 parallel prefill contract missing")
    if payload.get("same_weights_semantics") is not True:
        raise ValueError("G0.5 same-weight semantics mismatch")
    if payload.get("quantization_used") is not False:
        raise ValueError("G0.5 must not quantize inherited weights")
    if payload.get("production_activation_authorized") is not False:
        raise ValueError("G0.5 cannot self-authorize production")
    chunk = payload.get("max_chunk_size")
    if chunk not in VN97_MAMBA2_G05_CHUNKS:
        raise ValueError("G0.5 chunk size invalid")
    if (
        payload.get("valid_length_min") != 1
        or payload.get("valid_length_max") != chunk
    ):
        raise ValueError("G0.5 valid-length contract mismatch")
    if payload.get("inputs") != list(VN97_MAMBA2_G05_INPUTS):
        raise ValueError("G0.5 input contract mismatch")
    if payload.get("outputs") != list(VN97_MAMBA2_G05_OUTPUTS):
        raise ValueError("G0.5 output contract mismatch")

    files = payload.get("graph_files")
    if not isinstance(files, list) or not files:
        raise ValueError("G0.5 graph inventory missing")
    expected_graph = f"recurrent-{chunk}.onnx"
    seen = set()
    for record in files:
        if not isinstance(record, dict):
            raise ValueError("G0.5 graph record invalid")
        name = record.get("filename")
        if not _safe_filename(name) or name in seen:
            raise ValueError("G0.5 graph filename invalid")
        size = record.get("bytes")
        if type(size) is not int or size <= 0:
            raise ValueError("G0.5 graph byte size must be positive integer")
        _require_sha256(record.get("sha256"), label="G0.5 graph SHA-256")
        seen.add(name)
        file = resolved / name
        if file.is_symlink() or not file.is_file():
            raise ValueError(f"G0.5 graph file missing: {name}")
        if file.stat().st_size != record.get("bytes"):
            raise ValueError(f"G0.5 graph byte size mismatch: {name}")
        if _sha256_file(file) != record.get("sha256"):
            raise ValueError(f"G0.5 graph hash mismatch: {name}")
    if expected_graph not in seen:
        raise ValueError("G0.5 recurrent graph missing")
    return payload


