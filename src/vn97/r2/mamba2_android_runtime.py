from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from .mamba2_parallel_onnx import (
    VN97_MAMBA2_G05_INPUTS,
    VN97_MAMBA2_G05_OUTPUTS,
    verify_g05_bundle,
)


VN97_MAMBA2_G06_RUNTIME_SCHEMA = "VN97M2G06RUNTIME1"
VN97_MAMBA2_G06_RUNTIME_FILENAME = "runtime.vn97m2g06.json"


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


def compile_g06_runtime_descriptor(
    manifest: Mapping[str, object],
) -> dict[str, object]:
    if manifest.get("schema") != "VN97M2G05ONNX1":
        raise ValueError("G0.6 requires a G0.5 ONNX manifest")
    manifest_id = _require_sha256(
        manifest.get("manifest_id"),
        "G0.5 manifest ID",
    )
    capsule_id = _require_sha256(
        manifest.get("capsule_id"),
        "G0.5 capsule ID",
    )
    capsule_manifest_sha256 = _require_sha256(
        manifest.get("capsule_manifest_sha256"),
        "G0.5 capsule manifest SHA-256",
    )
    source_weight_sha256 = _require_sha256(
        manifest.get("source_weight_sha256"),
        "G0.5 source weight SHA-256",
    )
    if manifest.get("inputs") != list(VN97_MAMBA2_G05_INPUTS):
        raise ValueError("G0.6 input contract differs from G0.5")
    if manifest.get("outputs") != list(VN97_MAMBA2_G05_OUTPUTS):
        raise ValueError("G0.6 output contract differs from G0.5")
    if manifest.get("single_weight_graph") is not True:
        raise ValueError("G0.6 requires one shared-weight recurrent graph")
    if manifest.get("decode_via_valid_length_one") is not True:
        raise ValueError("G0.6 requires valid_length=1 decode")
    if manifest.get("parallel_prefill_ready") is not True:
        raise ValueError("G0.6 requires parallel prefill readiness")
    if manifest.get("same_weights_semantics") is not True:
        raise ValueError("G0.6 same-weight semantics missing")
    if manifest.get("quantization_used") is not False:
        raise ValueError("G0.6 must not introduce quantization")
    if manifest.get("production_activation_authorized") is not False:
        raise ValueError("G0.5 cannot self-authorize G0.6 production")

    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("G0.6 config missing")
    required = (
        "d_model",
        "n_layers",
        "vocab_size",
        "d_state",
        "d_conv",
        "expand",
        "head_dim",
        "n_groups",
    )
    values: dict[str, int] = {}
    for field in required:
        value = config.get(field)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"G0.6 config {field} must be positive integer")
        values[field] = value
    if (
        values["d_model"] != 2560
        or values["n_layers"] != 64
        or values["vocab_size"] != 50288
        or values["d_state"] != 128
        or values["d_conv"] != 4
        or values["expand"] != 2
        or values["head_dim"] != 64
        or values["n_groups"] != 1
    ):
        raise ValueError("G0.6 config is not the locked Mamba-2 2.7B geometry")

    d_inner = values["d_model"] * values["expand"]
    n_heads = d_inner // values["head_dim"]
    conv_dim = d_inner + 2 * values["d_state"]
    if d_inner != 5120 or n_heads != 80 or conv_dim != 5376:
        raise ValueError("G0.6 derived geometry mismatch")

    state = manifest.get("state_contract")
    if not isinstance(state, Mapping):
        raise ValueError("G0.6 state contract missing")
    expected_conv = [64, 1, 5376, 4]
    expected_ssm = [64, 1, 80, 64, 128]
    if state.get("conv_state_shape") != expected_conv:
        raise ValueError("G0.6 conv state geometry mismatch")
    if state.get("ssm_state_shape") != expected_ssm:
        raise ValueError("G0.6 SSD state geometry mismatch")
    if state.get("state_is_explicit") is not True:
        raise ValueError("G0.6 requires explicit recurrent state")
    if (
        state.get("state_carry_semantics")
        != "exact_between_graph_invocations"
    ):
        raise ValueError("G0.6 recurrent state carry semantics mismatch")

    graph_files = manifest.get("graph_files")
    if not isinstance(graph_files, list) or not graph_files:
        raise ValueError("G0.6 graph inventory missing")
    max_chunk = manifest.get("max_chunk_size")
    if max_chunk not in (8, 16, 32):
        raise ValueError("G0.6 max chunk size unsupported")
    graph_name = f"recurrent-{max_chunk}.onnx"
    normalized_files = []
    seen = set()
    for record in graph_files:
        if not isinstance(record, Mapping):
            raise ValueError("G0.6 graph record invalid")
        filename = record.get("filename")
        size = record.get("bytes")
        digest = _require_sha256(
            record.get("sha256"),
            "G0.6 graph file SHA-256",
        )
        if (
            not isinstance(filename, str)
            or not filename
            or "/" in filename
            or "\\" in filename
            or filename in seen
        ):
            raise ValueError("G0.6 graph filename invalid")
        if not isinstance(size, int) or size <= 0:
            raise ValueError("G0.6 graph file size invalid")
        seen.add(filename)
        normalized_files.append(
            {
                "filename": filename,
                "bytes": size,
                "sha256": digest,
            }
        )
    if graph_name not in seen:
        raise ValueError("G0.6 primary recurrent graph missing")

    body: dict[str, object] = {
        "schema": VN97_MAMBA2_G06_RUNTIME_SCHEMA,
        "g05_manifest_id": manifest_id,
        "capsule_id": capsule_id,
        "capsule_manifest_sha256": capsule_manifest_sha256,
        "source_weight_sha256": source_weight_sha256,
        "graph_filename": graph_name,
        "graph_files": sorted(
            normalized_files,
            key=lambda item: str(item["filename"]),
        ),
        "max_chunk_size": max_chunk,
        "valid_length_min": 1,
        "valid_length_max": max_chunk,
        "inputs": list(VN97_MAMBA2_G05_INPUTS),
        "outputs": list(VN97_MAMBA2_G05_OUTPUTS),
        "batch_size": 1,
        "vocab_size": values["vocab_size"],
        "d_model": values["d_model"],
        "n_layers": values["n_layers"],
        "d_inner": d_inner,
        "n_heads": n_heads,
        "head_dim": values["head_dim"],
        "d_state": values["d_state"],
        "d_conv": values["d_conv"],
        "conv_dim": conv_dim,
        "conv_state_shape": expected_conv,
        "ssm_state_shape": expected_ssm,
        "token_dtype": "int64",
        "valid_length_dtype": "int64",
        "state_dtype": str(state.get("dtype")),
        "single_weight_graph": True,
        "decode_via_valid_length_one": True,
        "parallel_prefill_ready": True,
        "same_weights_semantics": True,
        "quantization_used": False,
        "source_runtime_required": False,
        "production_activation_authorized": False,
    }
    descriptor = dict(body)
    descriptor["runtime_id"] = _sha256_bytes(
        b"VN97M2G06RUNTIME1\0" + _canonical_json(body)
    )
    return descriptor


def build_g06_runtime_descriptor(
    *,
    bundle_dir: Path,
    output_path: Path | None = None,
) -> dict[str, object]:
    manifest = verify_g05_bundle(bundle_dir)
    descriptor = compile_g06_runtime_descriptor(manifest)
    target = output_path or bundle_dir / VN97_MAMBA2_G06_RUNTIME_FILENAME
    if target.exists() or target.is_symlink():
        raise ValueError("G0.6 runtime descriptor output must not exist")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    temp.write_bytes(_canonical_json(descriptor) + b"\n")
    temp.replace(target)
    return descriptor
