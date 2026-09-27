from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from .onnx_autotune import verify_r2e4_profile
from .onnx_export import (
    R2_ONNX_INPUTS,
    R2_ONNX_OUTPUTS,
    load_r2_onnx_manifest,
    verify_r2_onnx_bundle,
)


R2F1_RUNTIME_SCHEMA = "VN97R2F1RUNTIME1"
R2F1_RUNTIME_FILENAME = "runtime.vn97ort1.json"


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


def compile_r2f1_runtime_package(
    manifest: Mapping[str, object],
    tuning: Mapping[str, object],
) -> dict[str, object]:
    bundle_id = _require_sha256(
        manifest.get("bundle_id"),
        label="F1 E2 bundle ID",
    )
    if tuning.get("bundle_id") != bundle_id:
        raise ValueError("F1 E4 tuning belongs to another E2 bundle")
    architecture = _require_sha256(
        manifest.get("architecture_fingerprint"),
        label="F1 architecture fingerprint",
    )
    if tuning.get("architecture_fingerprint") != architecture:
        raise ValueError("F1 E4 architecture fingerprint mismatch")
    if tuning.get("profile") != manifest.get("profile"):
        raise ValueError("F1 E4 fast/deep profile mismatch")

    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("F1 E2 config is missing")
    required_config = (
        "vocab_size",
        "d_inner",
        "d_state",
        "d_conv",
        "n_layers",
    )
    for field in required_config:
        value = config.get(field)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"F1 config {field} must be positive integer")

    active_layers = manifest.get("active_layers")
    if (
        not isinstance(active_layers, int)
        or active_layers <= 0
        or active_layers > int(config["n_layers"])
    ):
        raise ValueError("F1 active layer count is invalid")

    state_contract = manifest.get("state_contract")
    if not isinstance(state_contract, dict):
        raise ValueError("F1 E2 state contract is missing")
    if state_contract.get("state_is_explicit") is not True:
        raise ValueError("F1 requires explicit recurrent state")
    if (
        state_contract.get("state_carry_semantics")
        != "exact_between_graph_invocations"
    ):
        raise ValueError("F1 state carry semantics mismatch")

    conv_extent = max(int(config["d_conv"]) - 1, 0)
    if conv_extent <= 0:
        raise ValueError("F1 requires positive convolution state extent")

    graphs = manifest.get("graphs")
    if not isinstance(graphs, list) or not graphs:
        raise ValueError("F1 E2 graph records are missing")
    runtime_graphs: list[dict[str, object]] = []
    names: set[str] = set()
    step_seen = False
    chunk_sizes: list[int] = []
    for graph in graphs:
        if not isinstance(graph, dict):
            raise ValueError("F1 E2 graph record must be object")
        filename = graph.get("filename")
        kind = graph.get("kind")
        sequence_length = graph.get("sequence_length")
        if (
            not isinstance(filename, str)
            or not filename
            or filename in names
            or kind not in {"step", "chunk"}
            or not isinstance(sequence_length, int)
            or sequence_length <= 0
        ):
            raise ValueError("F1 E2 graph record is invalid")
        names.add(filename)
        expected_filename = (
            "step.onnx"
            if kind == "step"
            else f"chunk-{sequence_length}.onnx"
        )
        if filename != expected_filename:
            raise ValueError("F1 E2 graph filename mismatch")
        if graph.get("inputs") != list(R2_ONNX_INPUTS):
            raise ValueError("F1 E2 graph input contract mismatch")
        if graph.get("outputs") != list(R2_ONNX_OUTPUTS):
            raise ValueError("F1 E2 graph output contract mismatch")
        if graph.get("batch_axis_dynamic") is not True:
            raise ValueError("F1 requires dynamic batch axis")
        if graph.get("sequence_axis_dynamic") is not False:
            raise ValueError("F1 requires fixed exported sequence axis")

        graph_sha = _require_sha256(
            graph.get("sha256"),
            label=f"F1 graph {filename} SHA",
        )
        graph_bytes = graph.get("bytes")
        if not isinstance(graph_bytes, int) or graph_bytes <= 0:
            raise ValueError("F1 graph byte size is invalid")
        runtime_graphs.append(
            {
                "filename": filename,
                "kind": kind,
                "sequence_length": sequence_length,
                "sha256": graph_sha,
                "bytes": graph_bytes,
            }
        )
        if kind == "step":
            if sequence_length != 1:
                raise ValueError("F1 step graph length must be one")
            step_seen = True
        else:
            chunk_sizes.append(sequence_length)

    if not step_seen or not chunk_sizes:
        raise ValueError("F1 requires step and chunk graphs")
    runtime_graphs.sort(
        key=lambda item: (
            0 if item["kind"] == "step" else 1,
            int(item["sequence_length"]),
            str(item["filename"]),
        )
    )

    tuning_graphs = tuning.get("graph_policies")
    if not isinstance(tuning_graphs, list):
        raise ValueError("F1 E4 graph policy is missing")
    tuning_names = {
        item.get("graph_filename")
        for item in tuning_graphs
        if isinstance(item, dict)
    }
    if tuning_names != names:
        raise ValueError("F1 E4 graph coverage differs from E2")

    body = {
        "schema": R2F1_RUNTIME_SCHEMA,
        "bundle_id": bundle_id,
        "architecture_fingerprint": architecture,
        "profile": manifest.get("profile"),
        "tuning_id": _require_sha256(
            tuning.get("tuning_id"),
            label="F1 E4 tuning ID",
        ),
        "active_layers": active_layers,
        "vocab_size": int(config["vocab_size"]),
        "d_inner": int(config["d_inner"]),
        "d_state": int(config["d_state"]),
        "d_conv_state": conv_extent,
        "batch_size": 1,
        "inputs": list(R2_ONNX_INPUTS),
        "outputs": list(R2_ONNX_OUTPUTS),
        "graphs": runtime_graphs,
        "supported_chunk_sizes": sorted(chunk_sizes),
        "state_dtype": "float32",
        "token_dtype": "int64",
        "same_weights_semantics": True,
        "quantization_used": False,
    }
    package = dict(body)
    package["runtime_id"] = _sha256_bytes(
        b"VN97R2F1RUNTIME1\0" + _canonical_json(body)
    )
    return package


def verify_r2f1_runtime_package(
    path: Path,
    *,
    expected_bundle_id: str | None = None,
    expected_tuning_id: str | None = None,
) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("F1 runtime package must be regular non-symlink file")
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
        raise ValueError("F1 runtime package must be strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("F1 runtime package must be one JSON object")
    if payload.get("schema") != R2F1_RUNTIME_SCHEMA:
        raise ValueError("F1 runtime package schema mismatch")

    runtime_id = _require_sha256(
        payload.get("runtime_id"),
        label="F1 runtime ID",
    )
    body = dict(payload)
    body.pop("runtime_id", None)
    expected_id = _sha256_bytes(
        b"VN97R2F1RUNTIME1\0" + _canonical_json(body)
    )
    if runtime_id != expected_id:
        raise ValueError("F1 runtime package identity mismatch")

    bundle_id = _require_sha256(
        payload.get("bundle_id"),
        label="F1 bundle ID",
    )
    tuning_id = _require_sha256(
        payload.get("tuning_id"),
        label="F1 tuning ID",
    )
    _require_sha256(
        payload.get("architecture_fingerprint"),
        label="F1 architecture fingerprint",
    )
    if expected_bundle_id is not None and bundle_id != expected_bundle_id:
        raise ValueError("F1 package belongs to another E2 bundle")
    if expected_tuning_id is not None and tuning_id != expected_tuning_id:
        raise ValueError("F1 package belongs to another E4 tuning profile")

    if payload.get("batch_size") != 1:
        raise ValueError("F1 production mobile runtime requires batch_size=1")
    if payload.get("state_dtype") != "float32":
        raise ValueError("F1 state dtype mismatch")
    if payload.get("token_dtype") != "int64":
        raise ValueError("F1 token dtype mismatch")
    if payload.get("inputs") != list(R2_ONNX_INPUTS):
        raise ValueError("F1 input contract mismatch")
    if payload.get("outputs") != list(R2_ONNX_OUTPUTS):
        raise ValueError("F1 output contract mismatch")
    if payload.get("same_weights_semantics") is not True:
        raise ValueError("F1 same-weights semantics are not locked")
    if payload.get("quantization_used") is not False:
        raise ValueError("F1 must not introduce quantization")

    positive = (
        "active_layers",
        "vocab_size",
        "d_inner",
        "d_state",
        "d_conv_state",
    )
    for field in positive:
        value = payload.get(field)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"F1 {field} must be positive integer")

    graphs = payload.get("graphs")
    if not isinstance(graphs, list) or not graphs:
        raise ValueError("F1 graph list is missing")
    names: set[str] = set()
    chunks: list[int] = []
    step_seen = False
    for graph in graphs:
        if not isinstance(graph, dict):
            raise ValueError("F1 graph record is invalid")
        if set(graph) != {
            "filename",
            "kind",
            "sequence_length",
            "sha256",
            "bytes",
        }:
            raise ValueError("F1 graph fields mismatch")
        name = graph.get("filename")
        kind = graph.get("kind")
        length = graph.get("sequence_length")
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or kind not in {"step", "chunk"}
            or not isinstance(length, int)
            or length <= 0
        ):
            raise ValueError("F1 graph record is invalid")
        names.add(name)
        _require_sha256(
            graph.get("sha256"),
            label=f"F1 graph {name} SHA",
        )
        size = graph.get("bytes")
        if not isinstance(size, int) or size <= 0:
            raise ValueError("F1 graph bytes are invalid")
        expected_name = (
            "step.onnx"
            if kind == "step"
            else f"chunk-{length}.onnx"
        )
        if name != expected_name:
            raise ValueError("F1 graph filename mismatch")
        if kind == "step":
            if length != 1:
                raise ValueError("F1 step graph length must be one")
            step_seen = True
        else:
            chunks.append(length)
    if not step_seen or not chunks:
        raise ValueError("F1 package requires step and chunk graphs")

    supported = payload.get("supported_chunk_sizes")
    if (
        not isinstance(supported, list)
        or supported != sorted(chunks)
        or len(supported) != len(set(supported))
    ):
        raise ValueError("F1 supported chunk sizes mismatch")
    return payload


def build_r2f1_runtime_package(
    *,
    bundle_dir: Path,
    e4_profile_path: Path,
    output_path: Path | None = None,
) -> dict[str, object]:
    manifest = verify_r2_onnx_bundle(bundle_dir)
    tuning = verify_r2e4_profile(
        e4_profile_path,
        expected_bundle_id=str(manifest["bundle_id"]),
    )
    package = compile_r2f1_runtime_package(manifest, tuning)
    target = output_path or bundle_dir / R2F1_RUNTIME_FILENAME
    if target.exists() or target.is_symlink():
        raise ValueError("F1 runtime package output must not already exist")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    temp.write_bytes(_canonical_json(package) + b"\n")
    temp.replace(target)
    verify_r2f1_runtime_package(
        target,
        expected_bundle_id=str(manifest["bundle_id"]),
        expected_tuning_id=str(tuning["tuning_id"]),
    )
    return package
