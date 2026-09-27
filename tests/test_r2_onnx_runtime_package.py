from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vn97.r2.onnx_runtime_package import (
    R2F1_RUNTIME_SCHEMA,
    compile_r2f1_runtime_package,
    verify_r2f1_runtime_package,
)


BUNDLE = "a" * 64
ARCH = "b" * 64
TUNING = "c" * 64


def _manifest() -> dict[str, object]:
    return {
        "bundle_id": BUNDLE,
        "architecture_fingerprint": ARCH,
        "profile": "deep",
        "active_layers": 4,
        "config": {
            "vocab_size": 320,
            "d_inner": 128,
            "d_state": 8,
            "d_conv": 4,
            "n_layers": 4,
        },
        "state_contract": {
            "state_is_explicit": True,
            "state_carry_semantics": "exact_between_graph_invocations",
        },
        "graphs": [
            {
                "filename": "step.onnx",
                "kind": "step",
                "sequence_length": 1,
                "sha256": "d" * 64,
                "bytes": 101,
                "inputs": ["input_ids", "conv_state", "ssm_state"],
                "outputs": [
                    "logits",
                    "next_conv_state",
                    "next_ssm_state",
                ],
                "batch_axis_dynamic": True,
                "sequence_axis_dynamic": False,
            },
            {
                "filename": "chunk-32.onnx",
                "kind": "chunk",
                "sequence_length": 32,
                "sha256": "e" * 64,
                "bytes": 202,
                "inputs": ["input_ids", "conv_state", "ssm_state"],
                "outputs": [
                    "logits",
                    "next_conv_state",
                    "next_ssm_state",
                ],
                "batch_axis_dynamic": True,
                "sequence_axis_dynamic": False,
            },
            {
                "filename": "chunk-8.onnx",
                "kind": "chunk",
                "sequence_length": 8,
                "sha256": "f" * 64,
                "bytes": 150,
                "inputs": ["input_ids", "conv_state", "ssm_state"],
                "outputs": [
                    "logits",
                    "next_conv_state",
                    "next_ssm_state",
                ],
                "batch_axis_dynamic": True,
                "sequence_axis_dynamic": False,
            },
        ],
    }


def _tuning() -> dict[str, object]:
    return {
        "bundle_id": BUNDLE,
        "architecture_fingerprint": ARCH,
        "profile": "deep",
        "tuning_id": TUNING,
        "graph_policies": [
            {"graph_filename": "step.onnx"},
            {"graph_filename": "chunk-8.onnx"},
            {"graph_filename": "chunk-32.onnx"},
        ],
    }


def test_compile_runtime_package_is_deterministic_and_portable() -> None:
    first = compile_r2f1_runtime_package(
        _manifest(),
        _tuning(),
    )
    manifest = _manifest()
    manifest["graphs"] = list(reversed(manifest["graphs"]))
    second = compile_r2f1_runtime_package(
        manifest,
        _tuning(),
    )
    assert first == second
    assert first["schema"] == R2F1_RUNTIME_SCHEMA
    assert first["batch_size"] == 1
    assert first["d_conv_state"] == 3
    assert first["supported_chunk_sizes"] == [8, 32]
    assert [item["filename"] for item in first["graphs"]] == [
        "step.onnx",
        "chunk-8.onnx",
        "chunk-32.onnx",
    ]


def test_compile_rejects_tuning_graph_coverage_mismatch() -> None:
    tuning = _tuning()
    tuning["graph_policies"] = tuning["graph_policies"][:-1]
    with pytest.raises(ValueError, match="coverage differs"):
        compile_r2f1_runtime_package(
            _manifest(),
            tuning,
        )


def test_compile_rejects_state_contract_change() -> None:
    manifest = _manifest()
    manifest["state_contract"] = {
        "state_is_explicit": True,
        "state_carry_semantics": "other",
    }
    with pytest.raises(ValueError, match="state carry semantics"):
        compile_r2f1_runtime_package(
            manifest,
            _tuning(),
        )


def test_verify_roundtrip_and_semantic_tamper(tmp_path: Path) -> None:
    payload = compile_r2f1_runtime_package(
        _manifest(),
        _tuning(),
    )
    path = tmp_path / "runtime.vn97ort1.json"
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    verified = verify_r2f1_runtime_package(
        path,
        expected_bundle_id=BUNDLE,
        expected_tuning_id=TUNING,
    )
    assert verified == payload

    payload["batch_size"] = 2
    body = dict(payload)
    body.pop("runtime_id", None)
    payload["runtime_id"] = hashlib.sha256(
        b"VN97R2F1RUNTIME1\0"
        + json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="batch_size=1"):
        verify_r2f1_runtime_package(path)


def test_compile_rejects_graph_contract_drift() -> None:
    manifest = _manifest()
    manifest["graphs"][0]["inputs"] = ["input_ids"]
    with pytest.raises(ValueError, match="input contract"):
        compile_r2f1_runtime_package(
            manifest,
            _tuning(),
        )


def test_verify_rejects_duplicate_or_extra_fields(tmp_path: Path) -> None:
    payload = compile_r2f1_runtime_package(
        _manifest(),
        _tuning(),
    )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    duplicate = encoded[:-1] + ',"runtime_id":"' + payload["runtime_id"] + '"}'
    path = tmp_path / "duplicate.json"
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate keys"):
        verify_r2f1_runtime_package(path)

    payload["unexpected"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fields mismatch"):
        verify_r2f1_runtime_package(path)
