import json
from pathlib import Path

from vn97.r2.mamba2_android_runtime import (
    VN97_MAMBA2_G06_RUNTIME_SCHEMA,
    compile_g06_runtime_descriptor,
)


def _manifest() -> dict[str, object]:
    return {
        "schema": "VN97M2G05ONNX1",
        "manifest_id": "a" * 64,
        "capsule_id": "b" * 64,
        "capsule_manifest_sha256": "c" * 64,
        "source_weight_sha256": "d" * 64,
        "config": {
            "d_model": 2560,
            "n_layers": 64,
            "vocab_size": 50288,
            "d_state": 128,
            "d_conv": 4,
            "expand": 2,
            "head_dim": 64,
            "n_groups": 1,
            "rms_eps": 1e-5,
        },
        "max_chunk_size": 32,
        "valid_length_min": 1,
        "valid_length_max": 32,
        "inputs": [
            "input_ids",
            "valid_length",
            "conv_state",
            "ssm_state",
        ],
        "outputs": [
            "logits",
            "next_conv_state",
            "next_ssm_state",
        ],
        "graph_files": [
            {
                "filename": "recurrent-32.onnx",
                "bytes": 123,
                "sha256": "e" * 64,
            },
            {
                "filename": "recurrent-32.onnx.data",
                "bytes": 456,
                "sha256": "f" * 64,
            },
        ],
        "state_contract": {
            "batch_size": 1,
            "dtype": "float32",
            "conv_state_shape": [64, 1, 5376, 4],
            "ssm_state_shape": [64, 1, 80, 64, 128],
            "total_bytes_per_batch": 173277184,
            "state_is_explicit": True,
            "state_carry_semantics": "exact_between_graph_invocations",
        },
        "single_weight_graph": True,
        "decode_via_valid_length_one": True,
        "parallel_prefill_ready": True,
        "parallel_algorithm": "one_chunk_ssd_factorization",
        "same_weights_semantics": True,
        "quantization_used": False,
        "external_data_requested": True,
        "production_activation_authorized": False,
    }


def test_g06_descriptor_locks_four_input_mamba2_geometry() -> None:
    descriptor = compile_g06_runtime_descriptor(_manifest())
    assert descriptor["schema"] == VN97_MAMBA2_G06_RUNTIME_SCHEMA
    assert descriptor["inputs"] == [
        "input_ids",
        "valid_length",
        "conv_state",
        "ssm_state",
    ]
    assert descriptor["graph_filename"] == "recurrent-32.onnx"
    assert descriptor["conv_state_shape"] == [64, 1, 5376, 4]
    assert descriptor["ssm_state_shape"] == [64, 1, 80, 64, 128]
    assert descriptor["single_weight_graph"] is True
    assert descriptor["parallel_prefill_ready"] is True
    assert descriptor["production_activation_authorized"] is False
    assert len(descriptor["runtime_id"]) == 64


def test_g06_rejects_old_three_input_contract() -> None:
    manifest = _manifest()
    manifest["inputs"] = [
        "input_ids",
        "conv_state",
        "ssm_state",
    ]
    try:
        compile_g06_runtime_descriptor(manifest)
    except ValueError as exc:
        assert "input contract" in str(exc)
    else:
        raise AssertionError("old F1 three-input contract must be rejected")


def test_g06_rejects_production_self_authorization() -> None:
    manifest = _manifest()
    manifest["production_activation_authorized"] = True
    try:
        compile_g06_runtime_descriptor(manifest)
    except ValueError as exc:
        assert "self-authorize" in str(exc)
    else:
        raise AssertionError("G0.5 must not self-authorize G0.6")
