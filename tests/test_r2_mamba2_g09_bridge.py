import pytest

from vn97.r2.mamba2_cognition_bridge import compile_g09_bridge_binding


def _runtime() -> dict[str, object]:
    return {
        "schema": "VN97M2G06RUNTIME1",
        "runtime_id": "a" * 64,
        "g05_manifest_id": "b" * 64,
        "capsule_id": "c" * 64,
        "source_weight_sha256": "d" * 64,
        "vocab_size": 50_288,
        "same_weights_semantics": True,
        "quantization_used": False,
        "production_activation_authorized": False,
    }


def _tokenizer() -> dict[str, object]:
    return {
        "schema": "VN97M2G08TOK1",
        "tokenizer_id": "e" * 64,
        "capsule_id": "c" * 64,
        "token_id_space": 50_277,
        "runtime_logits_size": 50_288,
        "eos_token_id": 0,
        "same_token_ids_required": True,
        "production_activation_authorized": False,
    }


def _tuning() -> dict[str, object]:
    return {
        "schema": "VN97M2G07TUNE1",
        "tuning_id": "f" * 64,
        "runtime_id": "a" * 64,
        "profile_receipt_id": "1" * 64,
        "profile_is_device_measured": True,
        "production_activation_authorized": False,
    }


def test_g09_bridge_binds_runtime_tokenizer_tuning_without_authorizing() -> None:
    result = compile_g09_bridge_binding(
        _runtime(),
        _tokenizer(),
        _tuning(),
    )
    assert result["schema"] == "VN97M2G09BRIDGE1"
    assert result["runtime_id"] == "a" * 64
    assert result["tokenizer_id"] == "e" * 64
    assert result["tuning_id"] == "f" * 64
    assert result["capsule_id"] == "c" * 64
    assert result["token_id_space"] == 50_277
    assert result["runtime_logits_size"] == 50_288
    assert result["padded_logits_must_be_masked"] is True
    assert result["device_measured_tuning_required"] is True
    assert result["candidate_validation_only"] is True
    assert result["production_activation_authorized"] is False
    assert len(result["bridge_id"]) == 64


def test_g09_rejects_tokenizer_from_another_capsule() -> None:
    tokenizer = _tokenizer()
    tokenizer["capsule_id"] = "9" * 64
    with pytest.raises(ValueError, match="capsule identity"):
        compile_g09_bridge_binding(
            _runtime(),
            tokenizer,
            _tuning(),
        )


def test_g09_rejects_tuning_from_another_runtime() -> None:
    tuning = _tuning()
    tuning["runtime_id"] = "9" * 64
    with pytest.raises(ValueError, match="another runtime"):
        compile_g09_bridge_binding(
            _runtime(),
            _tokenizer(),
            tuning,
        )


def test_g09_rejects_non_device_measured_tuning() -> None:
    tuning = _tuning()
    tuning["profile_is_device_measured"] = False
    with pytest.raises(ValueError, match="device-measured"):
        compile_g09_bridge_binding(
            _runtime(),
            _tokenizer(),
            tuning,
        )


def test_g09_rejects_wrong_logits_width() -> None:
    tokenizer = _tokenizer()
    tokenizer["runtime_logits_size"] = 50_277
    with pytest.raises(ValueError, match="logits width"):
        compile_g09_bridge_binding(
            _runtime(),
            tokenizer,
            _tuning(),
        )
