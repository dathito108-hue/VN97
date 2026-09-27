from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vn97.r2.assistant_bridge import (
    R2F2_BINDING_SCHEMA,
    compile_r2f2_binding,
    verify_r2f2_binding,
)

BUNDLE = "a" * 64
RUNTIME = "b" * 64
ARCH = "c" * 64
CHECKPOINT = "d" * 64
TOKENIZER = "e" * 64


def _manifest() -> dict[str, object]:
    return {
        "bundle_id": BUNDLE,
        "architecture_fingerprint": ARCH,
        "profile": "deep",
        "active_layers": 12,
        "checkpoint_sha256": CHECKPOINT,
        "checkpoint_stage": "validated",
        "config": {
            "vocab_size": 32000,
            "d_model": 768,
            "n_layers": 12,
            "d_state": 16,
        },
    }


def _runtime() -> dict[str, object]:
    return {
        "runtime_id": RUNTIME,
        "bundle_id": BUNDLE,
        "architecture_fingerprint": ARCH,
        "profile": "deep",
        "active_layers": 12,
        "vocab_size": 32000,
        "d_state": 16,
    }


def test_f2_binding_is_deterministic_and_disables_legacy_fallback() -> None:
    first = compile_r2f2_binding(
        _manifest(),
        _runtime(),
        tokenizer_model_sha256=TOKENIZER,
    )
    second = compile_r2f2_binding(
        _manifest(),
        _runtime(),
        tokenizer_model_sha256=TOKENIZER,
    )
    assert first == second
    assert first["schema"] == R2F2_BINDING_SCHEMA
    assert first["checkpoint_sha256"] == CHECKPOINT
    assert first["tokenizer_model_sha256"] == TOKENIZER
    assert first["legacy_inference_fallback"] is False


def test_f2_binding_rejects_missing_real_checkpoint() -> None:
    manifest = _manifest()
    manifest["checkpoint_sha256"] = None
    with pytest.raises(ValueError, match="checkpoint"):
        compile_r2f2_binding(
            manifest,
            _runtime(),
            tokenizer_model_sha256=TOKENIZER,
        )


def test_f2_binding_rejects_cross_bundle_runtime() -> None:
    runtime = _runtime()
    runtime["bundle_id"] = "f" * 64
    with pytest.raises(ValueError, match="another E2 bundle"):
        compile_r2f2_binding(
            _manifest(),
            runtime,
            tokenizer_model_sha256=TOKENIZER,
        )


def test_verify_detects_semantic_tamper(tmp_path: Path) -> None:
    payload = compile_r2f2_binding(
        _manifest(),
        _runtime(),
        tokenizer_model_sha256=TOKENIZER,
    )
    path = tmp_path / "assistant.vn97r2f2.json"
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    assert verify_r2f2_binding(
        path,
        expected_runtime_id=RUNTIME,
        expected_tokenizer_model_sha256=TOKENIZER,
    ) == payload

    payload["legacy_inference_fallback"] = True
    body = dict(payload)
    body.pop("binding_id")
    payload["binding_id"] = hashlib.sha256(
        b"VN97R2F2BIND1\0"
        + json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fallback"):
        verify_r2f2_binding(path)
