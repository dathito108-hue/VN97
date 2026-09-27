from __future__ import annotations

import hashlib
import json
from pathlib import Path

from vn97.r2.turnkey_assets import (
    R2F6_APK_SCHEMA,
    R2F6_INDEX_FILENAME,
)


def test_f6_schema_and_index_filename_are_locked() -> None:
    assert R2F6_APK_SCHEMA == "VN97R2APK1"
    assert R2F6_INDEX_FILENAME == "assets.vn97r2apk1.json"


def test_f6_source_locks_no_legacy_fallback() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/vn97/r2/turnkey_assets.py"
    ).read_text(encoding="utf-8")
    assert '"legacy_inference_fallback": False' in source
    assert "expected_checkpoint_sha256" in source
    assert "tokenizer_model_sha256" in source
    assert "graph_names" in source
