from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .assistant_bridge import compile_r2f2_binding
from .onnx_autotune import verify_r2e4_profile
from .onnx_export import verify_r2_onnx_bundle
from .onnx_runtime_package import (
    R2F1_RUNTIME_FILENAME,
    verify_r2f1_runtime_package,
)

R2F6_APK_SCHEMA = "VN97R2APK1"
R2F6_INDEX_FILENAME = "assets.vn97r2apk1.json"
R2F6_TUNING_FILENAME = "autotune.vn97r2e4.json"
R2F6_BINDING_FILENAME = "assistant.vn97r2f2.json"
R2F6_MAX_FILE_BYTES = 768 * 1024 * 1024
R2F6_MAX_TOTAL_BYTES = 1536 * 1024 * 1024


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _safe_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    size = path.stat().st_size
    if not 0 < size <= R2F6_MAX_FILE_BYTES:
        raise ValueError(f"{label} byte size is outside F6 bounds")
    return path


def materialize_r2_turnkey_assets(
    *,
    source_bundle_dir: Path,
    tuning_profile_path: Path,
    output_dir: Path,
    expected_checkpoint_sha256: str,
    tokenizer_model_sha256: str,
) -> dict[str, object]:
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("R2 F6 output directory must not already exist")
    source = source_bundle_dir.resolve(strict=True)
    if source_bundle_dir.is_symlink() or not source.is_dir():
        raise ValueError("R2 F6 source bundle must be a real directory")

    manifest = verify_r2_onnx_bundle(source)
    if manifest.get("checkpoint_sha256") != expected_checkpoint_sha256:
        raise ValueError(
            "R2 F6 ONNX checkpoint differs from release candidate"
        )
    runtime = verify_r2f1_runtime_package(
        source / R2F1_RUNTIME_FILENAME,
        expected_bundle_id=str(manifest["bundle_id"]),
    )
    tuning = verify_r2e4_profile(
        tuning_profile_path,
        expected_bundle_id=str(manifest["bundle_id"]),
    )
    if runtime.get("tuning_id") != tuning.get("tuning_id"):
        raise ValueError("R2 F6 runtime and E4 tuning identities differ")

    binding = compile_r2f2_binding(
        manifest,
        runtime,
        tokenizer_model_sha256=tokenizer_model_sha256,
    )
    graph_records = runtime.get("graphs")
    if not isinstance(graph_records, list) or len(graph_records) < 2:
        raise ValueError("R2 F6 requires the F1 step + chunk graph set")
    graph_names = [str(item["filename"]) for item in graph_records]
    if len(graph_names) != len(set(graph_names)):
        raise ValueError("R2 F6 duplicate graph filename")

    planned = [
        R2F1_RUNTIME_FILENAME,
        R2F6_TUNING_FILENAME,
        R2F6_BINDING_FILENAME,
        *graph_names,
    ]
    if len(planned) != len(set(planned)):
        raise ValueError("R2 F6 asset filename collision")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".vn97-r2-assets-",
            dir=output_dir.parent,
        )
    )
    try:
        shutil.copyfile(
            _safe_file(
                source / R2F1_RUNTIME_FILENAME,
                label="R2 F1 runtime descriptor",
            ),
            staging / R2F1_RUNTIME_FILENAME,
        )
        shutil.copyfile(
            _safe_file(
                tuning_profile_path,
                label="R2 E4 tuning profile",
            ),
            staging / R2F6_TUNING_FILENAME,
        )
        (
            staging / R2F6_BINDING_FILENAME
        ).write_bytes(_canonical_json(binding) + b"\n")
        for name in graph_names:
            if Path(name).name != name:
                raise ValueError("R2 F6 graph filename is unsafe")
            shutil.copyfile(
                _safe_file(
                    source / name,
                    label=f"R2 graph {name}",
                ),
                staging / name,
            )

        records: list[dict[str, object]] = []
        total = 0
        for name in planned:
            path = _safe_file(staging / name, label=f"R2 asset {name}")
            size = path.stat().st_size
            total += size
            records.append(
                {
                    "bytes": size,
                    "name": name,
                    "sha256": _sha256_file(path),
                }
            )
        if total > R2F6_MAX_TOTAL_BYTES:
            raise ValueError("R2 F6 total asset bytes exceed APK bound")
        records.sort(key=lambda item: str(item["name"]))

        body: dict[str, object] = {
            "schema": R2F6_APK_SCHEMA,
            "binding_id": binding["binding_id"],
            "runtime_id": runtime["runtime_id"],
            "bundle_id": runtime["bundle_id"],
            "checkpoint_sha256": expected_checkpoint_sha256,
            "tokenizer_model_sha256": tokenizer_model_sha256,
            "tuning_id": tuning["tuning_id"],
            "files": records,
            "total_bytes": total,
            "legacy_inference_fallback": False,
        }
        result = dict(body)
        result["package_id"] = _sha256_bytes(
            b"VN97R2APK1\0" + _canonical_json(body)
        )
        (
            staging / R2F6_INDEX_FILENAME
        ).write_bytes(_canonical_json(result) + b"\n")

        for path in staging.iterdir():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        os.replace(staging, output_dir)
        staging = Path()
        return result
    finally:
        if staging and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
