from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any

import torch

from .mamba2_source_integrity import (
    PINNED_CONFIG_FILENAME,
    PINNED_SOURCE_REVISION,
    PINNED_WEIGHT_FILENAME,
    PINNED_WEIGHT_SHA256,
    PINNED_WEIGHT_SIZE_BYTES,
    inspect_pinned_source,
    write_source_receipt,
)
from .mamba2_transfer import (
    Mamba2SourceSpec,
    build_transfer_manifest,
    identity_tensor_mapping,
    validate_source_state_dict,
    write_transfer_manifest,
)


VN97_MAMBA2_G03_CAPSULE_SCHEMA = "VN97M2G03CAP1"
VN97_MAMBA2_G03_CAPSULE_MANIFEST = "capsule.vn97m2g03.json"
VN97_MAMBA2_G03_SOURCE_RECEIPT = "source.vn97m2source1.json"
VN97_MAMBA2_G03_TRANSFER_MANIFEST = "transfer.vn97m2g0transfer1.json"
VN97_MAMBA2_G03_WEIGHT_RELATIVE = "weights/pytorch_model.bin"
VN97_MAMBA2_G03_CONFIG_RELATIVE = "source/config.json"

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)

_MAX_SMALL_FILE_BYTES = 64 * 1024 * 1024


class VN97Mamba2G03Error(RuntimeError):
    pass


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
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular_file(
    path: Path,
    *,
    label: str,
    max_bytes: int | None = None,
) -> Path:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise VN97Mamba2G03Error(f"{label} is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise VN97Mamba2G03Error(
            f"{label} must be a regular non-symlink file"
        )
    if info.st_size <= 0:
        raise VN97Mamba2G03Error(f"{label} must be non-empty")
    if max_bytes is not None and info.st_size > max_bytes:
        raise VN97Mamba2G03Error(f"{label} exceeds size bound")
    return path.resolve(strict=True)


def _hardlink_verified(
    source: Path,
    target: Path,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target, follow_symlinks=False)
    except OSError as exc:
        if exc.errno == errno.EXDEV:
            raise VN97Mamba2G03Error(
                "zero-copy capsule requires source and output on the same filesystem"
            ) from exc
        raise

    source_info = os.stat(source, follow_symlinks=False)
    target_info = os.stat(target, follow_symlinks=False)
    if (
        source_info.st_dev != target_info.st_dev
        or source_info.st_ino != target_info.st_ino
        or source_info.st_size != target_info.st_size
    ):
        target.unlink(missing_ok=True)
        raise VN97Mamba2G03Error(
            "zero-copy payload is not the exact source inode"
        )


def _write_canonical_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = _canonical_json(dict(payload)) + b"\n"
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


@dataclass(frozen=True)
class VN97Mamba2G03CapsuleManifest:
    source_revision: str
    source_weight_sha256: str
    source_weight_size_bytes: int
    source_receipt_sha256: str
    transfer_manifest_sha256: str
    config_sha256: str
    tokenizer_files: tuple[tuple[str, int, str], ...]
    payload_relative_path: str = VN97_MAMBA2_G03_WEIGHT_RELATIVE
    payload_format: str = "official_pytorch_state_dict"
    logical_namespace: str = "vn97.core"
    transfer_semantics: str = "tensor_value_identity_1to1"
    zero_copy_materialization: bool = True
    source_runtime_required: bool = False

    def __post_init__(self) -> None:
        if self.source_revision != PINNED_SOURCE_REVISION:
            raise ValueError("G0.3 source revision mismatch")
        if self.source_weight_sha256 != PINNED_WEIGHT_SHA256:
            raise ValueError("G0.3 source weight identity mismatch")
        if self.source_weight_size_bytes != PINNED_WEIGHT_SIZE_BYTES:
            raise ValueError("G0.3 source weight size mismatch")
        for value, label in (
            (self.source_receipt_sha256, "source receipt SHA-256"),
            (self.transfer_manifest_sha256, "transfer manifest SHA-256"),
            (self.config_sha256, "config SHA-256"),
        ):
            if (
                len(value) != 64
                or any(ch not in "0123456789abcdef" for ch in value)
            ):
                raise ValueError(f"{label} must be lowercase SHA-256")
        expected_names = TOKENIZER_FILES
        actual_names = tuple(item[0] for item in self.tokenizer_files)
        if actual_names != expected_names:
            raise ValueError("G0.3 tokenizer file set/order mismatch")
        for name, size, digest in self.tokenizer_files:
            if "/" in name or "\\" in name or not name:
                raise ValueError("invalid tokenizer file name")
            if size <= 0 or size > _MAX_SMALL_FILE_BYTES:
                raise ValueError("invalid tokenizer file size")
            if (
                len(digest) != 64
                or any(ch not in "0123456789abcdef" for ch in digest)
            ):
                raise ValueError("invalid tokenizer file SHA-256")
        if self.payload_relative_path != VN97_MAMBA2_G03_WEIGHT_RELATIVE:
            raise ValueError("G0.3 payload path mismatch")
        if self.payload_format != "official_pytorch_state_dict":
            raise ValueError("G0.3 payload format mismatch")
        if self.logical_namespace != "vn97.core":
            raise ValueError("G0.3 logical namespace mismatch")
        if self.transfer_semantics != "tensor_value_identity_1to1":
            raise ValueError("G0.3 transfer semantics mismatch")
        if not self.zero_copy_materialization:
            raise ValueError("G0.3 must be zero-copy materialized")
        if self.source_runtime_required:
            raise ValueError("G0.3 must not require a Mamba runtime")

    def canonical_object(self) -> dict[str, object]:
        return {
            "schema": VN97_MAMBA2_G03_CAPSULE_SCHEMA,
            "source_revision": self.source_revision,
            "source_weight_sha256": self.source_weight_sha256,
            "source_weight_size_bytes": self.source_weight_size_bytes,
            "source_receipt_sha256": self.source_receipt_sha256,
            "transfer_manifest_sha256": self.transfer_manifest_sha256,
            "config_sha256": self.config_sha256,
            "tokenizer_files": [
                {"name": name, "size": size, "sha256": digest}
                for name, size, digest in self.tokenizer_files
            ],
            "payload_relative_path": self.payload_relative_path,
            "payload_format": self.payload_format,
            "logical_namespace": self.logical_namespace,
            "transfer_semantics": self.transfer_semantics,
            "zero_copy_materialization": self.zero_copy_materialization,
            "source_runtime_required": self.source_runtime_required,
        }

    def capsule_id(self) -> str:
        return hashlib.sha256(
            b"VN97M2G03CAP1\0" + _canonical_json(self.canonical_object())
        ).hexdigest()


class VN97Mamba2TensorView(Mapping[str, torch.Tensor]):
    """Logical VN97 tensor namespace over the immutable source payload.

    No tensor is cloned. The mapping only renames source keys in memory.
    """

    def __init__(
        self,
        source_state: Mapping[str, torch.Tensor],
        spec: Mamba2SourceSpec,
    ) -> None:
        validate_source_state_dict(source_state, spec)
        mapping = identity_tensor_mapping(spec)
        inverse = {target: source for source, target in mapping.items()}
        if len(inverse) != len(mapping):
            raise VN97Mamba2G03Error("logical tensor mapping is not one-to-one")
        self._source_state = source_state
        self._inverse = inverse

    def __getitem__(self, key: str) -> torch.Tensor:
        source = self._inverse[key]
        tensor = self._source_state.get(source)
        if tensor is None and source == "lm_head.weight":
            tensor = self._source_state["backbone.embedding.weight"]
        if tensor is None:
            raise KeyError(key)
        return tensor

    def __iter__(self) -> Iterator[str]:
        return iter(self._inverse)

    def __len__(self) -> int:
        return len(self._inverse)


@dataclass(frozen=True)
class VN97Mamba2LoadedCapsule:
    root: Path
    manifest: VN97Mamba2G03CapsuleManifest
    manifest_sha256: str
    spec: Mamba2SourceSpec
    tensors: VN97Mamba2TensorView


def _tokenizer_inventory(source_root: Path) -> tuple[tuple[str, int, str], ...]:
    tokenizer_root = source_root / "tokenizer"
    output: list[tuple[str, int, str]] = []
    for name in TOKENIZER_FILES:
        path = _require_regular_file(
            tokenizer_root / name,
            label=f"tokenizer/{name}",
            max_bytes=_MAX_SMALL_FILE_BYTES,
        )
        output.append((name, path.stat().st_size, _sha256_file(path)))
    return tuple(output)


def materialize_g03_capsule(
    source_root: str | Path,
    output_root: str | Path,
) -> VN97Mamba2G03CapsuleManifest:
    source = Path(source_root).resolve(strict=True)
    output = Path(output_root)
    if output.exists() or output.is_symlink():
        raise VN97Mamba2G03Error("G0.3 output root must not already exist")
    output.parent.mkdir(parents=True, exist_ok=True)

    spec, source_state, source_receipt = inspect_pinned_source(
        source,
        source_revision=PINNED_SOURCE_REVISION,
        verify_weight_sha256=True,
    )
    weight = _require_regular_file(
        source / PINNED_WEIGHT_FILENAME,
        label="Mamba-2 source weights",
    )
    config = _require_regular_file(
        source / PINNED_CONFIG_FILENAME,
        label="Mamba-2 source config",
        max_bytes=_MAX_SMALL_FILE_BYTES,
    )
    tokenizer_inventory = _tokenizer_inventory(source)

    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            suffix=".staging",
            dir=output.parent,
        )
    )
    try:
        (stage / "weights").mkdir()
        (stage / "source").mkdir()
        (stage / "tokenizer").mkdir()

        _hardlink_verified(
            weight,
            stage / VN97_MAMBA2_G03_WEIGHT_RELATIVE,
        )
        _hardlink_verified(
            config,
            stage / VN97_MAMBA2_G03_CONFIG_RELATIVE,
        )
        for name, _, _ in tokenizer_inventory:
            _hardlink_verified(
                source / "tokenizer" / name,
                stage / "tokenizer" / name,
            )

        source_receipt_path = stage / VN97_MAMBA2_G03_SOURCE_RECEIPT
        write_source_receipt(source_receipt_path, source_receipt)

        transfer = build_transfer_manifest(
            spec,
            source_revision=source_receipt.source_revision,
            source_config_sha256=source_receipt.config_sha256,
            source_weight_sha256=source_receipt.weight_sha256,
        )
        transfer_path = stage / VN97_MAMBA2_G03_TRANSFER_MANIFEST
        write_transfer_manifest(transfer_path, transfer)

        manifest = VN97Mamba2G03CapsuleManifest(
            source_revision=source_receipt.source_revision,
            source_weight_sha256=source_receipt.weight_sha256,
            source_weight_size_bytes=source_receipt.weight_size_bytes,
            source_receipt_sha256=_sha256_file(source_receipt_path),
            transfer_manifest_sha256=_sha256_file(transfer_path),
            config_sha256=source_receipt.config_sha256,
            tokenizer_files=tokenizer_inventory,
        )
        manifest_payload = {
            **manifest.canonical_object(),
            "capsule_id": manifest.capsule_id(),
        }
        manifest_path = stage / VN97_MAMBA2_G03_CAPSULE_MANIFEST
        _write_canonical_json(manifest_path, manifest_payload)

        # Bind the manifest directory contents after all files exist.
        dir_fd = os.open(stage, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        os.replace(stage, output)
        parent_fd = os.open(
            output.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _read_capsule_manifest(path: Path) -> VN97Mamba2G03CapsuleManifest:
    raw = path.read_bytes()
    if len(raw) > 256 * 1024:
        raise VN97Mamba2G03Error("G0.3 manifest exceeds size bound")
    try:
        payload = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VN97Mamba2G03Error("G0.3 manifest is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise VN97Mamba2G03Error("G0.3 manifest must be an object")
    if payload.get("schema") != VN97_MAMBA2_G03_CAPSULE_SCHEMA:
        raise VN97Mamba2G03Error("G0.3 manifest schema mismatch")
    tokenizer = payload.get("tokenizer_files")
    if not isinstance(tokenizer, list):
        raise VN97Mamba2G03Error("G0.3 tokenizer inventory missing")
    try:
        manifest = VN97Mamba2G03CapsuleManifest(
            source_revision=payload["source_revision"],
            source_weight_sha256=payload["source_weight_sha256"],
            source_weight_size_bytes=payload["source_weight_size_bytes"],
            source_receipt_sha256=payload["source_receipt_sha256"],
            transfer_manifest_sha256=payload["transfer_manifest_sha256"],
            config_sha256=payload["config_sha256"],
            tokenizer_files=tuple(
                (item["name"], item["size"], item["sha256"])
                for item in tokenizer
            ),
            payload_relative_path=payload["payload_relative_path"],
            payload_format=payload["payload_format"],
            logical_namespace=payload["logical_namespace"],
            transfer_semantics=payload["transfer_semantics"],
            zero_copy_materialization=payload["zero_copy_materialization"],
            source_runtime_required=payload["source_runtime_required"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise VN97Mamba2G03Error("invalid G0.3 manifest fields") from exc
    if payload.get("capsule_id") != manifest.capsule_id():
        raise VN97Mamba2G03Error("G0.3 capsule identity mismatch")
    return manifest


def load_g03_capsule(
    root: str | Path,
    *,
    verify_large_weight_sha256: bool = True,
) -> VN97Mamba2LoadedCapsule:
    resolved = Path(root).resolve(strict=True)
    manifest_path = _require_regular_file(
        resolved / VN97_MAMBA2_G03_CAPSULE_MANIFEST,
        label="G0.3 capsule manifest",
        max_bytes=256 * 1024,
    )
    manifest = _read_capsule_manifest(manifest_path)

    source_receipt = _require_regular_file(
        resolved / VN97_MAMBA2_G03_SOURCE_RECEIPT,
        label="G0.3 source receipt",
        max_bytes=256 * 1024,
    )
    transfer = _require_regular_file(
        resolved / VN97_MAMBA2_G03_TRANSFER_MANIFEST,
        label="G0.3 transfer manifest",
        max_bytes=2 * 1024 * 1024,
    )
    config = _require_regular_file(
        resolved / VN97_MAMBA2_G03_CONFIG_RELATIVE,
        label="G0.3 source config",
        max_bytes=_MAX_SMALL_FILE_BYTES,
    )
    weight = _require_regular_file(
        resolved / VN97_MAMBA2_G03_WEIGHT_RELATIVE,
        label="G0.3 weight payload",
    )

    if _sha256_file(source_receipt) != manifest.source_receipt_sha256:
        raise VN97Mamba2G03Error("G0.3 source receipt hash mismatch")
    if _sha256_file(transfer) != manifest.transfer_manifest_sha256:
        raise VN97Mamba2G03Error("G0.3 transfer manifest hash mismatch")
    if _sha256_file(config) != manifest.config_sha256:
        raise VN97Mamba2G03Error("G0.3 config hash mismatch")
    if weight.stat().st_size != manifest.source_weight_size_bytes:
        raise VN97Mamba2G03Error("G0.3 weight size mismatch")
    if verify_large_weight_sha256 and _sha256_file(weight) != manifest.source_weight_sha256:
        raise VN97Mamba2G03Error("G0.3 weight hash mismatch")

    for name, size, digest in manifest.tokenizer_files:
        path = _require_regular_file(
            resolved / "tokenizer" / name,
            label=f"G0.3 tokenizer/{name}",
            max_bytes=_MAX_SMALL_FILE_BYTES,
        )
        if path.stat().st_size != size or _sha256_file(path) != digest:
            raise VN97Mamba2G03Error(
                f"G0.3 tokenizer identity mismatch: {name}"
            )

    config_payload = json.loads(config.read_text(encoding="utf-8"))
    if not isinstance(config_payload, dict):
        raise VN97Mamba2G03Error("G0.3 config must be an object")
    spec = Mamba2SourceSpec.from_config(config_payload)
    spec.require_official_27b_contract()

    state = torch.load(
        weight,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(state, Mapping):
        raise VN97Mamba2G03Error("G0.3 weight payload is not a state dict")
    tensors = VN97Mamba2TensorView(state, spec)
    return VN97Mamba2LoadedCapsule(
        root=resolved,
        manifest=manifest,
        manifest_sha256=_sha256_file(manifest_path),
        spec=spec,
        tensors=tensors,
    )
