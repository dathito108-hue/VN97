import hashlib
import json
import os
from pathlib import Path

import pytest

from vn97.r2.mamba2_g03_capsule import (
    TOKENIZER_FILES,
    VN97_MAMBA2_G03_CAPSULE_MANIFEST,
    VN97_MAMBA2_G03_CONFIG_RELATIVE,
    VN97_MAMBA2_G03_SOURCE_RECEIPT,
    VN97_MAMBA2_G03_TRANSFER_MANIFEST,
    VN97_MAMBA2_G03_WEIGHT_RELATIVE,
    VN97Mamba2G03CapsuleManifest,
    VN97Mamba2G03Error,
    _hardlink_verified,
    verify_g03_capsule_structure,
)
from vn97.r2.mamba2_source_integrity import (
    PINNED_SOURCE_REVISION,
    PINNED_WEIGHT_SHA256,
    PINNED_WEIGHT_SIZE_BYTES,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _official_config_bytes() -> bytes:
    payload = {
        "d_model": 2560,
        "d_intermediate": 0,
        "n_layer": 64,
        "vocab_size": 50277,
        "ssm_cfg": {"layer": "Mamba2"},
        "attn_layer_idx": [],
        "attn_cfg": {},
        "rms_norm": True,
        "residual_in_fp32": True,
        "fused_add_norm": True,
        "pad_vocab_size_multiple": 16,
        "tie_embeddings": True,
    }
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def _write_structural_capsule(root: Path) -> VN97Mamba2G03CapsuleManifest:
    (root / "weights").mkdir(parents=True)
    (root / "source").mkdir()
    (root / "tokenizer").mkdir()

    weight = root / VN97_MAMBA2_G03_WEIGHT_RELATIVE
    with weight.open("wb") as handle:
        handle.truncate(PINNED_WEIGHT_SIZE_BYTES)

    config = _official_config_bytes()
    (root / VN97_MAMBA2_G03_CONFIG_RELATIVE).write_bytes(config)

    source_receipt = b'{"schema":"fixture-source"}\n'
    transfer = b'{"schema":"fixture-transfer"}\n'
    (root / VN97_MAMBA2_G03_SOURCE_RECEIPT).write_bytes(source_receipt)
    (root / VN97_MAMBA2_G03_TRANSFER_MANIFEST).write_bytes(transfer)

    inventory = []
    for index, name in enumerate(TOKENIZER_FILES):
        data = f"tokenizer-fixture-{index}-{name}".encode()
        (root / "tokenizer" / name).write_bytes(data)
        inventory.append((name, len(data), _sha(data)))

    manifest = VN97Mamba2G03CapsuleManifest(
        source_revision=PINNED_SOURCE_REVISION,
        source_weight_sha256=PINNED_WEIGHT_SHA256,
        source_weight_size_bytes=PINNED_WEIGHT_SIZE_BYTES,
        source_receipt_sha256=_sha(source_receipt),
        transfer_manifest_sha256=_sha(transfer),
        config_sha256=_sha(config),
        tokenizer_files=tuple(inventory),
    )
    payload = {
        **manifest.canonical_object(),
        "capsule_id": manifest.capsule_id(),
    }
    (root / VN97_MAMBA2_G03_CAPSULE_MANIFEST).write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    return manifest


def test_zero_copy_hardlink_survives_source_unlink(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    target = tmp_path / "capsule" / "weights.bin"
    payload = b"VN97-MAMBA2-ZERO-COPY" * 97
    source.write_bytes(payload)

    _hardlink_verified(source, target)
    source_info = os.stat(source)
    target_info = os.stat(target)
    assert source_info.st_dev == target_info.st_dev
    assert source_info.st_ino == target_info.st_ino

    source.unlink()
    assert target.read_bytes() == payload


def test_structural_capsule_verifies_without_reading_sparse_5gb_payload(
    tmp_path: Path,
) -> None:
    root = tmp_path / "capsule"
    expected = _write_structural_capsule(root)

    verified = verify_g03_capsule_structure(
        root,
        verify_large_weight_sha256=False,
    )
    assert verified.manifest.capsule_id() == expected.capsule_id()
    assert verified.spec.d_model == 2560
    assert verified.spec.n_layers == 64
    assert verified.spec.n_heads == 80
    assert verified.weight_path.stat().st_size == PINNED_WEIGHT_SIZE_BYTES


def test_structural_capsule_fails_closed_on_sidecar_tamper(
    tmp_path: Path,
) -> None:
    root = tmp_path / "capsule"
    _write_structural_capsule(root)
    with (root / VN97_MAMBA2_G03_TRANSFER_MANIFEST).open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(VN97Mamba2G03Error, match="transfer manifest hash"):
        verify_g03_capsule_structure(
            root,
            verify_large_weight_sha256=False,
        )


def test_manifest_forbids_runtime_dependency() -> None:
    inventory = tuple(
        (name, 1, "a" * 64)
        for name in TOKENIZER_FILES
    )
    with pytest.raises(ValueError, match="must not require a Mamba runtime"):
        VN97Mamba2G03CapsuleManifest(
            source_revision=PINNED_SOURCE_REVISION,
            source_weight_sha256=PINNED_WEIGHT_SHA256,
            source_weight_size_bytes=PINNED_WEIGHT_SIZE_BYTES,
            source_receipt_sha256="b" * 64,
            transfer_manifest_sha256="c" * 64,
            config_sha256="d" * 64,
            tokenizer_files=inventory,
            source_runtime_required=True,
        )
