from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch

from .mamba2_transfer import (
    Mamba2SourceSpec,
    VN97_MAMBA2_SOURCE_MODEL_ID,
    VN97_MAMBA2_SOURCE_TOKENIZER_ID,
    VN97_MAMBA2_SOURCE_TOKENIZER_REVISION,
    sha256_file,
    validate_source_state_dict,
)


VN97_MAMBA2_SOURCE_RECEIPT_SCHEMA = "VN97M2SOURCE1"

# Official state-spaces/mamba2-2.7b model-release commit.
PINNED_SOURCE_REVISION = "99b226cc377d131cccc610ed4346db564f381f1e"
PINNED_WEIGHT_SHA256 = (
    "254d89bfef6dd9f44c8e3f54e0730af3f10c482969117cb37fb33486348b00be"
)
PINNED_WEIGHT_SIZE_BYTES = 5_405_424_282
PINNED_WEIGHT_FILENAME = "pytorch_model.bin"
PINNED_CONFIG_FILENAME = "config.json"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


@dataclass(frozen=True)
class Mamba2SourceReceipt:
    schema: str
    source_model_id: str
    source_revision: str
    tokenizer_model_id: str
    tokenizer_revision: str
    config_sha256: str
    weight_sha256: str
    weight_size_bytes: int
    tensor_count: int
    unique_core_parameters: int
    mmap_used: bool

    def canonical_object(self) -> dict[str, object]:
        return asdict(self)

    def receipt_id(self) -> str:
        return hashlib.sha256(
            b"VN97M2SOURCE1\0" + _canonical_json(self.canonical_object())
        ).hexdigest()


def require_pinned_revision(source_revision: str) -> None:
    if source_revision != PINNED_SOURCE_REVISION:
        raise ValueError(
            "Mamba-2 source revision is not pinned to the official model "
            f"release: got {source_revision!r}"
        )


def load_source_config(root: str | Path) -> tuple[Mamba2SourceSpec, Path]:
    resolved = Path(root).resolve(strict=True)
    path = resolved / PINNED_CONFIG_FILENAME
    raw = path.read_text(encoding="utf-8")
    if "\x00" in raw:
        raise ValueError("Mamba-2 config contains NUL")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Mamba-2 config must be a JSON object")
    spec = Mamba2SourceSpec.from_config(payload)
    spec.require_official_27b_contract()
    return spec, path


def load_mmap_state_dict(
    root: str | Path,
) -> tuple[Mapping[str, torch.Tensor], Path]:
    """Memory-map the official monolithic torch checkpoint.

    The source release is one ~5.4 GB pytorch_model.bin. mmap=True keeps
    storages file-backed on CPU and avoids eagerly materializing a second full
    copy before namespace conversion.
    """

    resolved = Path(root).resolve(strict=True)
    path = resolved / PINNED_WEIGHT_FILENAME
    if not path.is_file():
        raise ValueError(
            f"missing official Mamba-2 weight file: {PINNED_WEIGHT_FILENAME}"
        )
    size = path.stat().st_size
    if size != PINNED_WEIGHT_SIZE_BYTES:
        raise ValueError(
            "Mamba-2 weight size mismatch: "
            f"got {size}, expected {PINNED_WEIGHT_SIZE_BYTES}"
        )
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(payload, Mapping):
        raise ValueError("Mamba-2 checkpoint is not a state_dict mapping")
    if not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in payload.items()
    ):
        raise ValueError("Mamba-2 state_dict contains non-tensor entries")
    return payload, path


def inspect_pinned_source(
    root: str | Path,
    *,
    source_revision: str,
    verify_weight_sha256: bool = True,
) -> tuple[Mamba2SourceSpec, Mapping[str, torch.Tensor], Mamba2SourceReceipt]:
    require_pinned_revision(source_revision)
    spec, config_path = load_source_config(root)
    state, weight_path = load_mmap_state_dict(root)
    validate_source_state_dict(state, spec)

    weight_sha = (
        sha256_file(weight_path)
        if verify_weight_sha256
        else PINNED_WEIGHT_SHA256
    )
    if weight_sha != PINNED_WEIGHT_SHA256:
        raise ValueError(
            "Mamba-2 weight SHA-256 does not match the pinned official release"
        )

    from .mamba2_transfer import expected_unique_parameter_count

    receipt = Mamba2SourceReceipt(
        schema=VN97_MAMBA2_SOURCE_RECEIPT_SCHEMA,
        source_model_id=VN97_MAMBA2_SOURCE_MODEL_ID,
        source_revision=source_revision,
        tokenizer_model_id=VN97_MAMBA2_SOURCE_TOKENIZER_ID,
        tokenizer_revision=VN97_MAMBA2_SOURCE_TOKENIZER_REVISION,
        config_sha256=sha256_file(config_path),
        weight_sha256=weight_sha,
        weight_size_bytes=weight_path.stat().st_size,
        tensor_count=len(state),
        unique_core_parameters=expected_unique_parameter_count(spec),
        mmap_used=True,
    )
    return spec, state, receipt


def write_source_receipt(
    path: str | Path,
    receipt: Mamba2SourceReceipt,
) -> None:
    if receipt.schema != VN97_MAMBA2_SOURCE_RECEIPT_SCHEMA:
        raise ValueError("wrong source receipt schema")
    payload = {
        **receipt.canonical_object(),
        "receipt_id": receipt.receipt_id(),
    }
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temp = resolved.with_name(resolved.name + ".tmp")
    temp.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="ascii",
    )
    temp.replace(resolved)
