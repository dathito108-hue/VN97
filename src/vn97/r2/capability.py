from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping

import torch

from .config import R2_ARCHITECTURE_ID


CAPABILITY_PACK_SCHEMA = "VN97R2CAP1"


@dataclass(frozen=True)
class CapabilityManifest:
    name: str
    version: int
    parent_checkpoint_sha256: str
    training_recipe_sha256: str
    dataset_sha256: tuple[str, ...]
    task_families: tuple[str, ...]
    tool_schemas: tuple[str, ...] = ()
    architecture_id: str = R2_ARCHITECTURE_ID
    schema: str = CAPABILITY_PACK_SCHEMA

    def __post_init__(self) -> None:
        if not self.name or any(ch.isspace() for ch in self.name):
            raise ValueError("capability name must be a non-empty stable token")
        if self.version <= 0:
            raise ValueError("capability version must be positive")
        if self.architecture_id != R2_ARCHITECTURE_ID:
            raise ValueError("capability pack architecture mismatch")
        if self.schema != CAPABILITY_PACK_SCHEMA:
            raise ValueError("unsupported capability pack schema")
        for digest in (
            self.parent_checkpoint_sha256,
            self.training_recipe_sha256,
            *self.dataset_sha256,
        ):
            if len(digest) != 64:
                raise ValueError("capability digest must be SHA-256 hex")
            int(digest, 16)
        if not self.task_families:
            raise ValueError("capability pack must declare at least one task family")

    def canonical_dict(self) -> dict[str, object]:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.canonical_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(b"VN97R2CAP\0" + payload).hexdigest()


def validate_delta(
    base_state: Mapping[str, torch.Tensor],
    delta_state: Mapping[str, torch.Tensor],
) -> None:
    if not delta_state:
        raise ValueError("capability delta must not be empty")
    unknown = sorted(set(delta_state) - set(base_state))
    if unknown:
        raise ValueError(
            "capability delta contains unknown parameters: "
            + ", ".join(unknown[:8])
        )
    for name, delta in delta_state.items():
        base = base_state[name]
        if tuple(delta.shape) != tuple(base.shape):
            raise ValueError(
                f"shape mismatch for {name}: "
                f"{tuple(delta.shape)} != {tuple(base.shape)}"
            )
        if not delta.dtype.is_floating_point:
            raise ValueError(f"delta tensor {name} must be floating point")
        if not torch.isfinite(delta).all():
            raise ValueError(f"delta tensor {name} contains non-finite values")


def merge_capability_delta(
    base_state: Mapping[str, torch.Tensor],
    delta_state: Mapping[str, torch.Tensor],
    *,
    alpha: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Bake an approved capability delta into one VN97-R2 checkpoint.

    This produces one merged state_dict. It does not introduce an adapter
    backend, model router, or runtime dependency.
    """
    if not torch.isfinite(torch.tensor(alpha)):
        raise ValueError("alpha must be finite")
    validate_delta(base_state, delta_state)

    merged: dict[str, torch.Tensor] = {}
    for name, base in base_state.items():
        delta = delta_state.get(name)
        if delta is None:
            merged[name] = base.detach().clone()
        else:
            merged[name] = (
                base.detach().to(torch.float32)
                + float(alpha) * delta.detach().to(torch.float32)
            ).to(dtype=base.dtype)
    return merged


def capability_ledger_entry(
    manifest: CapabilityManifest,
    *,
    delta_sha256: str,
    merged_checkpoint_sha256: str,
) -> dict[str, object]:
    for digest in (delta_sha256, merged_checkpoint_sha256):
        if len(digest) != 64:
            raise ValueError("ledger digest must be SHA-256 hex")
        int(digest, 16)
    return {
        "schema": CAPABILITY_PACK_SCHEMA,
        "manifest_fingerprint": manifest.fingerprint(),
        "name": manifest.name,
        "version": manifest.version,
        "delta_sha256": delta_sha256,
        "merged_checkpoint_sha256": merged_checkpoint_sha256,
        "task_families": list(manifest.task_families),
        "tool_schemas": list(manifest.tool_schemas),
    }
