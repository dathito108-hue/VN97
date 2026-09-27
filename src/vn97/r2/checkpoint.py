from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
from typing import Any, Iterable

import torch

from .config import R2_ARCHITECTURE_ID, VN97R2Config
from .model import VN97R2Model


R2_CHECKPOINT_SCHEMA = "VN97R2CP1"


class VN97R2CheckpointError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.unlink(missing_ok=True)
    torch.save(payload, temp)
    temp.replace(path)


def save_r2_checkpoint(
    path: str | Path,
    model: VN97R2Model,
    *,
    stage: str,
    capability_ledger: Iterable[dict[str, object]] = (),
    metadata: dict[str, object] | None = None,
) -> str:
    if not stage:
        raise ValueError("stage must be non-empty")
    resolved = Path(path)
    payload = {
        "schema": R2_CHECKPOINT_SCHEMA,
        "architecture_id": R2_ARCHITECTURE_ID,
        "config": asdict(model.config),
        "config_fingerprint": model.config.fingerprint(),
        "stage": stage,
        "capability_ledger": list(capability_ledger),
        "metadata": dict(metadata or {}),
        "state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
    }
    _atomic_torch_save(resolved, payload)
    return sha256_file(resolved)


def load_r2_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[VN97R2Model, dict[str, Any]]:
    resolved = Path(path).resolve(strict=True)
    payload = torch.load(
        resolved,
        map_location=map_location,
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise VN97R2CheckpointError("checkpoint payload must be a mapping")
    if payload.get("schema") != R2_CHECKPOINT_SCHEMA:
        raise VN97R2CheckpointError("unsupported R2 checkpoint schema")
    if payload.get("architecture_id") != R2_ARCHITECTURE_ID:
        raise VN97R2CheckpointError("R2 architecture id mismatch")

    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        raise VN97R2CheckpointError("checkpoint config missing")
    config = VN97R2Config(**config_payload)
    if payload.get("config_fingerprint") != config.fingerprint():
        raise VN97R2CheckpointError("checkpoint config fingerprint mismatch")

    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise VN97R2CheckpointError("checkpoint state_dict missing")

    model = VN97R2Model(config)
    model.load_state_dict(state, strict=True)

    ledger = payload.get("capability_ledger", [])
    if not isinstance(ledger, list):
        raise VN97R2CheckpointError("capability ledger must be a list")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise VN97R2CheckpointError("checkpoint metadata must be a mapping")
    if not isinstance(payload.get("stage"), str) or not payload["stage"]:
        raise VN97R2CheckpointError("checkpoint stage missing")

    evidence = {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "schema": payload["schema"],
        "architecture_id": payload["architecture_id"],
        "config_fingerprint": payload["config_fingerprint"],
        "stage": payload["stage"],
        "capability_ledger": ledger,
        "metadata": metadata,
    }
    return model, evidence
