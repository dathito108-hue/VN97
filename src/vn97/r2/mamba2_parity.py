from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch


VN97_MAMBA2_PARITY_SCHEMA = "VN97M2G02PARITY1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{label} must be lowercase SHA-256")


@dataclass(frozen=True)
class Mamba2TensorParity:
    name: str
    shape: tuple[int, ...]
    max_abs_error: float
    mean_abs_error: float
    max_rel_error: float
    atol: float
    rtol: float
    passed: bool

    def canonical_object(self) -> dict[str, object]:
        body = asdict(self)
        body["shape"] = list(self.shape)
        return body


def compare_tensor(
    name: str,
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> Mamba2TensorParity:
    if not name or "\x00" in name:
        raise ValueError("parity tensor name is invalid")
    if source.shape != target.shape:
        raise ValueError(
            f"parity shape mismatch for {name}: "
            f"{tuple(source.shape)} != {tuple(target.shape)}"
        )
    if atol < 0.0 or rtol < 0.0:
        raise ValueError("parity tolerances cannot be negative")
    if not torch.isfinite(source).all() or not torch.isfinite(target).all():
        raise ValueError(f"non-finite parity tensor: {name}")

    source_f = source.detach().float().cpu()
    target_f = target.detach().float().cpu()
    diff = (source_f - target_f).abs()
    denom = source_f.abs().clamp_min(1e-12)
    rel = diff / denom
    passed = bool(
        torch.allclose(
            source_f,
            target_f,
            atol=atol,
            rtol=rtol,
        )
    )
    return Mamba2TensorParity(
        name=name,
        shape=tuple(int(value) for value in source.shape),
        max_abs_error=float(diff.max().item()) if diff.numel() else 0.0,
        mean_abs_error=float(diff.mean().item()) if diff.numel() else 0.0,
        max_rel_error=float(rel.max().item()) if rel.numel() else 0.0,
        atol=float(atol),
        rtol=float(rtol),
        passed=passed,
    )


@dataclass(frozen=True)
class Mamba2ParityReceipt:
    source_weight_sha256: str
    vn97_checkpoint_sha256: str
    token_probe_sha256: str
    precision: str
    implementation_source: str
    comparisons: tuple[Mamba2TensorParity, ...]
    source_generated_tokens_sha256: str
    vn97_generated_tokens_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.source_weight_sha256, "source_weight_sha256")
        _require_sha256(self.vn97_checkpoint_sha256, "vn97_checkpoint_sha256")
        _require_sha256(self.token_probe_sha256, "token_probe_sha256")
        _require_sha256(
            self.source_generated_tokens_sha256,
            "source_generated_tokens_sha256",
        )
        _require_sha256(
            self.vn97_generated_tokens_sha256,
            "vn97_generated_tokens_sha256",
        )
        if not self.precision or not self.implementation_source:
            raise ValueError("parity precision/source must be non-empty")
        if not self.comparisons:
            raise ValueError("parity receipt must contain comparisons")

    @property
    def passed(self) -> bool:
        return (
            all(item.passed for item in self.comparisons)
            and self.source_generated_tokens_sha256
            == self.vn97_generated_tokens_sha256
        )

    def canonical_object(self) -> dict[str, object]:
        return {
            "schema": VN97_MAMBA2_PARITY_SCHEMA,
            "source_weight_sha256": self.source_weight_sha256,
            "vn97_checkpoint_sha256": self.vn97_checkpoint_sha256,
            "token_probe_sha256": self.token_probe_sha256,
            "precision": self.precision,
            "implementation_source": self.implementation_source,
            "comparisons": [
                item.canonical_object()
                for item in self.comparisons
            ],
            "source_generated_tokens_sha256":
                self.source_generated_tokens_sha256,
            "vn97_generated_tokens_sha256":
                self.vn97_generated_tokens_sha256,
            "passed": self.passed,
        }

    def receipt_id(self) -> str:
        return hashlib.sha256(
            b"VN97M2G02PARITY1\0"
            + _canonical_json(self.canonical_object())
        ).hexdigest()


def hash_token_probe(tokens: Sequence[int]) -> str:
    if not tokens:
        raise ValueError("token probe must not be empty")
    payload = bytearray()
    for token in tokens:
        if token < 0 or token > 0x7FFF_FFFF:
            raise ValueError("token id outside signed 31-bit range")
        payload.extend(int(token).to_bytes(4, "little", signed=False))
    return hashlib.sha256(
        b"VN97M2TOKENS1\0" + bytes(payload)
    ).hexdigest()


def hash_generated_tokens(tokens: Sequence[int]) -> str:
    if not tokens:
        raise ValueError("generated token sequence must not be empty")
    payload = bytearray()
    for token in tokens:
        if token < 0 or token > 0x7FFF_FFFF:
            raise ValueError("generated token id outside signed 31-bit range")
        payload.extend(int(token).to_bytes(4, "little", signed=False))
    return hashlib.sha256(
        b"VN97M2GEN1\0" + bytes(payload)
    ).hexdigest()


def require_production_parity(receipt: Mamba2ParityReceipt) -> None:
    required_prefixes = (
        "layer0.hidden",
        "layer0.conv_state",
        "layer0.ssm_state",
        "final.logits",
    )
    names = {item.name for item in receipt.comparisons}
    missing = [
        prefix
        for prefix in required_prefixes
        if prefix not in names
    ]
    if missing:
        raise ValueError(
            "parity receipt lacks required comparisons: "
            + ", ".join(missing)
        )
    if not receipt.passed:
        failed = [
            item.name
            for item in receipt.comparisons
            if not item.passed
        ]
        if (
            receipt.source_generated_tokens_sha256
            != receipt.vn97_generated_tokens_sha256
        ):
            failed.append("generated_tokens")
        raise ValueError(
            "Mamba-2 production parity failed: "
            + ", ".join(failed)
        )


def write_parity_receipt(
    path: str | Path,
    receipt: Mamba2ParityReceipt,
) -> None:
    require_production_parity(receipt)
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


def load_trace(path: str | Path) -> Mapping[str, torch.Tensor]:
    payload = torch.load(
        Path(path).resolve(strict=True),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, Mapping):
        raise ValueError("parity trace must be a tensor mapping")
    if not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in payload.items()
    ):
        raise ValueError("parity trace contains non-tensor entries")
    return payload
