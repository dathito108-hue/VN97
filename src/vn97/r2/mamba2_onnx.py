from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn

from .mamba2_g03_capsule import (
    VN97Mamba2LoadedCapsule,
    load_g03_capsule,
)
from .mamba2_ssd_reference import (
    Mamba2LayerReferenceState,
    Mamba2ReferenceConfig,
    initial_layer_state,
    mamba2_mixer_step_ref,
)
from .mamba2_source_integrity import PINNED_WEIGHT_SHA256


VN97_MAMBA2_ONNX_SCHEMA = "VN97M2ONNX1"
VN97_MAMBA2_ONNX_OPSET = 18
VN97_MAMBA2_ONNX_INPUTS = (
    "input_ids",
    "conv_state",
    "ssm_state",
)
VN97_MAMBA2_ONNX_OUTPUTS = (
    "logits",
    "next_conv_state",
    "next_ssm_state",
)
VN97_MAMBA2_ONNX_SUPPORTED_CHUNKS = (8, 16, 32, 64)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Mamba2OnnxStateContract:
    n_layers: int
    conv_dim: int
    d_conv: int
    n_heads: int
    head_dim: int
    d_state: int
    state_dtype: str = "float32"
    token_dtype: str = "int64"

    def __post_init__(self) -> None:
        expected = (64, 5376, 4, 80, 64, 128)
        actual = (
            self.n_layers,
            self.conv_dim,
            self.d_conv,
            self.n_heads,
            self.head_dim,
            self.d_state,
        )
        if actual != expected:
            raise ValueError(
                f"Mamba-2 2.7B ONNX state contract drift: {actual} != {expected}"
            )
        if self.state_dtype != "float32":
            raise ValueError("Mamba-2 ONNX state dtype must be float32")
        if self.token_dtype != "int64":
            raise ValueError("Mamba-2 ONNX token dtype must be int64")

    def canonical_object(self) -> dict[str, object]:
        return {
            "n_layers": self.n_layers,
            "conv_state": {
                "dtype": self.state_dtype,
                "shape": [
                    self.n_layers,
                    "batch",
                    self.conv_dim,
                    self.d_conv,
                ],
            },
            "ssm_state": {
                "dtype": self.state_dtype,
                "shape": [
                    self.n_layers,
                    "batch",
                    self.n_heads,
                    self.head_dim,
                    self.d_state,
                ],
            },
            "token_dtype": self.token_dtype,
            "state_is_explicit": True,
            "state_carry_semantics": "exact_between_graph_invocations",
            "same_weights_semantics": True,
        }


def official_27b_state_contract() -> Mamba2OnnxStateContract:
    return Mamba2OnnxStateContract(
        n_layers=64,
        conv_dim=5376,
        d_conv=4,
        n_heads=80,
        head_dim=64,
        d_state=128,
    )


def stack_mamba2_states(
    states: Sequence[Mamba2LayerReferenceState],
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(states) != 64:
        raise ValueError("Mamba-2 state list must contain exactly 64 layers")
    conv = torch.stack([item.conv for item in states], dim=0)
    ssm = torch.stack([item.ssm for item in states], dim=0)
    if conv.ndim != 4:
        raise ValueError("stacked Mamba-2 conv state must be rank-4")
    if ssm.ndim != 5:
        raise ValueError("stacked Mamba-2 SSD state must be rank-5")
    return conv, ssm


def unstack_mamba2_states(
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
) -> tuple[Mamba2LayerReferenceState, ...]:
    contract = official_27b_state_contract()
    if conv_state.ndim != 4 or ssm_state.ndim != 5:
        raise ValueError("Mamba-2 stacked state rank mismatch")
    if tuple(conv_state.shape[0:1] + conv_state.shape[2:]) != (
        contract.n_layers,
        contract.conv_dim,
        contract.d_conv,
    ):
        raise ValueError("Mamba-2 stacked conv state shape mismatch")
    if tuple(ssm_state.shape[0:1] + ssm_state.shape[2:]) != (
        contract.n_layers,
        contract.n_heads,
        contract.head_dim,
        contract.d_state,
    ):
        raise ValueError("Mamba-2 stacked SSD state shape mismatch")
    if conv_state.shape[1] != ssm_state.shape[1]:
        raise ValueError("Mamba-2 state batch dimensions differ")
    return tuple(
        Mamba2LayerReferenceState(
            conv=conv_state[index],
            ssm=ssm_state[index],
        )
        for index in range(contract.n_layers)
    )


def initial_official_27b_state(
    batch_size: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    config = Mamba2ReferenceConfig(d_model=2560)
    states = tuple(
        initial_layer_state(
            config,
            batch_size,
            device=device,
            dtype=dtype,
        )
        for _ in range(64)
    )
    return stack_mamba2_states(states)


@dataclass(frozen=True)
class Mamba2OnnxGraphRecord:
    filename: str
    kind: str
    sequence_length: int
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        if self.kind not in {"step", "chunk"}:
            raise ValueError("Mamba-2 ONNX graph kind must be step or chunk")
        expected = (
            "step.onnx"
            if self.kind == "step"
            else f"chunk-{self.sequence_length}.onnx"
        )
        if self.filename != expected:
            raise ValueError("Mamba-2 ONNX graph filename mismatch")
        if self.sequence_length <= 0 or self.bytes <= 0:
            raise ValueError("Mamba-2 ONNX graph dimensions must be positive")
        if len(self.sha256) != 64:
            raise ValueError("Mamba-2 ONNX graph SHA-256 is invalid")

    def canonical_object(self) -> dict[str, object]:
        return {
            "filename": self.filename,
            "kind": self.kind,
            "sequence_length": self.sequence_length,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "inputs": list(VN97_MAMBA2_ONNX_INPUTS),
            "outputs": list(VN97_MAMBA2_ONNX_OUTPUTS),
            "batch_axis_dynamic": True,
            "sequence_axis_dynamic": False,
        }


def build_mamba2_onnx_manifest(
    *,
    capsule_id: str,
    capsule_manifest_sha256: str,
    graphs: Sequence[Mamba2OnnxGraphRecord],
    chunks: Sequence[int],
) -> dict[str, object]:
    if len(capsule_id) != 64 or len(capsule_manifest_sha256) != 64:
        raise ValueError("Mamba-2 ONNX capsule identity is invalid")
    resolved_chunks = tuple(sorted(set(int(value) for value in chunks)))
    if not resolved_chunks or any(
        value not in VN97_MAMBA2_ONNX_SUPPORTED_CHUNKS
        for value in resolved_chunks
    ):
        raise ValueError("unsupported Mamba-2 ONNX chunk sizes")
    graph_list = tuple(graphs)
    names = {item.filename for item in graph_list}
    if "step.onnx" not in names:
        raise ValueError("Mamba-2 ONNX manifest requires step.onnx")
    expected_chunks = {f"chunk-{value}.onnx" for value in resolved_chunks}
    if not expected_chunks.issubset(names):
        raise ValueError("Mamba-2 ONNX manifest lacks required chunk graphs")

    body: dict[str, object] = {
        "schema": VN97_MAMBA2_ONNX_SCHEMA,
        "architecture": "VN97-MAMBA2-G0",
        "source_weight_sha256": PINNED_WEIGHT_SHA256,
        "capsule_id": capsule_id,
        "capsule_manifest_sha256": capsule_manifest_sha256,
        "opset": VN97_MAMBA2_ONNX_OPSET,
        "state_contract": official_27b_state_contract().canonical_object(),
        "supported_chunk_sizes": list(resolved_chunks),
        "graphs": [
            item.canonical_object()
            for item in sorted(
                graph_list,
                key=lambda item: (
                    0 if item.kind == "step" else 1,
                    item.sequence_length,
                ),
            )
        ],
        "same_weights_semantics": True,
        "quantization_used": False,
        "augmentation_effect": "exact_zero",
        "production_activation_authorized": False,
        "source_runtime_required": False,
    }
    return {
        **body,
        "bundle_id": _sha256_bytes(
            b"VN97M2ONNX1\0" + _canonical_json(body)
        ),
    }


class Mamba2TinyMixerOnnxAdapter(nn.Module):
    """Tiny shape-reduced oracle adapter for export mechanics only.

    It exercises the same equations and explicit conv/SSD state topology as
    the production Mamba-2 reference without allocating the 2.7B model.
    """

    def __init__(
        self,
        tensors: Mapping[str, torch.Tensor],
        config: Mamba2ReferenceConfig,
    ) -> None:
        super().__init__()
        self.config = config
        for name, tensor in tensors.items():
            safe = name.replace(".", "__")
            self.register_buffer(safe, tensor.detach().clone())

    def _tensors(self) -> dict[str, torch.Tensor]:
        return {
            name.replace("__", "."): tensor
            for name, tensor in self.named_buffers()
        }

    def forward(
        self,
        hidden: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out, next_state = mamba2_mixer_step_ref(
            hidden,
            Mamba2LayerReferenceState(
                conv=conv_state,
                ssm=ssm_state,
            ),
            self._tensors(),
            self.config,
        )
        return out, next_state.conv, next_state.ssm


def export_tiny_mamba2_step_graph(
    *,
    tensors: Mapping[str, torch.Tensor],
    config: Mamba2ReferenceConfig,
    output_path: Path,
    batch_size: int = 1,
) -> Mamba2OnnxGraphRecord:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adapter = Mamba2TinyMixerOnnxAdapter(tensors, config).eval()
    state = initial_layer_state(
        config,
        batch_size,
        device="cpu",
        dtype=torch.float32,
    )
    hidden = torch.zeros(batch_size, config.d_model, dtype=torch.float32)
    temp = output_path.with_name(output_path.name + ".tmp")
    temp.unlink(missing_ok=True)
    torch.onnx.export(
        adapter,
        (hidden, state.conv, state.ssm),
        temp,
        export_params=True,
        opset_version=VN97_MAMBA2_ONNX_OPSET,
        do_constant_folding=True,
        input_names=["hidden", "conv_state", "ssm_state"],
        output_names=["out", "next_conv_state", "next_ssm_state"],
        dynamic_axes={
            "hidden": {0: "batch"},
            "conv_state": {0: "batch"},
            "ssm_state": {0: "batch"},
            "out": {0: "batch"},
            "next_conv_state": {0: "batch"},
            "next_ssm_state": {0: "batch"},
        },
        dynamo=False,
    )
    temp.replace(output_path)
    return Mamba2OnnxGraphRecord(
        filename=output_path.name,
        kind="step",
        sequence_length=1,
        sha256=_sha256_file(output_path),
        bytes=output_path.stat().st_size,
    )


def capsule_identity_for_onnx(
    capsule: VN97Mamba2LoadedCapsule,
) -> tuple[str, str]:
    return (
        capsule.manifest.capsule_id(),
        capsule.manifest_sha256,
    )


def load_capsule_for_onnx(root: Path) -> VN97Mamba2LoadedCapsule:
    return load_g03_capsule(
        root,
        verify_large_weight_sha256=True,
    )
