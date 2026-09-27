from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch


VN97_MAMBA2_G0_TRANSFER_SCHEMA = "VN97M2G0TRANSFER1"
VN97_MAMBA2_G0_CHECKPOINT_SCHEMA = "VN97M2G0CP1"
VN97_MAMBA2_SOURCE_MODEL_ID = "state-spaces/mamba2-2.7b"
VN97_MAMBA2_SOURCE_LICENSE = "apache-2.0"
VN97_MAMBA2_SOURCE_TOKENIZER_ID = "EleutherAI/gpt-neox-20b"
VN97_MAMBA2_SOURCE_TOKENIZER_REVISION = (
    "364ae95407723fadd1d47b023c1efb92a4d891c3"
)

MAMBA2_27B_D_MODEL = 2560
MAMBA2_27B_N_LAYERS = 64
MAMBA2_27B_VOCAB_SIZE = 50277
MAMBA2_27B_PAD_MULTIPLE = 16
MAMBA2_27B_D_STATE = 128
MAMBA2_27B_D_CONV = 4
MAMBA2_27B_EXPAND = 2
MAMBA2_27B_HEAD_DIM = 64
MAMBA2_27B_N_GROUPS = 1


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: str, label: str) -> str:
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


@dataclass(frozen=True)
class Mamba2SourceSpec:
    d_model: int
    n_layers: int
    vocab_size: int
    pad_vocab_size_multiple: int
    d_state: int
    d_conv: int
    expand: int
    head_dim: int
    n_groups: int
    d_intermediate: int
    rms_norm: bool
    residual_in_fp32: bool
    fused_add_norm: bool
    tie_embeddings: bool
    attention_layers: tuple[int, ...]
    ssm_layer: str
    ssm_override_keys: tuple[str, ...]

    @property
    def padded_vocab_size(self) -> int:
        multiple = self.pad_vocab_size_multiple
        return ((self.vocab_size + multiple - 1) // multiple) * multiple

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    @property
    def n_heads(self) -> int:
        if self.d_inner % self.head_dim:
            raise ValueError("Mamba2 d_inner must be divisible by head_dim")
        return self.d_inner // self.head_dim

    @property
    def conv_dim(self) -> int:
        return self.d_inner + 2 * self.n_groups * self.d_state

    @property
    def in_proj_dim(self) -> int:
        return (
            2 * self.d_inner
            + 2 * self.n_groups * self.d_state
            + self.n_heads
        )

    @property
    def recurrent_ssm_state_shape(self) -> tuple[int, int, int]:
        return (self.n_heads, self.head_dim, self.d_state)

    @property
    def recurrent_conv_state_shape(self) -> tuple[int, int]:
        return (self.conv_dim, self.d_conv)

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "Mamba2SourceSpec":
        ssm_cfg = config.get("ssm_cfg", {})
        if not isinstance(ssm_cfg, Mapping):
            raise ValueError("ssm_cfg must be a mapping")
        attention = config.get("attn_layer_idx", [])
        if not isinstance(attention, list):
            raise ValueError("attn_layer_idx must be a list")
        return cls(
            d_model=int(config.get("d_model", 0)),
            n_layers=int(config.get("n_layer", 0)),
            vocab_size=int(config.get("vocab_size", 0)),
            pad_vocab_size_multiple=int(config.get("pad_vocab_size_multiple", 1)),
            d_state=int(ssm_cfg.get("d_state", MAMBA2_27B_D_STATE)),
            d_conv=int(ssm_cfg.get("d_conv", MAMBA2_27B_D_CONV)),
            expand=int(ssm_cfg.get("expand", MAMBA2_27B_EXPAND)),
            head_dim=int(ssm_cfg.get("headdim", MAMBA2_27B_HEAD_DIM)),
            n_groups=int(ssm_cfg.get("ngroups", MAMBA2_27B_N_GROUPS)),
            d_intermediate=int(config.get("d_intermediate", 0)),
            rms_norm=bool(config.get("rms_norm", False)),
            residual_in_fp32=bool(config.get("residual_in_fp32", False)),
            fused_add_norm=bool(config.get("fused_add_norm", False)),
            tie_embeddings=bool(config.get("tie_embeddings", False)),
            attention_layers=tuple(int(value) for value in attention),
            ssm_layer=str(ssm_cfg.get("layer", "")),
            ssm_override_keys=tuple(
                sorted(str(key) for key in ssm_cfg if key != "layer")
            ),
        )

    def require_official_27b_contract(self) -> None:
        expected = {
            "d_model": MAMBA2_27B_D_MODEL,
            "n_layers": MAMBA2_27B_N_LAYERS,
            "vocab_size": MAMBA2_27B_VOCAB_SIZE,
            "pad_vocab_size_multiple": MAMBA2_27B_PAD_MULTIPLE,
            "d_state": MAMBA2_27B_D_STATE,
            "d_conv": MAMBA2_27B_D_CONV,
            "expand": MAMBA2_27B_EXPAND,
            "head_dim": MAMBA2_27B_HEAD_DIM,
            "n_groups": MAMBA2_27B_N_GROUPS,
            "d_intermediate": 0,
            "rms_norm": True,
            "residual_in_fp32": True,
            "fused_add_norm": True,
            "tie_embeddings": True,
            "attention_layers": (),
            "ssm_layer": "Mamba2",
            "ssm_override_keys": (),
        }
        actual = {
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "vocab_size": self.vocab_size,
            "pad_vocab_size_multiple": self.pad_vocab_size_multiple,
            "d_state": self.d_state,
            "d_conv": self.d_conv,
            "expand": self.expand,
            "head_dim": self.head_dim,
            "n_groups": self.n_groups,
            "d_intermediate": self.d_intermediate,
            "rms_norm": self.rms_norm,
            "residual_in_fp32": self.residual_in_fp32,
            "fused_add_norm": self.fused_add_norm,
            "tie_embeddings": self.tie_embeddings,
            "attention_layers": self.attention_layers,
            "ssm_layer": self.ssm_layer,
            "ssm_override_keys": self.ssm_override_keys,
        }
        mismatches = [
            f"{key}={actual[key]!r} expected {value!r}"
            for key, value in expected.items()
            if actual[key] != value
        ]
        if mismatches:
            raise ValueError(
                "source is not the locked Mamba-2 2.7B contract: "
                + "; ".join(mismatches)
            )
        if self.padded_vocab_size != 50288:
            raise ValueError("padded vocabulary contract changed")
        if self.d_inner != 5120 or self.n_heads != 80:
            raise ValueError("Mamba2 head/inner contract changed")
        if self.conv_dim != 5376 or self.in_proj_dim != 10576:
            raise ValueError("Mamba2 projection contract changed")

    def canonical_object(self) -> dict[str, object]:
        body = asdict(self)
        body["attention_layers"] = list(self.attention_layers)
        body["ssm_override_keys"] = list(self.ssm_override_keys)
        body.update(
            {
                "padded_vocab_size": self.padded_vocab_size,
                "d_inner": self.d_inner,
                "n_heads": self.n_heads,
                "conv_dim": self.conv_dim,
                "in_proj_dim": self.in_proj_dim,
                "recurrent_ssm_state_shape": list(self.recurrent_ssm_state_shape),
                "recurrent_conv_state_shape": list(self.recurrent_conv_state_shape),
            }
        )
        return body


def expected_source_shapes(
    spec: Mamba2SourceSpec,
    *,
    include_tied_lm_head: bool = True,
) -> dict[str, tuple[int, ...]]:
    spec.require_official_27b_contract()
    shapes: dict[str, tuple[int, ...]] = {
        "backbone.embedding.weight": (
            spec.padded_vocab_size,
            spec.d_model,
        ),
        "backbone.norm_f.weight": (spec.d_model,),
    }
    if include_tied_lm_head:
        shapes["lm_head.weight"] = (
            spec.padded_vocab_size,
            spec.d_model,
        )
    for layer in range(spec.n_layers):
        prefix = f"backbone.layers.{layer}"
        mixer = f"{prefix}.mixer"
        shapes.update(
            {
                f"{prefix}.norm.weight": (spec.d_model,),
                f"{mixer}.in_proj.weight": (
                    spec.in_proj_dim,
                    spec.d_model,
                ),
                f"{mixer}.conv1d.weight": (
                    spec.conv_dim,
                    1,
                    spec.d_conv,
                ),
                f"{mixer}.conv1d.bias": (spec.conv_dim,),
                f"{mixer}.dt_bias": (spec.n_heads,),
                f"{mixer}.A_log": (spec.n_heads,),
                f"{mixer}.D": (spec.n_heads,),
                f"{mixer}.norm.weight": (spec.d_inner,),
                f"{mixer}.out_proj.weight": (
                    spec.d_model,
                    spec.d_inner,
                ),
            }
        )
    return shapes


def expected_unique_parameter_count(spec: Mamba2SourceSpec) -> int:
    shapes = expected_source_shapes(spec, include_tied_lm_head=False)
    total = 0
    for shape in shapes.values():
        count = 1
        for dim in shape:
            count *= dim
        total += count
    return total


def identity_tensor_mapping(spec: Mamba2SourceSpec) -> dict[str, str]:
    source = expected_source_shapes(spec, include_tied_lm_head=True)
    mapping: dict[str, str] = {}
    for name in source:
        if name == "backbone.embedding.weight":
            target = "vn97.core.embedding.weight"
        elif name == "backbone.norm_f.weight":
            target = "vn97.core.final_norm.weight"
        elif name == "lm_head.weight":
            target = "vn97.core.lm_head.weight"
        else:
            target = "vn97.core." + name.removeprefix("backbone.")
        mapping[name] = target
    return mapping


def validate_source_state_dict(
    tensors: Mapping[str, torch.Tensor],
    spec: Mamba2SourceSpec,
) -> None:
    expected = expected_source_shapes(spec, include_tied_lm_head=False)
    missing = [name for name in expected if name not in tensors]
    if missing:
        raise ValueError(
            "Mamba-2 source is missing required tensors: "
            + ", ".join(missing[:8])
        )
    for name, shape in expected.items():
        actual = tuple(int(value) for value in tensors[name].shape)
        if actual != shape:
            raise ValueError(
                f"Mamba-2 tensor shape mismatch for {name}: "
                f"got {actual}, expected {shape}"
            )
        if not tensors[name].is_floating_point():
            raise ValueError(f"Mamba-2 tensor {name} is not floating point")

    head = tensors.get("lm_head.weight")
    embedding = tensors["backbone.embedding.weight"]
    if head is not None:
        if tuple(head.shape) != tuple(embedding.shape):
            raise ValueError("tied LM head shape differs from embedding")
        if not torch.equal(head.detach().cpu(), embedding.detach().cpu()):
            raise ValueError(
                "Mamba-2 2.7B declares tied embeddings but LM head differs"
            )

    allowed = set(expected)
    allowed.add("lm_head.weight")
    unexpected = sorted(name for name in tensors if name not in allowed)
    if unexpected:
        raise ValueError(
            "unexpected source tensors would break exact transfer: "
            + ", ".join(unexpected[:8])
        )


def convert_state_dict_1to1(
    tensors: Mapping[str, torch.Tensor],
    spec: Mamba2SourceSpec,
) -> dict[str, torch.Tensor]:
    validate_source_state_dict(tensors, spec)
    mapping = identity_tensor_mapping(spec)
    converted: dict[str, torch.Tensor] = {}
    for source, target in mapping.items():
        tensor = tensors.get(source)
        if tensor is None and source == "lm_head.weight":
            tensor = tensors["backbone.embedding.weight"]
        if tensor is None:
            raise AssertionError(f"validated tensor disappeared: {source}")
        converted[target] = tensor
    return converted


def build_transfer_manifest(
    spec: Mamba2SourceSpec,
    *,
    source_revision: str,
    source_config_sha256: str,
    source_weight_sha256: str,
) -> dict[str, object]:
    spec.require_official_27b_contract()
    if not source_revision or "\x00" in source_revision:
        raise ValueError("source_revision must be non-empty")
    _require_sha256(source_config_sha256, "source_config_sha256")
    _require_sha256(source_weight_sha256, "source_weight_sha256")

    body: dict[str, object] = {
        "schema": VN97_MAMBA2_G0_TRANSFER_SCHEMA,
        "source_model_id": VN97_MAMBA2_SOURCE_MODEL_ID,
        "source_license": VN97_MAMBA2_SOURCE_LICENSE,
        "source_revision": source_revision,
        "source_config_sha256": source_config_sha256,
        "source_weight_sha256": source_weight_sha256,
        "source_spec": spec.canonical_object(),
        "unique_core_parameters": expected_unique_parameter_count(spec),
        "transfer_semantics": "tensor_value_identity_1to1",
        "tokenizer_semantics": "source_tokenizer_preserved",
        "source_tokenizer_model_id": VN97_MAMBA2_SOURCE_TOKENIZER_ID,
        "source_tokenizer_revision": VN97_MAMBA2_SOURCE_TOKENIZER_REVISION,
        "quantization_used": False,
        "lossy_mapping_used": False,
        "core_reinitialized": False,
        "augmentation_effect_at_g0": "exact_zero",
        "mapping": identity_tensor_mapping(spec),
    }
    manifest_id = _sha256_bytes(
        b"VN97M2G0TRANSFER1\0" + _canonical_json(body)
    )
    return {**body, "manifest_id": manifest_id}


def write_transfer_manifest(
    path: str | Path,
    manifest: Mapping[str, object],
) -> None:
    body = dict(manifest)
    manifest_id = body.pop("manifest_id", None)
    expected_id = _sha256_bytes(
        b"VN97M2G0TRANSFER1\0" + _canonical_json(body)
    )
    if manifest_id != expected_id:
        raise ValueError("transfer manifest identity mismatch")
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        dict(manifest),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    temp = resolved.with_name(resolved.name + ".tmp")
    temp.write_text(encoded, encoding="ascii")
    temp.replace(resolved)


def save_g0_checkpoint(
    path: str | Path,
    *,
    converted_state_dict: Mapping[str, torch.Tensor],
    manifest: Mapping[str, object],
) -> str:
    if manifest.get("schema") != VN97_MAMBA2_G0_TRANSFER_SCHEMA:
        raise ValueError("wrong Mamba-2 transfer manifest schema")
    body = dict(manifest)
    manifest_id = body.pop("manifest_id", None)
    if manifest_id != _sha256_bytes(
        b"VN97M2G0TRANSFER1\0" + _canonical_json(body)
    ):
        raise ValueError("transfer manifest identity mismatch")

    payload = {
        "schema": VN97_MAMBA2_G0_CHECKPOINT_SCHEMA,
        "architecture_generation": "G0_DIRECT_TRANSFER",
        "transfer_manifest": dict(manifest),
        "augmentation": {
            "multi_timescale_state": "present_zero_impact",
            "state_highway": "present_zero_impact",
            "vn97mem1_retrieval": "external_interface_zero_impact",
            "recurrent_reasoning": "orchestration_only",
        },
        "state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in converted_state_dict.items()
        },
    }
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temp = resolved.with_name(resolved.name + ".tmp")
    torch.save(payload, temp)
    temp.replace(resolved)
    return sha256_file(resolved)
