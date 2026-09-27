from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba2_g03_capsule import (
    VN97Mamba2LoadedCapsule,
    load_g03_capsule,
)
from .mamba2_ssd_reference import Mamba2ReferenceConfig
from .mamba2_transfer import Mamba2SourceSpec


VN97_MAMBA2_G04_ONNX_SCHEMA = "VN97M2G04ONNX1"
VN97_MAMBA2_G04_OPSET = 18
VN97_MAMBA2_G04_INPUTS = (
    "input_ids",
    "conv_state",
    "ssm_state",
)
VN97_MAMBA2_G04_OUTPUTS = (
    "logits",
    "next_conv_state",
    "next_ssm_state",
)


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
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    raise ValueError(f"unsupported Mamba-2 ONNX dtype: {dtype}")


@dataclass(frozen=True)
class Mamba2OnnxConfig:
    d_model: int
    n_layers: int
    vocab_size: int
    d_state: int
    d_conv: int
    expand: int
    head_dim: int
    n_groups: int
    rms_eps: float = 1.0e-5

    @classmethod
    def from_source_spec(cls, spec: Mamba2SourceSpec) -> "Mamba2OnnxConfig":
        return cls(
            d_model=spec.d_model,
            n_layers=spec.n_layers,
            vocab_size=spec.padded_vocab_size,
            d_state=spec.d_state,
            d_conv=spec.d_conv,
            expand=spec.expand,
            head_dim=spec.head_dim,
            n_groups=spec.n_groups,
        )

    def __post_init__(self) -> None:
        Mamba2ReferenceConfig(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            head_dim=self.head_dim,
            n_groups=self.n_groups,
            rms_eps=self.rms_eps,
        )
        if self.n_layers <= 0 or self.vocab_size <= 0:
            raise ValueError("Mamba-2 ONNX layers/vocab must be positive")
        if self.n_groups != 1:
            raise ValueError(
                "R2-G0.4 currently locks the official Mamba-2 2.7B ngroups=1"
            )

    @property
    def d_inner(self) -> int:
        return self.d_model * self.expand

    @property
    def n_heads(self) -> int:
        return self.d_inner // self.head_dim

    @property
    def conv_dim(self) -> int:
        return self.d_inner + 2 * self.n_groups * self.d_state


def _frozen_parameter(value: torch.Tensor) -> nn.Parameter:
    return nn.Parameter(value.detach(), requires_grad=False)


class VN97Mamba2OnnxLayer(nn.Module):
    def __init__(
        self,
        *,
        config: Mamba2OnnxConfig,
        block_norm: torch.Tensor,
        in_proj: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor,
        dt_bias: torch.Tensor,
        a_log: torch.Tensor,
        d_skip: torch.Tensor,
        mixer_norm: torch.Tensor,
        out_proj: torch.Tensor,
    ) -> None:
        super().__init__()
        self.config = config
        self.block_norm = _frozen_parameter(block_norm)
        self.in_proj = _frozen_parameter(in_proj)
        self.conv_weight = _frozen_parameter(conv_weight)
        self.conv_bias = _frozen_parameter(conv_bias)
        self.dt_bias = _frozen_parameter(dt_bias)
        self.a_log = _frozen_parameter(a_log)
        self.d_skip = _frozen_parameter(d_skip)
        self.mixer_norm = _frozen_parameter(mixer_norm)
        self.out_proj = _frozen_parameter(out_proj)

    def _rms_norm(
        self,
        value: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        source = value.float()
        inv = torch.rsqrt(
            source.square().mean(dim=-1, keepdim=True)
            + self.config.rms_eps
        )
        return (source * inv * weight.float()).to(dtype=weight.dtype)

    def _gated_rms_norm(
        self,
        value: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        # Official Mamba2 default is norm_before_gate=False. Keep every
        # leading dimension so the same layer works for recurrent [B,D]
        # and parallel-prefill [B,L,D] execution.
        if value.shape != gate.shape:
            raise ValueError("Mamba-2 gated RMSNorm value/gate shape mismatch")
        if value.shape[-1] != self.config.d_inner:
            raise ValueError("Mamba-2 gated RMSNorm width mismatch")
        source = value.float() * F.silu(gate.float())
        group_size = self.config.d_inner // self.config.n_groups
        grouped = source.reshape(
            *source.shape[:-1],
            self.config.n_groups,
            group_size,
        )
        inv = torch.rsqrt(
            grouped.square().mean(dim=-1, keepdim=True)
            + self.config.rms_eps
        )
        out = (grouped * inv).reshape_as(source)
        return (
            out * self.mixer_norm.float()
        ).to(dtype=value.dtype)

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        residual_next = residual.float() + hidden.float()
        normalized = self._rms_norm(
            residual_next.to(dtype=self.block_norm.dtype),
            self.block_norm,
        )

        zxbcdt = F.linear(normalized, self.in_proj)
        z, xbc, dt = torch.split(
            zxbcdt,
            [
                cfg.d_inner,
                cfg.conv_dim,
                cfg.n_heads,
            ],
            dim=-1,
        )

        next_conv = torch.roll(conv_state, shifts=-1, dims=-1)
        next_conv = torch.cat(
            (
                next_conv[..., :-1],
                xbc.unsqueeze(-1),
            ),
            dim=-1,
        )
        xbc = torch.sum(
            next_conv
            * self.conv_weight[:, 0, :].to(
                device=next_conv.device,
                dtype=next_conv.dtype,
            ),
            dim=-1,
        )
        xbc = F.silu(
            xbc
            + self.conv_bias.to(
                device=xbc.device,
                dtype=xbc.dtype,
            )
        )
        x, b_value, c_value = torch.split(
            xbc,
            [
                cfg.d_inner,
                cfg.d_state,
                cfg.d_state,
            ],
            dim=-1,
        )

        a = -torch.exp(self.a_log.float())
        dt_value = F.softplus(
            dt
            + self.dt_bias.to(
                device=dt.device,
                dtype=dt.dtype,
            )
        )
        d_a = torch.exp(dt_value.float() * a.unsqueeze(0))

        x_heads = x.reshape(
            x.shape[0],
            cfg.n_heads,
            cfg.head_dim,
        )
        d_b_x = torch.einsum(
            "bh,bn,bhp->bhpn",
            dt_value.float(),
            b_value.float(),
            x_heads.float(),
        )
        next_ssm = (
            ssm_state.float()
            * d_a[:, :, None, None]
            + d_b_x
        ).to(dtype=ssm_state.dtype)

        y = torch.einsum(
            "bhpn,bn->bhp",
            next_ssm.to(dtype=x_heads.dtype),
            c_value.to(dtype=x_heads.dtype),
        )
        y = (
            y
            + self.d_skip.to(
                device=y.device,
                dtype=y.dtype,
            )[None, :, None]
            * x_heads
        ).reshape(x.shape[0], cfg.d_inner)
        y = self._gated_rms_norm(y, z)
        output = F.linear(y, self.out_proj)
        return output, residual_next, next_conv, next_ssm


class VN97Mamba2StepOnnx(nn.Module):
    """One-token exact recurrent VN97/Mamba-2 lowering with explicit state."""

    def __init__(
        self,
        config: Mamba2OnnxConfig,
        tensors: Mapping[str, torch.Tensor],
    ) -> None:
        super().__init__()
        self.config = config
        self.embedding = _frozen_parameter(
            tensors["vn97.core.embedding.weight"]
        )
        self.final_norm = _frozen_parameter(
            tensors["vn97.core.final_norm.weight"]
        )
        layers = []
        for index in range(config.n_layers):
            prefix = f"vn97.core.layers.{index}"
            mixer = f"{prefix}.mixer"
            layers.append(
                VN97Mamba2OnnxLayer(
                    config=config,
                    block_norm=tensors[f"{prefix}.norm.weight"],
                    in_proj=tensors[f"{mixer}.in_proj.weight"],
                    conv_weight=tensors[f"{mixer}.conv1d.weight"],
                    conv_bias=tensors[f"{mixer}.conv1d.bias"],
                    dt_bias=tensors[f"{mixer}.dt_bias"],
                    a_log=tensors[f"{mixer}.A_log"],
                    d_skip=tensors[f"{mixer}.D"],
                    mixer_norm=tensors[f"{mixer}.norm.weight"],
                    out_proj=tensors[f"{mixer}.out_proj.weight"],
                )
            )
        self.layers = nn.ModuleList(layers)

    @property
    def parameter_dtype(self) -> torch.dtype:
        return self.embedding.dtype

    def initial_state(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        dtype = self.parameter_dtype
        device = self.embedding.device
        return (
            torch.zeros(
                self.config.n_layers,
                batch_size,
                self.config.conv_dim,
                self.config.d_conv,
                dtype=dtype,
                device=device,
            ),
            torch.zeros(
                self.config.n_layers,
                batch_size,
                self.config.n_heads,
                self.config.head_dim,
                self.config.d_state,
                dtype=dtype,
                device=device,
            ),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        if input_ids.ndim != 1:
            raise ValueError("step input_ids must be [batch]")
        hidden = F.embedding(input_ids, self.embedding)
        residual = torch.zeros_like(hidden, dtype=torch.float32)
        next_conv_layers = []
        next_ssm_layers = []

        for index, layer in enumerate(self.layers):
            hidden, residual, next_conv, next_ssm = layer(
                hidden,
                residual,
                conv_state[index],
                ssm_state[index],
            )
            next_conv_layers.append(next_conv)
            next_ssm_layers.append(next_ssm)

        residual = residual + hidden.float()
        source = residual
        inv = torch.rsqrt(
            source.square().mean(dim=-1, keepdim=True)
            + cfg.rms_eps
        )
        hidden = (
            source
            * inv
            * self.final_norm.float()
        ).to(dtype=self.embedding.dtype)
        logits = F.linear(hidden, self.embedding)
        return (
            logits,
            torch.stack(next_conv_layers, dim=0),
            torch.stack(next_ssm_layers, dim=0),
        )


class VN97Mamba2ChunkExactOnnx(nn.Module):
    """Fixed-size exact recurrent unroll.

    This preserves step semantics. It is not the later parallel SSD prefill
    lowering and must not be labelled as such.
    """

    def __init__(
        self,
        step: VN97Mamba2StepOnnx,
        *,
        chunk_size: int,
    ) -> None:
        super().__init__()
        if chunk_size <= 1 or chunk_size > 32:
            raise ValueError("G0.4 exact chunk size must be in [2, 32]")
        self.step = step
        self.chunk_size = int(chunk_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("chunk input_ids must be [batch, sequence]")
        logits = []
        conv = conv_state
        ssm = ssm_state
        for index in range(self.chunk_size):
            current, conv, ssm = self.step(
                input_ids[:, index],
                conv,
                ssm,
            )
            logits.append(current)
        return torch.stack(logits, dim=1), conv, ssm


def _state_contract(
    config: Mamba2OnnxConfig,
    *,
    dtype: torch.dtype,
    batch_size: int,
) -> dict[str, object]:
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    conv_elements = (
        config.n_layers
        * config.conv_dim
        * config.d_conv
    )
    ssm_elements = (
        config.n_layers
        * config.n_heads
        * config.head_dim
        * config.d_state
    )
    return {
        "state_is_explicit": True,
        "state_carry_semantics": "exact_between_graph_invocations",
        "dtype": _dtype_name(dtype),
        "bytes_per_element": bytes_per_element,
        "conv_state": {
            "shape": [
                config.n_layers,
                batch_size,
                config.conv_dim,
                config.d_conv,
            ],
            "elements_per_batch": conv_elements,
            "bytes_per_batch": conv_elements * bytes_per_element,
        },
        "ssm_state": {
            "shape": [
                config.n_layers,
                batch_size,
                config.n_heads,
                config.head_dim,
                config.d_state,
            ],
            "elements_per_batch": ssm_elements,
            "bytes_per_batch": ssm_elements * bytes_per_element,
        },
        "total_elements_per_batch": conv_elements + ssm_elements,
        "total_bytes_per_batch": (
            conv_elements + ssm_elements
        ) * bytes_per_element,
    }


def _collect_graph_files(root: Path, graph_name: str) -> list[dict[str, object]]:
    records = []
    prefix = graph_name
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.is_symlink():
            continue
        if path.name == prefix or path.name.startswith(prefix + "."):
            records.append(
                {
                    "filename": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    if not any(item["filename"] == graph_name for item in records):
        raise ValueError("ONNX graph file was not produced")
    return records


def _export_supports_external_data() -> bool:
    try:
        return "external_data" in inspect.signature(
            torch.onnx.export
        ).parameters
    except (TypeError, ValueError):
        return False


def export_mamba2_step_onnx(
    model: VN97Mamba2StepOnnx,
    output_dir: Path,
    *,
    capsule_id: str,
    capsule_manifest_sha256: str,
    source_weight_sha256: str,
    example_batch_size: int = 1,
    export_external_data: bool = True,
) -> dict[str, object]:
    _require_sha256(capsule_id, label="G0.4 capsule ID")
    _require_sha256(
        capsule_manifest_sha256,
        label="G0.4 capsule manifest SHA-256",
    )
    _require_sha256(
        source_weight_sha256,
        label="G0.4 source weight SHA-256",
    )
    if example_batch_size != 1:
        raise ValueError(
            "R2-G0.4 mobile ONNX currently requires batch_size=1"
        )
    if output_dir.exists():
        if (
            output_dir.is_symlink()
            or not output_dir.is_dir()
            or any(output_dir.iterdir())
        ):
            raise ValueError("G0.4 ONNX output must be new or empty")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    root = output_dir.resolve(strict=True)

    if export_external_data and not _export_supports_external_data():
        raise RuntimeError(
            "installed PyTorch ONNX exporter lacks external_data support; "
            "R2-G0.4 production export requires a newer PyTorch"
        )

    model = model.cpu().eval()
    conv, ssm = model.initial_state(example_batch_size)
    input_ids = torch.zeros(
        example_batch_size,
        dtype=torch.long,
    )
    graph_path = root / "step.onnx"
    kwargs = dict(
        export_params=True,
        opset_version=VN97_MAMBA2_G04_OPSET,
        do_constant_folding=True,
        input_names=list(VN97_MAMBA2_G04_INPUTS),
        output_names=list(VN97_MAMBA2_G04_OUTPUTS),
        dynamo=True,
    )
    if _export_supports_external_data():
        kwargs["external_data"] = bool(export_external_data)
    torch.onnx.export(
        model,
        (input_ids, conv, ssm),
        graph_path,
        **kwargs,
    )

    config = asdict(model.config)
    body: dict[str, object] = {
        "schema": VN97_MAMBA2_G04_ONNX_SCHEMA,
        "capsule_id": capsule_id,
        "capsule_manifest_sha256": capsule_manifest_sha256,
        "source_weight_sha256": source_weight_sha256,
        "config": config,
        "opset": VN97_MAMBA2_G04_OPSET,
        "graph_kind": "step",
        "batch_size": example_batch_size,
        "graph_files": _collect_graph_files(root, "step.onnx"),
        "inputs": list(VN97_MAMBA2_G04_INPUTS),
        "outputs": list(VN97_MAMBA2_G04_OUTPUTS),
        "state_contract": _state_contract(
            model.config,
            dtype=model.parameter_dtype,
            batch_size=example_batch_size,
        ),
        "same_weights_semantics": True,
        "quantization_used": False,
        "external_data_requested": bool(export_external_data),
        "parallel_prefill_ready": False,
        "chunk_semantics": "not_exported_in_g04",
        "production_activation_authorized": False,
    }
    manifest_id = hashlib.sha256(
        b"VN97M2G04ONNX1\0" + _canonical_json(body)
    ).hexdigest()
    manifest = {**body, "manifest_id": manifest_id}
    (root / "manifest.vn97m2g04.json").write_bytes(
        _canonical_json(manifest) + b"\n"
    )
    verify_mamba2_g04_bundle(root)
    return manifest


def export_g03_capsule_step_onnx(
    capsule_root: Path,
    output_dir: Path,
    *,
    example_batch_size: int = 1,
) -> dict[str, object]:
    capsule: VN97Mamba2LoadedCapsule = load_g03_capsule(
        capsule_root,
        verify_large_weight_sha256=True,
    )
    config = Mamba2OnnxConfig.from_source_spec(capsule.spec)
    model = VN97Mamba2StepOnnx(config, capsule.tensors)
    return export_mamba2_step_onnx(
        model,
        output_dir,
        capsule_id=capsule.manifest.capsule_id(),
        capsule_manifest_sha256=capsule.manifest_sha256,
        source_weight_sha256=capsule.manifest.source_weight_sha256,
        example_batch_size=example_batch_size,
        export_external_data=True,
    )


def verify_mamba2_g04_bundle(root: Path) -> dict[str, object]:
    resolved = root.resolve(strict=True)
    path = resolved / "manifest.vn97m2g04.json"
    payload = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(payload, dict):
        raise ValueError("G0.4 ONNX manifest must be an object")
    if payload.get("schema") != VN97_MAMBA2_G04_ONNX_SCHEMA:
        raise ValueError("G0.4 ONNX schema mismatch")
    manifest_id = _require_sha256(
        payload.get("manifest_id"),
        label="G0.4 manifest ID",
    )
    body = dict(payload)
    body.pop("manifest_id", None)
    expected = hashlib.sha256(
        b"VN97M2G04ONNX1\0" + _canonical_json(body)
    ).hexdigest()
    if manifest_id != expected:
        raise ValueError("G0.4 ONNX manifest identity mismatch")
    for field in (
        "capsule_id",
        "capsule_manifest_sha256",
        "source_weight_sha256",
    ):
        _require_sha256(payload.get(field), label=f"G0.4 {field}")
    if payload.get("batch_size") != 1:
        raise ValueError("G0.4 mobile graph must use batch_size=1")
    if payload.get("same_weights_semantics") is not True:
        raise ValueError("G0.4 same-weight semantics are not locked")
    if payload.get("quantization_used") is not False:
        raise ValueError("G0.4 must not quantize the inherited core")
    if payload.get("parallel_prefill_ready") is not False:
        raise ValueError("G0.4 must not claim parallel prefill")
    if payload.get("production_activation_authorized") is not False:
        raise ValueError("G0.4 cannot self-authorize production activation")
    if payload.get("inputs") != list(VN97_MAMBA2_G04_INPUTS):
        raise ValueError("G0.4 input contract mismatch")
    if payload.get("outputs") != list(VN97_MAMBA2_G04_OUTPUTS):
        raise ValueError("G0.4 output contract mismatch")

    files = payload.get("graph_files")
    if not isinstance(files, list) or not files:
        raise ValueError("G0.4 graph file inventory missing")
    seen = set()
    for record in files:
        if not isinstance(record, dict):
            raise ValueError("G0.4 graph record is invalid")
        name = record.get("filename")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("G0.4 graph filename invalid")
        seen.add(name)
        target = resolved / name
        if target.is_symlink() or not target.is_file():
            raise ValueError(f"G0.4 graph file missing: {name}")
        if target.stat().st_size != record.get("bytes"):
            raise ValueError(f"G0.4 graph byte size mismatch: {name}")
        if _sha256_file(target) != record.get("sha256"):
            raise ValueError(f"G0.4 graph SHA-256 mismatch: {name}")
    if "step.onnx" not in seen:
        raise ValueError("G0.4 step.onnx is missing")
    return payload


def run_ort_step(
    graph_path: Path,
    *,
    input_ids: torch.Tensor,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(graph_path),
        providers=["CPUExecutionProvider"],
    )
    outputs = session.run(
        None,
        {
            "input_ids": input_ids.detach().cpu().numpy().astype(
                np.int64,
                copy=False,
            ),
            "conv_state": conv_state.detach().cpu().numpy(),
            "ssm_state": ssm_state.detach().cpu().numpy(),
        },
    )
    return tuple(torch.from_numpy(value) for value in outputs)  # type: ignore[return-value]


@torch.inference_mode()
def validate_ort_step_parity(
    model: VN97Mamba2StepOnnx,
    bundle_dir: Path,
    *,
    batch_size: int = 1,
    seed: int = 9704,
) -> dict[str, float]:
    manifest = verify_mamba2_g04_bundle(bundle_dir)
    if batch_size != int(manifest["batch_size"]):
        raise ValueError(
            "G0.4 parity batch size must match the static mobile graph"
        )
    if manifest["graph_kind"] != "step":
        raise ValueError("G0.4 parity requires a step graph")
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(
        0,
        model.config.vocab_size,
        (batch_size,),
        generator=generator,
        dtype=torch.long,
    )
    conv, ssm = model.initial_state(batch_size)
    reference = model(input_ids, conv, ssm)
    actual = run_ort_step(
        bundle_dir / "step.onnx",
        input_ids=input_ids,
        conv_state=conv,
        ssm_state=ssm,
    )
    names = ("logits", "conv_state", "ssm_state")
    metrics: dict[str, float] = {}
    for name, expected, observed in zip(names, reference, actual):
        diff = (
            expected.detach().float().cpu()
            - observed.detach().float().cpu()
        ).abs()
        metrics[f"max_{name}_abs_error"] = (
            float(diff.max().item()) if diff.numel() else 0.0
        )
    return metrics
