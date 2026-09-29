from __future__ import annotations

from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba2_g03_capsule import load_g03_capsule
from .mamba2_onnx import (
    Mamba2OnnxConfig,
    VN97Mamba2StepOnnx,
)


from .mamba2_bundle_contract import (
    VN97_MAMBA2_G05_SCHEMA,
    VN97_MAMBA2_G05_OPSET,
    VN97_MAMBA2_G05_INPUTS,
    VN97_MAMBA2_G05_OUTPUTS,
    VN97_MAMBA2_G05_CHUNKS,
    verify_g05_bundle,
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
    raise ValueError(f"unsupported G0.5 dtype: {dtype}")


class VN97Mamba2ParallelChunkOnnx(nn.Module):
    """One fixed chunk with parallel projections and precision-preserving state scan.

    valid_length chooses a prefix in [1, chunk_size]. Invalid suffix positions
    have masked hidden/residual values and explicitly retain the previous SSM state.
    The convolution final state is selected after exactly valid_length updates.

    With valid_length=1 this same graph is the recurrent decode graph.
    With valid_length=chunk_size it is the parallel prefill graph.
    """

    def __init__(
        self,
        step: VN97Mamba2StepOnnx,
        *,
        chunk_size: int,
    ) -> None:
        super().__init__()
        if chunk_size not in VN97_MAMBA2_G05_CHUNKS:
            raise ValueError(
                f"G0.5 chunk_size must be one of {VN97_MAMBA2_G05_CHUNKS}"
            )
        self.step = step
        self.chunk_size = int(chunk_size)

    @property
    def config(self) -> Mamba2OnnxConfig:
        return self.step.config

    @property
    def parameter_dtype(self) -> torch.dtype:
        return self.step.parameter_dtype

    def initial_state(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.step.initial_state(1)

    def _prefix_mask(
        self,
        valid_length: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        positions = torch.arange(
            self.chunk_size,
            dtype=valid_length.dtype,
            device=device,
        )
        mask = positions < valid_length[0]
        return mask.to(dtype=dtype).reshape(1, self.chunk_size, 1)

    def _state_scan(self, layer, dt_value, b_value, c_value, x_heads,
                    ssm_state, valid_length):
        # FP16 rounding after each transition is not associative. Keep the
        # projection/convolution batched, but use the canonical token update
        # for the state and readout in this same fixed-size graph.
        state = ssm_state
        outputs = []
        for index in range(self.chunk_size):
            y, candidate = layer._ssm_update(
                dt_value[:, index], b_value[:, index], c_value[:, index],
                x_heads[:, index], state,
            )
            active = index < valid_length[0]
            state = torch.where(active, candidate, state)
            outputs.append(torch.where(active, y, torch.zeros_like(y)))
        return torch.stack(outputs, dim=1), state

    def _causal_convolution_affine(self, layer, conv_window: torch.Tensor) -> torch.Tensor:
        # Match the step kernel's activation-dtype product, reduction, and bias
        # rounding. A fused conv1d does not preserve those FP16 boundaries.
        # Stack fixed windows so tokens are still evaluated in parallel.
        windows = torch.stack(
            [conv_window[..., i:i + self.config.d_conv]
             for i in range(self.chunk_size)],
            dim=2,
        )
        weight = layer.conv_weight[:, 0, :].to(
            device=windows.device, dtype=windows.dtype,
        )
        affine = torch.sum(windows * weight[None, :, None, :], dim=-1)
        affine = affine + layer.conv_bias.to(
            device=affine.device, dtype=affine.dtype,
        )[None, :, None]
        return affine.transpose(1, 2)

    def _parallel_layer(
        self,
        layer,
        hidden: torch.Tensor,
        residual: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        valid_length: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        cfg = self.config
        residual_next = (
            residual.float() + hidden.float()
        ) * mask.float()
        normalized = layer._rms_norm(
            residual_next.to(dtype=layer.block_norm.dtype),
            layer.block_norm,
        )
        normalized = normalized * mask.to(dtype=normalized.dtype)

        zxbcdt = F.linear(normalized, layer.in_proj)
        z, xbc_input, dt = torch.split(
            zxbcdt,
            [
                cfg.d_inner,
                cfg.conv_dim,
                cfg.n_heads,
            ],
            dim=-1,
        )

        xbc_input = xbc_input * mask.to(dtype=xbc_input.dtype)
        xbc_channels = xbc_input.transpose(1, 2)

        conv_window = torch.cat(
            (
                conv_state[..., 1:],
                xbc_channels,
            ),
            dim=-1,
        )
        xbc = F.silu(self._causal_convolution_affine(layer, conv_window))

        history = torch.cat(
            (conv_state, xbc_channels),
            dim=-1,
        )
        final_indices = (
            torch.arange(
                cfg.d_conv,
                dtype=valid_length.dtype,
                device=valid_length.device,
            )
            + valid_length[0]
        )
        next_conv = torch.index_select(
            history,
            -1,
            final_indices,
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
        x_heads = x.reshape(
            1,
            self.chunk_size,
            cfg.n_heads,
            cfg.head_dim,
        )

        # Invalid suffix positions must be exact no-op SSM transitions.
        dt_value = F.softplus(
            dt
            + layer.dt_bias.to(
                device=dt.device,
                dtype=dt.dtype,
            )
        )
        dt_value = dt_value * mask.to(dtype=dt_value.dtype)

        y, next_ssm = self._state_scan(
            layer, dt_value, b_value, c_value, x_heads, ssm_state, valid_length,
        )
        y = (
            y
            + layer.d_skip.to(
                device=y.device,
                dtype=y.dtype,
            )[None, None, :, None]
            * x_heads
        )
        y = y.reshape(
            1,
            self.chunk_size,
            cfg.d_inner,
        )
        y = layer._gated_rms_norm(y, z)
        output = F.linear(y, layer.out_proj)
        output = output * mask.to(dtype=output.dtype)
        return output, residual_next, next_conv, next_ssm

    def forward(
        self,
        input_ids: torch.Tensor,
        valid_length: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("G0.5 input_ids must be [1, chunk]")
        if valid_length.ndim != 1:
            raise ValueError("G0.5 valid_length must be [1]")

        mask = self._prefix_mask(
            valid_length,
            dtype=self.step.embedding.dtype,
            device=input_ids.device,
        )
        hidden = F.embedding(
            input_ids,
            self.step.embedding,
        )
        hidden = hidden * mask
        residual = torch.zeros_like(
            hidden,
            dtype=torch.float32,
        )
        next_conv_layers = []
        next_ssm_layers = []

        for index, layer in enumerate(self.step.layers):
            hidden, residual, next_conv, next_ssm = self._parallel_layer(
                layer,
                hidden,
                residual,
                conv_state[index],
                ssm_state[index],
                valid_length,
                mask,
            )
            next_conv_layers.append(next_conv)
            next_ssm_layers.append(next_ssm)

        residual = (
            residual + hidden.float()
        ) * mask.float()
        inv = torch.rsqrt(
            residual.square().mean(
                dim=-1,
                keepdim=True,
            )
            + self.config.rms_eps
        )
        hidden = (
            residual
            * inv
            * self.step.final_norm.float()
        ).to(dtype=self.step.embedding.dtype)
        hidden = hidden * mask
        logits = F.linear(
            hidden,
            self.step.embedding,
        )
        logits = logits * mask.to(dtype=logits.dtype)
        return (
            logits,
            torch.stack(next_conv_layers, dim=0),
            torch.stack(next_ssm_layers, dim=0),
        )


def repeated_step_reference(
    step: VN97Mamba2StepOnnx,
    input_ids: torch.Tensor,
    *,
    valid_length: int,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("reference input_ids must be [1, chunk]")
    if not 1 <= valid_length <= input_ids.shape[1]:
        raise ValueError("valid_length outside chunk")
    conv = conv_state
    ssm = ssm_state
    logits = []
    for index in range(valid_length):
        current, conv, ssm = step(
            input_ids[:, index],
            conv,
            ssm,
        )
        logits.append(current)
    stacked = torch.stack(logits, dim=1)
    if valid_length < input_ids.shape[1]:
        padding = torch.zeros(
            1,
            input_ids.shape[1] - valid_length,
            step.config.vocab_size,
            dtype=stacked.dtype,
            device=stacked.device,
        )
        stacked = torch.cat((stacked, padding), dim=1)
    return stacked, conv, ssm


def _state_contract(
    config: Mamba2OnnxConfig,
    *,
    dtype: torch.dtype,
) -> dict[str, object]:
    element_bytes = torch.tensor([], dtype=dtype).element_size()
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
        "batch_size": 1,
        "dtype": _dtype_name(dtype),
        "conv_state_shape": [
            config.n_layers,
            1,
            config.conv_dim,
            config.d_conv,
        ],
        "ssm_state_shape": [
            config.n_layers,
            1,
            config.n_heads,
            config.head_dim,
            config.d_state,
        ],
        "total_bytes_per_batch": (
            conv_elements + ssm_elements
        ) * element_bytes,
        "state_is_explicit": True,
        "state_carry_semantics": "exact_between_graph_invocations",
    }


def _export_supports_external_data() -> bool:
    try:
        return "external_data" in inspect.signature(
            torch.onnx.export
        ).parameters
    except (TypeError, ValueError):
        return False


def _graph_files(
    root: Path,
    graph_name: str,
) -> list[dict[str, object]]:
    records = []
    for path in sorted(root.iterdir()):
        if (
            path.is_file()
            and not path.is_symlink()
            and (
                path.name == graph_name
                or path.name.startswith(graph_name + ".")
            )
        ):
            records.append(
                {
                    "filename": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    if not any(item["filename"] == graph_name for item in records):
        raise ValueError("G0.5 recurrent ONNX graph missing")
    return records


def export_parallel_recurrent_onnx(
    model: VN97Mamba2ParallelChunkOnnx,
    output_dir: Path,
    *,
    capsule_id: str,
    capsule_manifest_sha256: str,
    source_weight_sha256: str,
    external_data: bool = True,
) -> dict[str, object]:
    _require_sha256(capsule_id, label="G0.5 capsule ID")
    _require_sha256(
        capsule_manifest_sha256,
        label="G0.5 capsule manifest SHA-256",
    )
    _require_sha256(
        source_weight_sha256,
        label="G0.5 source weight SHA-256",
    )
    if output_dir.exists():
        if (
            output_dir.is_symlink()
            or not output_dir.is_dir()
            or any(output_dir.iterdir())
        ):
            raise ValueError("G0.5 output must be new or empty")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    root = output_dir.resolve(strict=True)

    if external_data and not _export_supports_external_data():
        raise RuntimeError(
            "G0.5 production export requires PyTorch external_data support"
        )

    model = model.cpu().eval()
    conv, ssm = model.initial_state()
    tokens = torch.zeros(
        1,
        model.chunk_size,
        dtype=torch.long,
    )
    valid_length = torch.tensor(
        [model.chunk_size],
        dtype=torch.long,
    )
    graph_name = f"recurrent-{model.chunk_size}.onnx"
    graph_path = root / graph_name
    kwargs = dict(
        export_params=True,
        opset_version=VN97_MAMBA2_G05_OPSET,
        do_constant_folding=True,
        input_names=list(VN97_MAMBA2_G05_INPUTS),
        output_names=list(VN97_MAMBA2_G05_OUTPUTS),
        dynamo=True,
    )
    if _export_supports_external_data():
        kwargs["external_data"] = bool(external_data)
    torch.onnx.export(
        model,
        (tokens, valid_length, conv, ssm),
        graph_path,
        **kwargs,
    )

    body: dict[str, object] = {
        "schema": VN97_MAMBA2_G05_SCHEMA,
        "capsule_id": capsule_id,
        "capsule_manifest_sha256": capsule_manifest_sha256,
        "source_weight_sha256": source_weight_sha256,
        "config": asdict(model.config),
        "opset": VN97_MAMBA2_G05_OPSET,
        "max_chunk_size": model.chunk_size,
        "valid_length_min": 1,
        "valid_length_max": model.chunk_size,
        "inputs": list(VN97_MAMBA2_G05_INPUTS),
        "outputs": list(VN97_MAMBA2_G05_OUTPUTS),
        "graph_files": _graph_files(root, graph_name),
        "state_contract": _state_contract(
            model.config,
            dtype=model.parameter_dtype,
        ),
        "single_weight_graph": True,
        "decode_via_valid_length_one": True,
        "parallel_prefill_ready": True,
        "parallel_algorithm": "parallel_projection_conv_token_rounded_state_scan",
        "same_weights_semantics": True,
        "quantization_used": False,
        "external_data_requested": bool(external_data),
        "production_activation_authorized": False,
    }
    manifest_id = hashlib.sha256(
        b"VN97M2G05ONNX1\0" + _canonical_json(body)
    ).hexdigest()
    manifest = {**body, "manifest_id": manifest_id}
    (root / "manifest.vn97m2g05.json").write_bytes(
        _canonical_json(manifest) + b"\n"
    )
    verify_g05_bundle(root)
    return manifest


def export_g03_capsule_parallel_onnx(
    capsule_root: Path,
    output_dir: Path,
    *,
    chunk_size: int = 32,
) -> dict[str, object]:
    capsule = load_g03_capsule(
        capsule_root,
        verify_large_weight_sha256=True,
    )
    step = VN97Mamba2StepOnnx(
        Mamba2OnnxConfig.from_source_spec(capsule.spec),
        capsule.tensors,
    )
    model = VN97Mamba2ParallelChunkOnnx(
        step,
        chunk_size=chunk_size,
    )
    return export_parallel_recurrent_onnx(
        model,
        output_dir,
        capsule_id=capsule.manifest.capsule_id(),
        capsule_manifest_sha256=capsule.manifest_sha256,
        source_weight_sha256=capsule.manifest.source_weight_sha256,
        external_data=True,
    )


def run_ort_parallel(
    graph_path: Path,
    *,
    input_ids: torch.Tensor,
    valid_length: int,
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
            "valid_length": np.asarray([valid_length], dtype=np.int64),
            "conv_state": conv_state.detach().cpu().numpy(),
            "ssm_state": ssm_state.detach().cpu().numpy(),
        },
    )
    return tuple(torch.from_numpy(value) for value in outputs)  # type: ignore[return-value]


@torch.inference_mode()
def validate_parallel_parity(
    model: VN97Mamba2ParallelChunkOnnx,
    *,
    valid_length: int,
    conv_state: torch.Tensor | None = None,
    ssm_state: torch.Tensor | None = None,
    seed: int = 9705,
) -> dict[str, float]:
    if not 1 <= valid_length <= model.chunk_size:
        raise ValueError("valid_length outside G0.5 chunk")
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(
        0,
        model.config.vocab_size,
        (1, model.chunk_size),
        generator=generator,
        dtype=torch.long,
    )
    if conv_state is None or ssm_state is None:
        conv_state, ssm_state = model.initial_state()
    actual = model(
        tokens,
        torch.tensor([valid_length], dtype=torch.long),
        conv_state,
        ssm_state,
    )
    expected = repeated_step_reference(
        model.step,
        tokens,
        valid_length=valid_length,
        conv_state=conv_state,
        ssm_state=ssm_state,
    )
    names = ("logits", "conv_state", "ssm_state")
    metrics: dict[str, float] = {}
    for name, left, right in zip(names, actual, expected):
        diff = (
            left.detach().float().cpu()
            - right.detach().float().cpu()
        ).abs()
        metrics[f"max_{name}_abs_error"] = (
            float(diff.max().item()) if diff.numel() else 0.0
        )
    return metrics


@torch.inference_mode()
def validate_ort_parallel_parity(
    model: VN97Mamba2ParallelChunkOnnx,
    bundle_dir: Path,
    *,
    valid_length: int,
    seed: int = 9706,
) -> dict[str, float]:
    manifest = verify_g05_bundle(bundle_dir)
    if int(manifest["max_chunk_size"]) != model.chunk_size:
        raise ValueError("G0.5 model/bundle chunk mismatch")
    if not 1 <= valid_length <= model.chunk_size:
        raise ValueError("valid_length outside G0.5 chunk")

    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(
        0,
        model.config.vocab_size,
        (1, model.chunk_size),
        generator=generator,
        dtype=torch.long,
    )
    conv, ssm = model.initial_state()
    expected = repeated_step_reference(
        model.step,
        tokens,
        valid_length=valid_length,
        conv_state=conv,
        ssm_state=ssm,
    )
    graph = bundle_dir / f"recurrent-{model.chunk_size}.onnx"
    actual = run_ort_parallel(
        graph,
        input_ids=tokens,
        valid_length=valid_length,
        conv_state=conv,
        ssm_state=ssm,
    )
    names = ("logits", "conv_state", "ssm_state")
    metrics: dict[str, float] = {}
    for name, left, right in zip(names, actual, expected):
        diff = (
            left.detach().float().cpu()
            - right.detach().float().cpu()
        ).abs()
        metrics[f"max_{name}_abs_error"] = (
            float(diff.max().item()) if diff.numel() else 0.0
        )
    return metrics
