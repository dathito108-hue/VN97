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


VN97_MAMBA2_G05_SCHEMA = "VN97M2G05ONNX1"
VN97_MAMBA2_G05_OPSET = 18
VN97_MAMBA2_G05_INPUTS = (
    "input_ids",
    "valid_length",
    "conv_state",
    "ssm_state",
)
VN97_MAMBA2_G05_OUTPUTS = (
    "logits",
    "next_conv_state",
    "next_ssm_state",
)
VN97_MAMBA2_G05_CHUNKS = (8, 16, 32)


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
    """One fixed maximum chunk using a parallel one-chunk SSD factorization.

    valid_length chooses a prefix in [1, chunk_size]. Invalid suffix positions
    have dt=0 and masked hidden/residual values, so they do not advance the SSM.
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

        strict = torch.tril(
            torch.ones(
                chunk_size,
                chunk_size,
                dtype=torch.bool,
            ),
            diagonal=-1,
        )
        lower = torch.tril(
            torch.ones(
                chunk_size,
                chunk_size,
                dtype=torch.bool,
            ),
            diagonal=0,
        )
        self.register_buffer(
            "_strict_lower_mask",
            strict,
            persistent=False,
        )
        self.register_buffer(
            "_lower_mask",
            lower,
            persistent=False,
        )

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

    def _segment_sum(
        self,
        a_discrete: torch.Tensor,
    ) -> torch.Tensor:
        # a_discrete: [batch, heads, length].
        repeated = a_discrete.unsqueeze(-1).expand(
            -1,
            -1,
            -1,
            self.chunk_size,
        )
        zeros = torch.zeros_like(repeated)
        strict = torch.where(
            self._strict_lower_mask[None, None, :, :],
            repeated,
            zeros,
        )
        cumulative = torch.cumsum(strict, dim=-2)
        negative_inf = torch.full_like(
            cumulative,
            -torch.inf,
        )
        return torch.where(
            self._lower_mask[None, None, :, :],
            cumulative,
            negative_inf,
        )

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
        xbc = F.conv1d(
            conv_window,
            layer.conv_weight,
            layer.conv_bias,
            groups=cfg.conv_dim,
        ).transpose(1, 2)
        xbc = F.silu(xbc)

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

        a = -torch.exp(layer.a_log.float())
        a_discrete = (
            dt_value.float()
            * a.reshape(1, 1, cfg.n_heads)
        )
        x_discrete = (
            x_heads.float()
            * dt_value.float().unsqueeze(-1)
        )

        # Official 2.7B uses ngroups=1: B/C are shared across all heads.
        b_heads = b_value.float().unsqueeze(2).expand(
            -1,
            -1,
            cfg.n_heads,
            -1,
        )
        c_heads = c_value.float().unsqueeze(2).expand(
            -1,
            -1,
            cfg.n_heads,
            -1,
        )

        a_by_head = a_discrete.permute(0, 2, 1)
        a_cumsum = torch.cumsum(a_by_head, dim=-1)
        transition = torch.exp(
            self._segment_sum(a_by_head)
        )

        # Intra-chunk causal contribution.
        y_diag = torch.einsum(
            "blhn,bshn,bhls,bshp->blhp",
            c_heads,
            b_heads,
            transition,
            x_discrete,
        )

        # Initial state contribution to every output position.
        state_decay_out = torch.exp(a_cumsum)
        y_initial = torch.einsum(
            "blhn,bhpn,bhl->blhp",
            c_heads,
            ssm_state.float(),
            state_decay_out,
        )

        # Final state after the whole fixed chunk. Invalid suffix dt=0 means
        # the final cumsum is exactly the state after valid_length tokens.
        decay_to_end = torch.exp(
            a_cumsum[:, :, -1:].expand_as(a_cumsum)
            - a_cumsum
        )
        chunk_state = torch.einsum(
            "blhn,bhl,blhp->bhpn",
            b_heads,
            decay_to_end,
            x_discrete,
        )
        initial_decay = torch.exp(
            a_cumsum[:, :, -1]
        )[:, :, None, None]
        next_ssm = (
            ssm_state.float() * initial_decay
            + chunk_state
        ).to(dtype=ssm_state.dtype)

        y = (y_diag + y_initial).to(dtype=x_heads.dtype)
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
        "parallel_algorithm": "one_chunk_ssd_factorization",
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


def verify_g05_bundle(root: Path) -> dict[str, object]:
    resolved = root.resolve(strict=True)
    path = resolved / "manifest.vn97m2g05.json"
    payload = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(payload, dict):
        raise ValueError("G0.5 manifest must be an object")
    if payload.get("schema") != VN97_MAMBA2_G05_SCHEMA:
        raise ValueError("G0.5 schema mismatch")
    manifest_id = _require_sha256(
        payload.get("manifest_id"),
        label="G0.5 manifest ID",
    )
    body = dict(payload)
    body.pop("manifest_id", None)
    expected = hashlib.sha256(
        b"VN97M2G05ONNX1\0" + _canonical_json(body)
    ).hexdigest()
    if manifest_id != expected:
        raise ValueError("G0.5 manifest identity mismatch")
    for field in (
        "capsule_id",
        "capsule_manifest_sha256",
        "source_weight_sha256",
    ):
        _require_sha256(payload.get(field), label=f"G0.5 {field}")
    if payload.get("single_weight_graph") is not True:
        raise ValueError("G0.5 must use one shared-weight recurrent graph")
    if payload.get("decode_via_valid_length_one") is not True:
        raise ValueError("G0.5 decode contract mismatch")
    if payload.get("parallel_prefill_ready") is not True:
        raise ValueError("G0.5 parallel prefill contract missing")
    if payload.get("same_weights_semantics") is not True:
        raise ValueError("G0.5 same-weight semantics mismatch")
    if payload.get("quantization_used") is not False:
        raise ValueError("G0.5 must not quantize inherited weights")
    if payload.get("production_activation_authorized") is not False:
        raise ValueError("G0.5 cannot self-authorize production")
    chunk = payload.get("max_chunk_size")
    if chunk not in VN97_MAMBA2_G05_CHUNKS:
        raise ValueError("G0.5 chunk size invalid")
    if (
        payload.get("valid_length_min") != 1
        or payload.get("valid_length_max") != chunk
    ):
        raise ValueError("G0.5 valid-length contract mismatch")
    if payload.get("inputs") != list(VN97_MAMBA2_G05_INPUTS):
        raise ValueError("G0.5 input contract mismatch")
    if payload.get("outputs") != list(VN97_MAMBA2_G05_OUTPUTS):
        raise ValueError("G0.5 output contract mismatch")

    files = payload.get("graph_files")
    if not isinstance(files, list) or not files:
        raise ValueError("G0.5 graph inventory missing")
    expected_graph = f"recurrent-{chunk}.onnx"
    seen = set()
    for record in files:
        if not isinstance(record, dict):
            raise ValueError("G0.5 graph record invalid")
        name = record.get("filename")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("G0.5 graph filename invalid")
        seen.add(name)
        file = resolved / name
        if file.is_symlink() or not file.is_file():
            raise ValueError(f"G0.5 graph file missing: {name}")
        if file.stat().st_size != record.get("bytes"):
            raise ValueError(f"G0.5 graph byte size mismatch: {name}")
        if _sha256_file(file) != record.get("sha256"):
            raise ValueError(f"G0.5 graph hash mismatch: {name}")
    if expected_graph not in seen:
        raise ValueError("G0.5 recurrent graph missing")
    return payload


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
