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


VN97_MAMBA2_G06_SCHEMA = "VN97M2G06ONNX1"
VN97_MAMBA2_G06_OPSET = 18
VN97_MAMBA2_G06_INPUTS = (
    "input_ids",
    "conv_state",
    "ssm_state",
)
VN97_MAMBA2_G06_OUTPUTS = (
    "logits",
    "next_conv_state",
    "next_ssm_state",
)
VN97_MAMBA2_G06_MAX_SEQUENCE = 32


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
    raise ValueError(f"unsupported G0.6 state dtype: {dtype}")


class VN97Mamba2DynamicSequenceOnnx(nn.Module):
    """Batch-one Mamba-2 graph with truly dynamic sequence length 1..max.

    The recurrent state is explicit. A length-1 invocation is decode; a longer
    invocation is parallel SSD prefill. Compute therefore scales with the
    actual sequence length instead of the fixed maximum chunk.
    """

    def __init__(
        self,
        step: VN97Mamba2StepOnnx,
        *,
        max_sequence_length: int = VN97_MAMBA2_G06_MAX_SEQUENCE,
    ) -> None:
        super().__init__()
        if max_sequence_length < 2 or max_sequence_length > 64:
            raise ValueError("G0.6 max sequence length must be in [2, 64]")
        self.step = step
        self.max_sequence_length = int(max_sequence_length)

    @property
    def config(self) -> Mamba2OnnxConfig:
        return self.step.config

    @property
    def parameter_dtype(self) -> torch.dtype:
        return self.step.parameter_dtype

    def initial_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.step.initial_state(1)

    @staticmethod
    def _stable_segment_sum(
        a_by_head: torch.Tensor,
    ) -> torch.Tensor:
        # a_by_head: [1, H, L]. Construct the same stable lower-triangular
        # segment sums used by the Mamba-2 minimal SSD formulation.
        length = a_by_head.shape[-1]
        repeated = a_by_head.unsqueeze(-1).expand(
            -1,
            -1,
            -1,
            length,
        )
        positions = torch.arange(
            length,
            device=a_by_head.device,
        )
        row = positions.reshape(length, 1)
        col = positions.reshape(1, length)
        strict_lower = row > col
        lower = row >= col
        masked = torch.where(
            strict_lower[None, None, :, :],
            repeated,
            torch.zeros_like(repeated),
        )
        cumulative = torch.cumsum(masked, dim=-2)
        return torch.where(
            lower[None, None, :, :],
            cumulative,
            torch.full_like(cumulative, -torch.inf),
        )

    def _parallel_layer(
        self,
        layer,
        hidden: torch.Tensor,
        residual: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        cfg = self.config
        residual_next = residual.float() + hidden.float()
        normalized = layer._rms_norm(
            residual_next.to(dtype=layer.block_norm.dtype),
            layer.block_norm,
        )

        zxbcdt = F.linear(normalized, layer.in_proj)
        z, xbc_input, dt = torch.split(
            zxbcdt,
            [cfg.d_inner, cfg.conv_dim, cfg.n_heads],
            dim=-1,
        )
        xbc_channels = xbc_input.transpose(1, 2)

        conv_window = torch.cat(
            (conv_state[..., 1:], xbc_channels),
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
        next_conv = history[..., -cfg.d_conv:]

        x, b_value, c_value = torch.split(
            xbc,
            [cfg.d_inner, cfg.d_state, cfg.d_state],
            dim=-1,
        )
        length = x.shape[1]
        x_heads = x.reshape(
            1,
            length,
            cfg.n_heads,
            cfg.head_dim,
        )

        dt_value = F.softplus(
            dt
            + layer.dt_bias.to(
                device=dt.device,
                dtype=dt.dtype,
            )
        )
        a = -torch.exp(layer.a_log.float())
        a_discrete = (
            dt_value.float()
            * a.reshape(1, 1, cfg.n_heads)
        )
        x_discrete = (
            x_heads.float()
            * dt_value.float().unsqueeze(-1)
        )

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
            self._stable_segment_sum(a_by_head)
        )

        y_diag = torch.einsum(
            "blhn,bshn,bhls,bshp->blhp",
            c_heads,
            b_heads,
            transition,
            x_discrete,
        )
        state_decay_out = torch.exp(a_cumsum)
        y_initial = torch.einsum(
            "blhn,bhpn,bhl->blhp",
            c_heads,
            ssm_state.float(),
            state_decay_out,
        )

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
        y = y.reshape(1, length, cfg.d_inner)
        y = layer._gated_rms_norm(y, z)
        output = F.linear(y, layer.out_proj)
        return output, residual_next, next_conv, next_ssm

    def forward(
        self,
        input_ids: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = F.embedding(
            input_ids,
            self.step.embedding,
        )
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
            )
            next_conv_layers.append(next_conv)
            next_ssm_layers.append(next_ssm)

        residual = residual + hidden.float()
        inv = torch.rsqrt(
            residual.square().mean(dim=-1, keepdim=True)
            + self.config.rms_eps
        )
        hidden = (
            residual
            * inv
            * self.step.final_norm.float()
        ).to(dtype=self.step.embedding.dtype)
        # Keep sampling output uniformly FP32 even when inherited weights and
        # recurrent state are FP16/BF16. This is a representation cast only.
        logits = F.linear(
            hidden,
            self.step.embedding,
        ).float()
        return (
            logits,
            torch.stack(next_conv_layers, dim=0),
            torch.stack(next_ssm_layers, dim=0),
        )


def repeated_step_reference(
    step: VN97Mamba2StepOnnx,
    input_ids: torch.Tensor,
    *,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("G0.6 reference input must be [1, sequence]")
    if input_ids.shape[1] < 1:
        raise ValueError("G0.6 sequence must not be empty")
    conv = conv_state
    ssm = ssm_state
    logits = []
    for index in range(input_ids.shape[1]):
        current, conv, ssm = step(
            input_ids[:, index],
            conv,
            ssm,
        )
        logits.append(current.float())
    return torch.stack(logits, dim=1), conv, ssm


def _state_contract(
    config: Mamba2OnnxConfig,
    *,
    dtype: torch.dtype,
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
        "batch_size": 1,
        "dtype": _dtype_name(dtype),
        "bytes_per_element": bytes_per_element,
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
        "conv_state_elements": conv_elements,
        "ssm_state_elements": ssm_elements,
        "total_bytes_per_batch": (
            conv_elements + ssm_elements
        ) * bytes_per_element,
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
        raise ValueError("G0.6 dynamic recurrent graph missing")
    return records


def export_dynamic_sequence_onnx(
    model: VN97Mamba2DynamicSequenceOnnx,
    output_dir: Path,
    *,
    capsule_id: str,
    capsule_manifest_sha256: str,
    source_weight_sha256: str,
    example_sequence_length: int = 8,
    external_data: bool = True,
) -> dict[str, object]:
    _require_sha256(capsule_id, label="G0.6 capsule ID")
    _require_sha256(
        capsule_manifest_sha256,
        label="G0.6 capsule manifest SHA-256",
    )
    _require_sha256(
        source_weight_sha256,
        label="G0.6 source weight SHA-256",
    )
    if not 2 <= example_sequence_length <= model.max_sequence_length:
        raise ValueError("G0.6 export example sequence is outside bounds")
    if output_dir.exists():
        if (
            output_dir.is_symlink()
            or not output_dir.is_dir()
            or any(output_dir.iterdir())
        ):
            raise ValueError("G0.6 output must be new or empty")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    root = output_dir.resolve(strict=True)

    if external_data and not _export_supports_external_data():
        raise RuntimeError(
            "G0.6 production export requires PyTorch external_data support"
        )

    model = model.cpu().eval()
    conv, ssm = model.initial_state()
    tokens = torch.zeros(
        1,
        example_sequence_length,
        dtype=torch.long,
    )
    sequence = torch.export.Dim(
        "sequence",
        min=1,
        max=model.max_sequence_length,
    )
    graph_name = "recurrent-dynamic.onnx"
    graph_path = root / graph_name
    kwargs = dict(
        export_params=True,
        opset_version=VN97_MAMBA2_G06_OPSET,
        input_names=list(VN97_MAMBA2_G06_INPUTS),
        output_names=list(VN97_MAMBA2_G06_OUTPUTS),
        dynamo=True,
        dynamic_shapes=(
            {1: sequence},
            None,
            None,
        ),
    )
    if _export_supports_external_data():
        kwargs["external_data"] = bool(external_data)
    torch.onnx.export(
        model,
        (tokens, conv, ssm),
        graph_path,
        **kwargs,
    )

    body: dict[str, object] = {
        "schema": VN97_MAMBA2_G06_SCHEMA,
        "capsule_id": capsule_id,
        "capsule_manifest_sha256": capsule_manifest_sha256,
        "source_weight_sha256": source_weight_sha256,
        "config": asdict(model.config),
        "opset": VN97_MAMBA2_G06_OPSET,
        "graph_filename": graph_name,
        "graph_files": _graph_files(root, graph_name),
        "batch_size": 1,
        "sequence_length_min": 1,
        "sequence_length_max": model.max_sequence_length,
        "inputs": list(VN97_MAMBA2_G06_INPUTS),
        "outputs": list(VN97_MAMBA2_G06_OUTPUTS),
        "logits_dtype": "float32",
        "state_contract": _state_contract(
            model.config,
            dtype=model.parameter_dtype,
        ),
        "single_weight_graph": True,
        "decode_sequence_length": 1,
        "parallel_prefill_ready": True,
        "parallel_algorithm": "dynamic_one_chunk_ssd_factorization",
        "same_weights_semantics": True,
        "quantization_used": False,
        "external_data_requested": bool(external_data),
        "production_activation_authorized": False,
    }
    manifest_id = hashlib.sha256(
        b"VN97M2G06ONNX1\0" + _canonical_json(body)
    ).hexdigest()
    manifest = {**body, "manifest_id": manifest_id}
    (root / "manifest.vn97m2g06.json").write_bytes(
        _canonical_json(manifest) + b"\n"
    )
    verify_g06_bundle(root)
    return manifest


def export_g03_capsule_dynamic_onnx(
    capsule_root: Path,
    output_dir: Path,
    *,
    max_sequence_length: int = VN97_MAMBA2_G06_MAX_SEQUENCE,
) -> dict[str, object]:
    capsule = load_g03_capsule(
        capsule_root,
        verify_large_weight_sha256=True,
    )
    step = VN97Mamba2StepOnnx(
        Mamba2OnnxConfig.from_source_spec(capsule.spec),
        capsule.tensors,
    )
    model = VN97Mamba2DynamicSequenceOnnx(
        step,
        max_sequence_length=max_sequence_length,
    )
    return export_dynamic_sequence_onnx(
        model,
        output_dir,
        capsule_id=capsule.manifest.capsule_id(),
        capsule_manifest_sha256=capsule.manifest_sha256,
        source_weight_sha256=capsule.manifest.source_weight_sha256,
        external_data=True,
    )


def verify_g06_bundle(root: Path) -> dict[str, object]:
    resolved = root.resolve(strict=True)
    path = resolved / "manifest.vn97m2g06.json"
    payload = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(payload, dict):
        raise ValueError("G0.6 manifest must be an object")
    if payload.get("schema") != VN97_MAMBA2_G06_SCHEMA:
        raise ValueError("G0.6 schema mismatch")
    manifest_id = _require_sha256(
        payload.get("manifest_id"),
        label="G0.6 manifest ID",
    )
    body = dict(payload)
    body.pop("manifest_id", None)
    expected = hashlib.sha256(
        b"VN97M2G06ONNX1\0" + _canonical_json(body)
    ).hexdigest()
    if manifest_id != expected:
        raise ValueError("G0.6 manifest identity mismatch")
    for field in (
        "capsule_id",
        "capsule_manifest_sha256",
        "source_weight_sha256",
    ):
        _require_sha256(payload.get(field), label=f"G0.6 {field}")
    if payload.get("batch_size") != 1:
        raise ValueError("G0.6 requires batch=1")
    minimum = payload.get("sequence_length_min")
    maximum = payload.get("sequence_length_max")
    if minimum != 1 or not isinstance(maximum, int) or maximum < 2:
        raise ValueError("G0.6 dynamic sequence bounds invalid")
    if payload.get("decode_sequence_length") != 1:
        raise ValueError("G0.6 decode sequence contract mismatch")
    if payload.get("logits_dtype") != "float32":
        raise ValueError("G0.6 sampler logits must be float32")
    if payload.get("single_weight_graph") is not True:
        raise ValueError("G0.6 must use one weight-owning graph")
    if payload.get("parallel_prefill_ready") is not True:
        raise ValueError("G0.6 parallel prefill must be enabled")
    if payload.get("same_weights_semantics") is not True:
        raise ValueError("G0.6 same-weight semantics mismatch")
    if payload.get("quantization_used") is not False:
        raise ValueError("G0.6 must not quantize inherited weights")
    if payload.get("production_activation_authorized") is not False:
        raise ValueError("G0.6 cannot self-authorize production")
    if payload.get("inputs") != list(VN97_MAMBA2_G06_INPUTS):
        raise ValueError("G0.6 input contract mismatch")
    if payload.get("outputs") != list(VN97_MAMBA2_G06_OUTPUTS):
        raise ValueError("G0.6 output contract mismatch")

    graph_name = payload.get("graph_filename")
    if graph_name != "recurrent-dynamic.onnx":
        raise ValueError("G0.6 graph filename mismatch")
    files = payload.get("graph_files")
    if not isinstance(files, list) or not files:
        raise ValueError("G0.6 graph inventory missing")
    seen = set()
    for record in files:
        if not isinstance(record, dict):
            raise ValueError("G0.6 graph record invalid")
        name = record.get("filename")
        if (
            not isinstance(name, str)
            or not name
            or "/" in name
            or "\\" in name
            or name in seen
        ):
            raise ValueError("G0.6 graph filename invalid")
        seen.add(name)
        file = resolved / name
        if file.is_symlink() or not file.is_file():
            raise ValueError(f"G0.6 graph file missing: {name}")
        if file.stat().st_size != record.get("bytes"):
            raise ValueError(f"G0.6 graph size mismatch: {name}")
        if _sha256_file(file) != record.get("sha256"):
            raise ValueError(f"G0.6 graph hash mismatch: {name}")
    if graph_name not in seen:
        raise ValueError("G0.6 primary graph missing")
    return payload


def run_ort_dynamic(
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
def validate_dynamic_native_parity(
    model: VN97Mamba2DynamicSequenceOnnx,
    *,
    sequence_length: int,
    nonzero_state: bool,
    seed: int = 97060,
) -> dict[str, float]:
    if not 1 <= sequence_length <= model.max_sequence_length:
        raise ValueError("G0.6 sequence length outside bounds")
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(
        0,
        model.config.vocab_size,
        (1, sequence_length),
        generator=generator,
        dtype=torch.long,
    )
    conv, ssm = model.initial_state()
    if nonzero_state:
        conv = torch.randn(
            conv.shape,
            generator=generator,
            dtype=conv.dtype,
        ) * 0.01
        ssm = torch.randn(
            ssm.shape,
            generator=generator,
            dtype=ssm.dtype,
        ) * 0.01

    actual = model(tokens, conv, ssm)
    expected = repeated_step_reference(
        model.step,
        tokens,
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


@torch.inference_mode()
def validate_dynamic_ort_parity(
    model: VN97Mamba2DynamicSequenceOnnx,
    bundle_dir: Path,
    *,
    sequence_length: int,
    seed: int = 97061,
) -> dict[str, float]:
    manifest = verify_g06_bundle(bundle_dir)
    if not 1 <= sequence_length <= int(manifest["sequence_length_max"]):
        raise ValueError("G0.6 ORT sequence length outside bounds")
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(
        0,
        model.config.vocab_size,
        (1, sequence_length),
        generator=generator,
        dtype=torch.long,
    )
    conv, ssm = model.initial_state()
    expected = repeated_step_reference(
        model.step,
        tokens,
        conv_state=conv,
        ssm_state=ssm,
    )
    actual = run_ort_dynamic(
        bundle_dir / "recurrent-dynamic.onnx",
        input_ids=tokens,
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
