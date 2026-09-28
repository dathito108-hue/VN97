from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba2_g03_capsule import load_g03_capsule
from .mamba2_kaggle_k2a_ort import _canonical_json, _load_json, _prompt_cases
from .mamba2_kaggle_k2b_provider_split import _epsilon_metrics
from .mamba2_kaggle_k2f_dbx_operands import (
    SCHEMA as K2F_SCHEMA,
    _export_onnx,
    _metrics,
    _run_ort_cpu,
    _verify_receipt,
)
from .mamba2_onnx import (
    Mamba2OnnxConfig,
    VN97Mamba2OnnxLayer,
    VN97Mamba2StepOnnx,
    verify_mamba2_g04_bundle,
)


SCHEMA = "VN97M2K2GBPATH1"

STAGES = (
    "normalized",
    "xbc_projected",
    "next_conv",
    "conv_affine",
    "activated_xbc",
    "b_value",
)


class VN97Mamba2BPathTrace(nn.Module):
    """Layer prefix exposing every B-producing boundary."""

    def __init__(self, layer: VN97Mamba2OnnxLayer) -> None:
        super().__init__()
        self.layer = layer

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        layer = self.layer
        cfg = layer.config

        residual_next = residual.float() + hidden.float()
        normalized = layer._rms_norm(
            residual_next.to(dtype=layer.block_norm.dtype),
            layer.block_norm,
        )

        zxbcdt = F.linear(normalized, layer.in_proj)
        _, xbc_projected, _ = torch.split(
            zxbcdt,
            [cfg.d_inner, cfg.conv_dim, cfg.n_heads],
            dim=-1,
        )

        next_conv = torch.roll(conv_state, shifts=-1, dims=-1)
        next_conv = torch.cat(
            (next_conv[..., :-1], xbc_projected.unsqueeze(-1)),
            dim=-1,
        )

        conv_affine = torch.sum(
            next_conv
            * layer.conv_weight[:, 0, :].to(
                device=next_conv.device,
                dtype=next_conv.dtype,
            ),
            dim=-1,
        )
        conv_affine = conv_affine + layer.conv_bias.to(
            device=conv_affine.device,
            dtype=conv_affine.dtype,
        )
        activated_xbc = F.silu(conv_affine)

        _, b_value, _ = torch.split(
            activated_xbc,
            [cfg.d_inner, cfg.d_state, cfg.d_state],
            dim=-1,
        )

        return (
            normalized,
            xbc_projected,
            next_conv,
            conv_affine,
            activated_xbc,
            b_value,
        )


def _b_from_normalized(
    layer: VN97Mamba2OnnxLayer,
    normalized: torch.Tensor,
    conv_state: torch.Tensor,
) -> torch.Tensor:
    cfg = layer.config
    zxbcdt = F.linear(normalized, layer.in_proj)
    _, xbc_projected, _ = torch.split(
        zxbcdt,
        [cfg.d_inner, cfg.conv_dim, cfg.n_heads],
        dim=-1,
    )
    return _b_from_xbc(layer, xbc_projected, conv_state)


def _b_from_xbc(
    layer: VN97Mamba2OnnxLayer,
    xbc_projected: torch.Tensor,
    conv_state: torch.Tensor,
) -> torch.Tensor:
    next_conv = torch.roll(conv_state, shifts=-1, dims=-1)
    next_conv = torch.cat(
        (next_conv[..., :-1], xbc_projected.unsqueeze(-1)),
        dim=-1,
    )
    return _b_from_next_conv(layer, next_conv)


def _b_from_next_conv(
    layer: VN97Mamba2OnnxLayer,
    next_conv: torch.Tensor,
) -> torch.Tensor:
    conv_affine = torch.sum(
        next_conv
        * layer.conv_weight[:, 0, :].to(
            device=next_conv.device,
            dtype=next_conv.dtype,
        ),
        dim=-1,
    )
    conv_affine = conv_affine + layer.conv_bias.to(
        device=conv_affine.device,
        dtype=conv_affine.dtype,
    )
    return _b_from_conv_affine(layer, conv_affine)


def _b_from_conv_affine(
    layer: VN97Mamba2OnnxLayer,
    conv_affine: torch.Tensor,
) -> torch.Tensor:
    return _b_from_activated(layer, F.silu(conv_affine))


def _b_from_activated(
    layer: VN97Mamba2OnnxLayer,
    activated_xbc: torch.Tensor,
) -> torch.Tensor:
    cfg = layer.config
    _, b_value, _ = torch.split(
        activated_xbc,
        [cfg.d_inner, cfg.d_state, cfg.d_state],
        dim=-1,
    )
    return b_value


def _dbx_from_b(
    dt_value: torch.Tensor,
    b_value: torch.Tensor,
    x_heads: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum(
        "bh,bn,bhp->bhpn",
        dt_value,
        b_value,
        x_heads,
    )


def _first_over(
    values: dict[str, dict[str, Any]],
    *,
    reference_units: float,
    order: tuple[str, ...],
) -> str | None:
    for name in order:
        block = values[name]
        if float(block["d_b_x"]["max_epsilon_units"]) > reference_units:
            return name
    return None


@torch.inference_mode()
def run_k2g(
    *,
    capsule_root: Path,
    bundle_dir: Path,
    k2f_receipt_path: Path,
    output_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    k2f = _verify_receipt(
        k2f_receipt_path,
        schema=K2F_SCHEMA,
        label="K2F receipt",
    )
    if k2f.get("diagnosis") != "upstream_operand_drift_amplification":
        raise ValueError(
            "K2G requires K2F upstream_operand_drift_amplification"
        )

    b_sensitivity = float(
        k2f["operand_sensitivity"]["ort_b_only"]["max_epsilon_units"]
    )
    dt_sensitivity = float(
        k2f["operand_sensitivity"]["ort_dt_only"]["max_epsilon_units"]
    )
    x_sensitivity = float(
        k2f["operand_sensitivity"]["ort_x_only"]["max_epsilon_units"]
    )
    reference_units = float(
        k2f["diagnostic_reference_max_epsilon_units"]
    )
    if b_sensitivity <= reference_units:
        raise ValueError("K2G requires B-driven dBx amplification evidence")
    if b_sensitivity <= max(dt_sensitivity, x_sensitivity):
        raise ValueError("K2G requires B to be the dominant isolated operand")

    manifest = verify_mamba2_g04_bundle(bundle_dir)
    for field in ("g04_manifest_id", "capsule_id", "source_weight_sha256"):
        manifest_field = "manifest_id" if field == "g04_manifest_id" else field
        if manifest.get(manifest_field) != k2f.get(field):
            raise ValueError(f"K2G {field} lineage mismatch")

    trace_layer = int(k2f["trace_layer"])
    epsilon = float(k2f["fp16_epsilon"])

    capsule = load_g03_capsule(
        capsule_root,
        verify_large_weight_sha256=False,
    )
    model = VN97Mamba2StepOnnx(
        Mamba2OnnxConfig.from_source_spec(capsule.spec),
        capsule.tensors,
    ).cpu().eval()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(capsule_root / "tokenizer"),
        local_files_only=True,
        use_fast=True,
    )
    prompt_cases = _prompt_cases(tokenizer)
    token_id = int(prompt_cases[0][0])
    if token_id != int(k2f["probe_token_id"]):
        raise ValueError("K2G probe token differs from K2F")

    token = torch.tensor([token_id], dtype=torch.long)
    conv, ssm = model.initial_state(1)
    hidden = F.embedding(token, model.embedding)
    residual = torch.zeros_like(hidden, dtype=torch.float32)

    for index in range(trace_layer):
        hidden, residual, _, _ = model.layers[index](
            hidden,
            residual,
            conv[index],
            ssm[index],
        )

    layer = model.layers[trace_layer]
    trace = VN97Mamba2BPathTrace(layer).cpu().eval()

    p_values = trace(hidden, residual, conv[trace_layer])
    p = {
        name: value.detach().cpu().numpy().copy()
        for name, value in zip(STAGES, p_values, strict=True)
    }

    # Exact PyTorch dBx reference and the dt/x operands are recomputed from
    # the canonical layer prefix so K2G changes only the B path.
    residual_next = residual.float() + hidden.float()
    normalized = layer._rms_norm(
        residual_next.to(dtype=layer.block_norm.dtype),
        layer.block_norm,
    )
    zxbcdt = F.linear(normalized, layer.in_proj)
    _, xbc_projected, dt = torch.split(
        zxbcdt,
        [
            layer.config.d_inner,
            layer.config.conv_dim,
            layer.config.n_heads,
        ],
        dim=-1,
    )
    next_conv = torch.roll(conv[trace_layer], shifts=-1, dims=-1)
    next_conv = torch.cat(
        (next_conv[..., :-1], xbc_projected.unsqueeze(-1)),
        dim=-1,
    )
    activated = torch.sum(
        next_conv
        * layer.conv_weight[:, 0, :].to(
            device=next_conv.device,
            dtype=next_conv.dtype,
        ),
        dim=-1,
    )
    activated = F.silu(
        activated
        + layer.conv_bias.to(
            device=activated.device,
            dtype=activated.dtype,
        )
    )
    x_value, p_b_value, _ = torch.split(
        activated,
        [
            layer.config.d_inner,
            layer.config.d_state,
            layer.config.d_state,
        ],
        dim=-1,
    )
    dt_value = F.softplus(
        dt
        + layer.dt_bias.to(
            device=dt.device,
            dtype=dt.dtype,
        )
    )
    x_heads = x_value.reshape(
        x_value.shape[0],
        layer.config.n_heads,
        layer.config.head_dim,
    )
    reference_dbx = _dbx_from_b(dt_value, p_b_value, x_heads)

    if output_dir.exists():
        if output_dir.is_symlink() or any(output_dir.iterdir()):
            raise ValueError("K2G output directory must be new or empty")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)

    graph_path = output_dir / f"layer-{trace_layer}-b-path.onnx"
    _export_onnx(
        trace,
        (hidden, residual, conv[trace_layer]),
        graph_path,
        input_names=["hidden", "residual", "conv_state"],
        output_names=list(STAGES),
    )
    actual = _run_ort_cpu(
        graph_path,
        inputs={
            "hidden": hidden.detach().numpy(),
            "residual": residual.detach().numpy(),
            "conv_state": conv[trace_layer].detach().numpy(),
        },
        outputs=list(STAGES),
    )
    o = {
        name: value
        for name, value in zip(STAGES, actual, strict=True)
    }

    stage_metrics = {
        name: _epsilon_metrics(
            p[name],
            o[name],
            epsilon=epsilon,
        )
        for name in STAGES
    }

    # Replay every ORT boundary through the remaining B path in PyTorch.
    candidate_b: dict[str, torch.Tensor] = {
        "normalized": _b_from_normalized(
            layer,
            torch.from_numpy(o["normalized"]),
            conv[trace_layer],
        ),
        "xbc_projected": _b_from_xbc(
            layer,
            torch.from_numpy(o["xbc_projected"]),
            conv[trace_layer],
        ),
        "next_conv": _b_from_next_conv(
            layer,
            torch.from_numpy(o["next_conv"]),
        ),
        "conv_affine": _b_from_conv_affine(
            layer,
            torch.from_numpy(o["conv_affine"]),
        ),
        "activated_xbc": _b_from_activated(
            layer,
            torch.from_numpy(o["activated_xbc"]),
        ),
        "b_value": torch.from_numpy(o["b_value"]),
    }

    counterfactuals: dict[str, dict[str, Any]] = {}
    for name in STAGES:
        b_candidate = candidate_b[name]
        candidate_dbx = _dbx_from_b(
            dt_value,
            b_candidate,
            x_heads,
        )
        counterfactuals[name] = {
            "b_value": _metrics(
                p_b_value,
                b_candidate,
                epsilon=epsilon,
            ),
            "d_b_x": _metrics(
                reference_dbx,
                candidate_dbx,
                epsilon=epsilon,
            ),
        }

    first_sensitive = _first_over(
        counterfactuals,
        reference_units=reference_units,
        order=STAGES,
    )

    if first_sensitive == "normalized":
        diagnosis = "normalization_boundary_sensitive"
    elif first_sensitive == "xbc_projected":
        diagnosis = "in_proj_xbc_rounding_amplified"
    elif first_sensitive == "next_conv":
        diagnosis = "conv_state_assembly_rounding_amplified"
    elif first_sensitive == "conv_affine":
        diagnosis = "conv_affine_rounding_amplified"
    elif first_sensitive == "activated_xbc":
        diagnosis = "silu_rounding_amplified"
    elif first_sensitive == "b_value":
        diagnosis = "b_slice_only"
    else:
        diagnosis = "b_path_amplification_not_reproduced"

    body: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "MEASURED",
        "k2f_receipt_id": k2f["receipt_id"],
        "g04_manifest_id": manifest["manifest_id"],
        "capsule_id": manifest["capsule_id"],
        "source_weight_sha256": manifest["source_weight_sha256"],
        "trace_layer": trace_layer,
        "probe_token_id": token_id,
        "provider": "CPUExecutionProvider",
        "dtype": str(manifest["state_contract"]["dtype"]),
        "fp16_epsilon": epsilon,
        "diagnostic_reference_max_epsilon_units": reference_units,
        "k2f_b_sensitivity_max_epsilon_units": b_sensitivity,
        "stage_order": list(STAGES),
        "stage_metrics": stage_metrics,
        "counterfactuals": counterfactuals,
        "first_sensitive_boundary": first_sensitive,
        "diagnosis": diagnosis,
        "weights_changed": False,
        "architecture_changed": False,
        "acceptance_threshold_defined": False,
        "production_activation_authorized": False,
    }
    body["receipt_id"] = hashlib.sha256(
        SCHEMA.encode("ascii") + b"\0" + _canonical_json(body)
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_json(body) + b"\n")

    del trace
    del layer
    del model
    del capsule
    gc.collect()
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capsule-root", required=True, type=Path)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--k2f-receipt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/vn97-k2g-b-path.json"),
    )
    args = parser.parse_args()

    receipt = run_k2g(
        capsule_root=args.capsule_root.resolve(strict=True),
        bundle_dir=args.bundle_dir.resolve(strict=True),
        k2f_receipt_path=args.k2f_receipt.resolve(strict=True),
        output_dir=args.output_dir,
        output_path=args.output,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
