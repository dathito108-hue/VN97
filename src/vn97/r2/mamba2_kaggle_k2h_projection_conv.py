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
from .mamba2_kaggle_k2a_ort import _canonical_json, _prompt_cases
from .mamba2_kaggle_k2f_dbx_operands import (
    _export_onnx,
    _metrics,
    _run_ort_cpu,
    _verify_receipt,
)
from .mamba2_kaggle_k2g_b_path import (
    SCHEMA as K2G_SCHEMA,
    _b_from_conv_affine,
    _b_from_xbc,
    _dbx_from_b,
)
from .mamba2_onnx import (
    Mamba2OnnxConfig,
    VN97Mamba2OnnxLayer,
    VN97Mamba2StepOnnx,
    verify_mamba2_g04_bundle,
)


SCHEMA = "VN97M2K2HMICRO1"


class VN97Mamba2InProjMicro(nn.Module):
    def __init__(self, layer: VN97Mamba2OnnxLayer, mode: str) -> None:
        super().__init__()
        if mode not in {"fp16_linear", "fp16_matmul", "fp32_widened"}:
            raise ValueError(f"unsupported K2H in-proj mode: {mode}")
        self.layer = layer
        self.mode = mode

    def forward(self, normalized: torch.Tensor) -> torch.Tensor:
        cfg = self.layer.config
        weight = self.layer.in_proj
        if self.mode == "fp16_linear":
            projected = F.linear(normalized, weight)
        elif self.mode == "fp16_matmul":
            projected = torch.matmul(normalized, weight.transpose(0, 1))
        else:
            projected = F.linear(
                normalized.float(),
                weight.float(),
            ).to(dtype=normalized.dtype)
        _, xbc, _ = torch.split(
            projected,
            [cfg.d_inner, cfg.conv_dim, cfg.n_heads],
            dim=-1,
        )
        return xbc


class VN97Mamba2ConvAffineMicro(nn.Module):
    def __init__(self, layer: VN97Mamba2OnnxLayer, mode: str) -> None:
        super().__init__()
        if mode not in {"fp16_reduce", "fp32_widened"}:
            raise ValueError(f"unsupported K2H conv mode: {mode}")
        self.layer = layer
        self.mode = mode

    def forward(self, next_conv: torch.Tensor) -> torch.Tensor:
        weight = self.layer.conv_weight[:, 0, :]
        bias = self.layer.conv_bias
        if self.mode == "fp16_reduce":
            out = torch.sum(
                next_conv
                * weight.to(
                    device=next_conv.device,
                    dtype=next_conv.dtype,
                ),
                dim=-1,
            )
            return out + bias.to(
                device=out.device,
                dtype=out.dtype,
            )

        out = torch.sum(
            next_conv.float()
            * weight.float().to(device=next_conv.device),
            dim=-1,
        )
        out = out + bias.float().to(device=out.device)
        return out.to(dtype=next_conv.dtype)


def _choose_candidate(
    metrics: dict[str, dict[str, Any]],
    *,
    reference_units: float,
) -> str | None:
    passing = [
        (name, float(block["d_b_x"]["max_epsilon_units"]))
        for name, block in metrics.items()
        if float(block["d_b_x"]["max_epsilon_units"]) <= reference_units
    ]
    if not passing:
        return None
    passing.sort(key=lambda item: (item[1], item[0]))
    return passing[0][0]


@torch.inference_mode()
def run_k2h(
    *,
    capsule_root: Path,
    bundle_dir: Path,
    k2g_receipt_path: Path,
    output_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    k2g = _verify_receipt(
        k2g_receipt_path,
        schema=K2G_SCHEMA,
        label="K2G receipt",
    )
    if k2g.get("diagnosis") != "in_proj_xbc_rounding_amplified":
        raise ValueError("K2H requires K2G in_proj_xbc_rounding_amplified")
    if k2g.get("first_sensitive_boundary") != "xbc_projected":
        raise ValueError("K2H requires xbc_projected first-sensitive boundary")

    manifest = verify_mamba2_g04_bundle(bundle_dir)
    for field in ("g04_manifest_id", "capsule_id", "source_weight_sha256"):
        manifest_field = "manifest_id" if field == "g04_manifest_id" else field
        if manifest.get(manifest_field) != k2g.get(field):
            raise ValueError(f"K2H {field} lineage mismatch")

    trace_layer = int(k2g["trace_layer"])
    epsilon = float(k2g["fp16_epsilon"])
    reference_units = float(
        k2g["diagnostic_reference_max_epsilon_units"]
    )

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
    token_id = int(_prompt_cases(tokenizer)[0][0])
    if token_id != int(k2g["probe_token_id"]):
        raise ValueError("K2H probe token differs from K2G")

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
    cfg = layer.config

    residual_next = residual.float() + hidden.float()
    normalized = layer._rms_norm(
        residual_next.to(dtype=layer.block_norm.dtype),
        layer.block_norm,
    )
    projected = F.linear(normalized, layer.in_proj)
    _, reference_xbc, dt = torch.split(
        projected,
        [cfg.d_inner, cfg.conv_dim, cfg.n_heads],
        dim=-1,
    )

    reference_next_conv = torch.roll(
        conv[trace_layer],
        shifts=-1,
        dims=-1,
    )
    reference_next_conv = torch.cat(
        (
            reference_next_conv[..., :-1],
            reference_xbc.unsqueeze(-1),
        ),
        dim=-1,
    )
    reference_conv_affine = VN97Mamba2ConvAffineMicro(
        layer,
        "fp16_reduce",
    )(reference_next_conv)
    reference_activated = F.silu(reference_conv_affine)
    x_value, reference_b, _ = torch.split(
        reference_activated,
        [cfg.d_inner, cfg.d_state, cfg.d_state],
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
        cfg.n_heads,
        cfg.head_dim,
    )
    reference_dbx = _dbx_from_b(
        dt_value,
        reference_b,
        x_heads,
    )

    if output_dir.exists():
        if output_dir.is_symlink() or any(output_dir.iterdir()):
            raise ValueError("K2H output directory must be new or empty")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)

    in_proj_metrics: dict[str, dict[str, Any]] = {}
    for mode in ("fp16_linear", "fp16_matmul", "fp32_widened"):
        micro = VN97Mamba2InProjMicro(layer, mode).cpu().eval()
        graph_path = output_dir / f"in-proj-{mode}.onnx"
        _export_onnx(
            micro,
            (normalized,),
            graph_path,
            input_names=["normalized"],
            output_names=["xbc_projected"],
        )
        ort_xbc = _run_ort_cpu(
            graph_path,
            inputs={"normalized": normalized.detach().numpy()},
            outputs=["xbc_projected"],
        )[0]
        ort_xbc_t = torch.from_numpy(ort_xbc)
        b_candidate = _b_from_xbc(
            layer,
            ort_xbc_t,
            conv[trace_layer],
        )
        dbx_candidate = _dbx_from_b(
            dt_value,
            b_candidate,
            x_heads,
        )
        in_proj_metrics[mode] = {
            "xbc_projected": _metrics(
                reference_xbc,
                ort_xbc,
                epsilon=epsilon,
            ),
            "b_value": _metrics(
                reference_b,
                b_candidate,
                epsilon=epsilon,
            ),
            "d_b_x": _metrics(
                reference_dbx,
                dbx_candidate,
                epsilon=epsilon,
            ),
        }
        del micro

    conv_metrics: dict[str, dict[str, Any]] = {}
    for mode in ("fp16_reduce", "fp32_widened"):
        micro = VN97Mamba2ConvAffineMicro(layer, mode).cpu().eval()
        graph_path = output_dir / f"conv-affine-{mode}.onnx"
        _export_onnx(
            micro,
            (reference_next_conv,),
            graph_path,
            input_names=["next_conv"],
            output_names=["conv_affine"],
        )
        ort_affine = _run_ort_cpu(
            graph_path,
            inputs={
                "next_conv": reference_next_conv.detach().numpy(),
            },
            outputs=["conv_affine"],
        )[0]
        ort_affine_t = torch.from_numpy(ort_affine)
        b_candidate = _b_from_conv_affine(
            layer,
            ort_affine_t,
        )
        dbx_candidate = _dbx_from_b(
            dt_value,
            b_candidate,
            x_heads,
        )
        conv_metrics[mode] = {
            "conv_affine": _metrics(
                reference_conv_affine,
                ort_affine,
                epsilon=epsilon,
            ),
            "b_value": _metrics(
                reference_b,
                b_candidate,
                epsilon=epsilon,
            ),
            "d_b_x": _metrics(
                reference_dbx,
                dbx_candidate,
                epsilon=epsilon,
            ),
        }
        del micro

    in_proj_candidate = _choose_candidate(
        in_proj_metrics,
        reference_units=reference_units,
    )
    conv_candidate = _choose_candidate(
        conv_metrics,
        reference_units=reference_units,
    )

    if in_proj_candidate and conv_candidate:
        diagnosis = "both_boundaries_have_bounded_candidates"
    elif in_proj_candidate:
        diagnosis = "in_proj_candidate_only"
    elif conv_candidate:
        diagnosis = "conv_candidate_only"
    else:
        diagnosis = "no_bounded_micro_candidate"

    body: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "MEASURED",
        "k2g_receipt_id": k2g["receipt_id"],
        "g04_manifest_id": manifest["manifest_id"],
        "capsule_id": manifest["capsule_id"],
        "source_weight_sha256": manifest["source_weight_sha256"],
        "trace_layer": trace_layer,
        "probe_token_id": token_id,
        "provider": "CPUExecutionProvider",
        "dtype": str(manifest["state_contract"]["dtype"]),
        "fp16_epsilon": epsilon,
        "diagnostic_reference_max_epsilon_units": reference_units,
        "in_proj_metrics": in_proj_metrics,
        "conv_affine_metrics": conv_metrics,
        "in_proj_candidate": in_proj_candidate,
        "conv_affine_candidate": conv_candidate,
        "diagnosis": diagnosis,
        "weights_changed": False,
        "architecture_changed": False,
        "production_graph_changed": False,
        "acceptance_threshold_defined": False,
        "production_activation_authorized": False,
    }
    body["receipt_id"] = hashlib.sha256(
        SCHEMA.encode("ascii") + b"\0" + _canonical_json(body)
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_json(body) + b"\n")

    del layer
    del model
    del capsule
    gc.collect()
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capsule-root", required=True, type=Path)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--k2g-receipt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/vn97-k2h-micro-isolation.json"),
    )
    args = parser.parse_args()

    receipt = run_k2h(
        capsule_root=args.capsule_root.resolve(strict=True),
        bundle_dir=args.bundle_dir.resolve(strict=True),
        k2g_receipt_path=args.k2g_receipt.resolve(strict=True),
        output_dir=args.output_dir,
        output_path=args.output,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
