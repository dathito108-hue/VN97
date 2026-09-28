from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba2_g03_capsule import load_g03_capsule
from .mamba2_kaggle_k2a_ort import _canonical_json, _load_json, _prompt_cases
from .mamba2_onnx import (
    Mamba2OnnxConfig,
    VN97Mamba2OnnxLayer,
    VN97Mamba2StepOnnx,
    _mamba2_d_b_x_activation,
    verify_mamba2_g04_bundle,
)


SCHEMA = "VN97M2K2DSTAGE1"
K2C_SCHEMA = "VN97M2K2CBACKEND1"

STAGE_NAMES = (
    "residual_next",
    "block_norm",
    "in_proj",
    "next_conv",
    "conv_affine",
    "activated_xbc",
    "dt_value",
    "d_a",
    "d_b_x",
    "next_ssm",
    "y_pre_gate",
    "gated_norm",
    "output",
)


class VN97Mamba2OnnxLayerTrace(nn.Module):
    """Diagnostic copy of one G0.4 layer with ordered intermediate outputs."""

    def __init__(self, layer: VN97Mamba2OnnxLayer) -> None:
        super().__init__()
        self.layer = layer

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        layer = self.layer
        cfg = layer.config

        residual_next = residual.float() + hidden.float()
        normalized = layer._rms_norm(
            residual_next.to(dtype=layer.block_norm.dtype),
            layer.block_norm,
        )

        zxbcdt = F.linear(normalized, layer.in_proj)
        z, xbc, dt = torch.split(
            zxbcdt,
            [cfg.d_inner, cfg.conv_dim, cfg.n_heads],
            dim=-1,
        )

        next_conv = torch.roll(conv_state, shifts=-1, dims=-1)
        next_conv = torch.cat(
            (next_conv[..., :-1], xbc.unsqueeze(-1)),
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

        x, b_value, c_value = torch.split(
            activated_xbc,
            [cfg.d_inner, cfg.d_state, cfg.d_state],
            dim=-1,
        )

        a = -torch.exp(layer.a_log.float())
        dt_value = F.softplus(
            dt
            + layer.dt_bias.to(
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
        d_b_x = _mamba2_d_b_x_activation(
            dt_value,
            b_value,
            x_heads,
        )
        next_ssm = (
            ssm_state.float()
            * d_a[:, :, None, None]
            + d_b_x
        ).to(dtype=ssm_state.dtype)

        y_pre_gate = torch.einsum(
            "bhpn,bn->bhp",
            next_ssm.to(dtype=x_heads.dtype),
            c_value.to(dtype=x_heads.dtype),
        )
        y_pre_gate = (
            y_pre_gate
            + layer.d_skip.to(
                device=y_pre_gate.device,
                dtype=y_pre_gate.dtype,
            )[None, :, None]
            * x_heads
        ).reshape(x.shape[0], cfg.d_inner)

        gated_norm = layer._gated_rms_norm(y_pre_gate, z)
        output = F.linear(gated_norm, layer.out_proj)

        return (
            residual_next,
            normalized,
            zxbcdt,
            next_conv,
            conv_affine,
            activated_xbc,
            dt_value,
            d_a,
            d_b_x,
            next_ssm,
            y_pre_gate,
            gated_norm,
            output,
        )


def _epsilon_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    epsilon: float,
) -> dict[str, float]:
    left = reference.astype(np.float32, copy=False)
    right = candidate.astype(np.float32, copy=False)
    diff = np.abs(left - right)
    scale = np.maximum(1.0, np.abs(left))
    units = diff / (epsilon * scale)
    return {
        "max_abs_error": float(diff.max(initial=0.0)),
        "mean_abs_error": float(diff.mean()) if diff.size else 0.0,
        "max_epsilon_units": float(units.max(initial=0.0)),
    }


def _verify_k2c_receipt(path: Path) -> dict[str, Any]:
    receipt = _load_json(path, "K2C receipt")
    if receipt.get("schema") != K2C_SCHEMA:
        raise ValueError("K2D requires K2C backend-matrix receipt")
    if receipt.get("status") != "MEASURED":
        raise ValueError("K2D requires K2C status MEASURED")
    if receipt.get("acceptance_threshold_defined") is not False:
        raise ValueError("K2C must remain diagnostic-only")
    if receipt.get("production_activation_authorized") is not False:
        raise ValueError("K2C must not authorize production")
    receipt_id = receipt.get("receipt_id")
    if not isinstance(receipt_id, str) or len(receipt_id) != 64:
        raise ValueError("K2D K2C receipt identity invalid")
    body = dict(receipt)
    body.pop("receipt_id", None)
    expected = hashlib.sha256(
        K2C_SCHEMA.encode("ascii") + b"\0" + _canonical_json(body)
    ).hexdigest()
    if receipt_id != expected:
        raise ValueError("K2D K2C receipt identity mismatch")
    return receipt


def _first_onnx_cpu_layer(k2c: dict[str, Any]) -> int:
    layerwise = k2c.get("layerwise_state")
    if not isinstance(layerwise, dict):
        raise ValueError("K2D K2C layerwise evidence missing")
    cpu = layerwise.get("pytorch_cpu_vs_ort_cpu")
    if not isinstance(cpu, dict):
        raise ValueError("K2D K2C CPU layerwise evidence missing")
    value = cpu.get("first_layer_over_reference")
    if not isinstance(value, int) or value < 0 or value >= 64:
        raise ValueError("K2D requires a concrete divergent ONNX CPU layer")
    return value


def _export_supports_external_data() -> bool:
    try:
        return "external_data" in inspect.signature(
            torch.onnx.export
        ).parameters
    except (TypeError, ValueError):
        return False


@torch.inference_mode()
def run_k2d(
    *,
    capsule_root: Path,
    bundle_dir: Path,
    k2c_receipt_path: Path,
    output_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    k2c = _verify_k2c_receipt(k2c_receipt_path)
    manifest = verify_mamba2_g04_bundle(bundle_dir)
    for field in ("g04_manifest_id", "capsule_id", "source_weight_sha256"):
        manifest_field = "manifest_id" if field == "g04_manifest_id" else field
        if manifest.get(manifest_field) != k2c.get(field):
            raise ValueError(f"K2D {field} lineage mismatch")

    trace_layer = _first_onnx_cpu_layer(k2c)
    diagnostic_reference_units = float(
        k2c["diagnostic_reference_max_epsilon_units"]
    )
    epsilon = float(k2c["fp16_epsilon"])

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
    trace = VN97Mamba2OnnxLayerTrace(layer).cpu().eval()

    reference_tensors = trace(
        hidden,
        residual,
        conv[trace_layer],
        ssm[trace_layer],
    )
    reference = {
        name: tensor.detach().cpu().numpy().copy()
        for name, tensor in zip(STAGE_NAMES, reference_tensors, strict=True)
    }

    original = layer(
        hidden,
        residual,
        conv[trace_layer],
        ssm[trace_layer],
    )
    if not torch.equal(reference_tensors[-1], original[0]):
        raise RuntimeError("K2D trace wrapper output differs from G0.4 layer")
    if not torch.equal(reference_tensors[0], original[1]):
        raise RuntimeError("K2D trace wrapper residual differs from G0.4 layer")
    if not torch.equal(reference_tensors[3], original[2]):
        raise RuntimeError("K2D trace wrapper conv state differs from G0.4 layer")
    if not torch.equal(reference_tensors[9], original[3]):
        raise RuntimeError("K2D trace wrapper SSM state differs from G0.4 layer")

    if output_dir.exists():
        if output_dir.is_symlink() or any(output_dir.iterdir()):
            raise ValueError("K2D trace output directory must be new or empty")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)

    graph_path = output_dir / f"layer-{trace_layer}-trace.onnx"
    kwargs: dict[str, Any] = {
        "export_params": True,
        "opset_version": 18,
        "do_constant_folding": True,
        "input_names": ["hidden", "residual", "conv_state", "ssm_state"],
        "output_names": list(STAGE_NAMES),
        "dynamo": True,
    }
    if _export_supports_external_data():
        kwargs["external_data"] = True

    torch.onnx.export(
        trace,
        (
            hidden,
            residual,
            conv[trace_layer],
            ssm[trace_layer],
        ),
        graph_path,
        **kwargs,
    )

    import onnxruntime as ort

    session = ort.InferenceSession(
        str(graph_path),
        providers=["CPUExecutionProvider"],
    )
    active = session.get_providers()
    if not active or active[0] != "CPUExecutionProvider":
        raise RuntimeError(
            "K2D trace session did not activate CPUExecutionProvider"
        )
    actual_values = session.run(
        list(STAGE_NAMES),
        {
            "hidden": hidden.detach().numpy(),
            "residual": residual.detach().numpy(),
            "conv_state": conv[trace_layer].detach().numpy(),
            "ssm_state": ssm[trace_layer].detach().numpy(),
        },
    )

    stage_metrics: dict[str, dict[str, float]] = {}
    first_stage_over: dict[str, Any] | None = None
    for name, actual in zip(STAGE_NAMES, actual_values, strict=True):
        metrics = _epsilon_metrics(
            reference[name],
            np.asarray(actual),
            epsilon=epsilon,
        )
        stage_metrics[name] = metrics
        if (
            first_stage_over is None
            and metrics["max_epsilon_units"] > diagnostic_reference_units
        ):
            first_stage_over = {
                "stage": name,
                **metrics,
            }

    body: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "MEASURED",
        "k2c_receipt_id": k2c["receipt_id"],
        "g04_manifest_id": manifest["manifest_id"],
        "capsule_id": manifest["capsule_id"],
        "source_weight_sha256": manifest["source_weight_sha256"],
        "trace_layer": trace_layer,
        "probe_token_id": token_id,
        "provider": active[0],
        "dtype": str(manifest["state_contract"]["dtype"]),
        "fp16_epsilon": epsilon,
        "diagnostic_reference_max_epsilon_units":
            diagnostic_reference_units,
        "stage_order": list(STAGE_NAMES),
        "stage_metrics": stage_metrics,
        "first_stage_over_reference": first_stage_over,
        "trace_wrapper_matches_g04_layer": True,
        "acceptance_threshold_defined": False,
        "production_activation_authorized": False,
    }
    body["receipt_id"] = hashlib.sha256(
        SCHEMA.encode("ascii") + b"\0" + _canonical_json(body)
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_json(body) + b"\n")

    del session
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
    parser.add_argument("--k2c-receipt", required=True, type=Path)
    parser.add_argument("--trace-output-dir", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/vn97-k2d-stage-trace.json"),
    )
    args = parser.parse_args()

    receipt = run_k2d(
        capsule_root=args.capsule_root.resolve(strict=True),
        bundle_dir=args.bundle_dir.resolve(strict=True),
        k2c_receipt_path=args.k2c_receipt.resolve(strict=True),
        output_dir=args.trace_output_dir,
        output_path=args.output,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
