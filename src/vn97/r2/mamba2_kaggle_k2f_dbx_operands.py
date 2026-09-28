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
from .mamba2_kaggle_k2b_provider_split import _epsilon_metrics
from .mamba2_onnx import (
    Mamba2OnnxConfig,
    VN97Mamba2OnnxLayer,
    VN97Mamba2StepOnnx,
    verify_mamba2_g04_bundle,
)


SCHEMA = "VN97M2K2FOPERAND1"
K2D_SCHEMA = "VN97M2K2DSTAGE1"
K2E_SCHEMA = "VN97M2K2EDBX1"


class VN97Mamba2DbxOperandPrefix(nn.Module):
    """Same-input layer prefix ending at the exact dBx operands."""

    def __init__(self, layer: VN97Mamba2OnnxLayer) -> None:
        super().__init__()
        self.layer = layer

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        layer = self.layer
        cfg = layer.config

        residual_next = residual.float() + hidden.float()
        normalized = layer._rms_norm(
            residual_next.to(dtype=layer.block_norm.dtype),
            layer.block_norm,
        )

        zxbcdt = F.linear(normalized, layer.in_proj)
        _, xbc, dt = torch.split(
            zxbcdt,
            [cfg.d_inner, cfg.conv_dim, cfg.n_heads],
            dim=-1,
        )

        next_conv = torch.roll(conv_state, shifts=-1, dims=-1)
        next_conv = torch.cat(
            (next_conv[..., :-1], xbc.unsqueeze(-1)),
            dim=-1,
        )
        activated_xbc = torch.sum(
            next_conv
            * layer.conv_weight[:, 0, :].to(
                device=next_conv.device,
                dtype=next_conv.dtype,
            ),
            dim=-1,
        )
        activated_xbc = F.silu(
            activated_xbc
            + layer.conv_bias.to(
                device=activated_xbc.device,
                dtype=activated_xbc.dtype,
            )
        )

        x, b_value, _ = torch.split(
            activated_xbc,
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
        x_heads = x.reshape(
            x.shape[0],
            cfg.n_heads,
            cfg.head_dim,
        )
        return dt_value, b_value, x_heads


class VN97Mamba2DbxMicro(nn.Module):
    def __init__(self, mode: str) -> None:
        super().__init__()
        if mode not in {
            "einsum",
            "mul_dt_x_b",
            "mul_b_x_dt",
            "fp32_widened",
        }:
            raise ValueError(f"unsupported K2F dBx mode: {mode}")
        self.mode = mode

    def forward(
        self,
        dt_value: torch.Tensor,
        b_value: torch.Tensor,
        x_heads: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "einsum":
            return torch.einsum(
                "bh,bn,bhp->bhpn",
                dt_value,
                b_value,
                x_heads,
            )
        if self.mode == "mul_dt_x_b":
            return (
                dt_value[:, :, None, None]
                * x_heads[:, :, :, None]
            ) * b_value[:, None, None, :]
        if self.mode == "mul_b_x_dt":
            return (
                b_value[:, None, None, :]
                * x_heads[:, :, :, None]
            ) * dt_value[:, :, None, None]
        widened = (
            dt_value.float()[:, :, None, None]
            * x_heads.float()[:, :, :, None]
            * b_value.float()[:, None, None, :]
        )
        return widened.to(dtype=dt_value.dtype)


def _verify_receipt(
    path: Path,
    *,
    schema: str,
    label: str,
) -> dict[str, Any]:
    receipt = _load_json(path, label)
    if receipt.get("schema") != schema:
        raise ValueError(f"K2F {label} schema mismatch")
    if receipt.get("status") != "MEASURED":
        raise ValueError(f"K2F {label} must be MEASURED")
    if receipt.get("acceptance_threshold_defined") is not False:
        raise ValueError(f"K2F {label} must remain diagnostic-only")
    if receipt.get("production_activation_authorized") is not False:
        raise ValueError(f"K2F {label} must not authorize production")
    receipt_id = receipt.get("receipt_id")
    if not isinstance(receipt_id, str) or len(receipt_id) != 64:
        raise ValueError(f"K2F {label} receipt identity invalid")
    body = dict(receipt)
    body.pop("receipt_id", None)
    expected = hashlib.sha256(
        schema.encode("ascii") + b"\0" + _canonical_json(body)
    ).hexdigest()
    if receipt_id != expected:
        raise ValueError(f"K2F {label} receipt identity mismatch")
    return receipt


def _export_supports_external_data() -> bool:
    try:
        return "external_data" in inspect.signature(
            torch.onnx.export
        ).parameters
    except (TypeError, ValueError):
        return False


def _export_onnx(
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    path: Path,
    *,
    input_names: list[str],
    output_names: list[str],
) -> None:
    kwargs: dict[str, Any] = {
        "export_params": True,
        "opset_version": 18,
        "do_constant_folding": True,
        "input_names": input_names,
        "output_names": output_names,
        "dynamo": True,
    }
    if _export_supports_external_data():
        kwargs["external_data"] = True
    torch.onnx.export(
        module.cpu().eval(),
        args,
        path,
        **kwargs,
    )


def _run_ort_cpu(
    path: Path,
    *,
    inputs: dict[str, np.ndarray],
    outputs: list[str],
) -> list[np.ndarray]:
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )
    active = session.get_providers()
    if not active or active[0] != "CPUExecutionProvider":
        raise RuntimeError("K2F ORT CPU provider did not activate")
    values = session.run(outputs, inputs)
    del session
    gc.collect()
    return [np.asarray(value).copy() for value in values]


def _metrics(
    reference: torch.Tensor | np.ndarray,
    candidate: torch.Tensor | np.ndarray,
    *,
    epsilon: float,
) -> dict[str, float]:
    left = (
        reference.detach().cpu().numpy()
        if isinstance(reference, torch.Tensor)
        else np.asarray(reference)
    )
    right = (
        candidate.detach().cpu().numpy()
        if isinstance(candidate, torch.Tensor)
        else np.asarray(candidate)
    )
    return _epsilon_metrics(left, right, epsilon=epsilon)


def _max_units(metrics: dict[str, float]) -> float:
    return float(metrics["max_epsilon_units"])


@torch.inference_mode()
def run_k2f(
    *,
    capsule_root: Path,
    bundle_dir: Path,
    k2d_receipt_path: Path,
    k2e_receipt_path: Path,
    output_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    k2d = _verify_receipt(
        k2d_receipt_path,
        schema=K2D_SCHEMA,
        label="K2D receipt",
    )
    k2e = _verify_receipt(
        k2e_receipt_path,
        schema=K2E_SCHEMA,
        label="K2E receipt",
    )
    if k2e.get("k2d_before_receipt_id") != k2d.get("receipt_id"):
        raise ValueError("K2F K2E/K2D lineage mismatch")
    if k2e.get("d_b_x_within_reference_after") is not False:
        raise ValueError("K2F requires failed K2E dBx repair evidence")

    before = float(k2e["before_d_b_x"]["max_epsilon_units"])
    after = float(k2e["after_d_b_x"]["max_epsilon_units"])
    if abs(before - after) > 1.0e-12:
        raise ValueError(
            "K2F expects K2E no-improvement evidence for this campaign"
        )

    manifest = verify_mamba2_g04_bundle(bundle_dir)
    for field in ("g04_manifest_id", "capsule_id", "source_weight_sha256"):
        manifest_field = "manifest_id" if field == "g04_manifest_id" else field
        if manifest.get(manifest_field) != k2d.get(field):
            raise ValueError(f"K2F {field} lineage mismatch")

    trace_layer = int(k2d["trace_layer"])
    epsilon = float(k2d["fp16_epsilon"])
    diagnostic_reference_units = float(
        k2d["diagnostic_reference_max_epsilon_units"]
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
    prompt_cases = _prompt_cases(tokenizer)
    token_id = int(prompt_cases[0][0])
    if token_id != int(k2d["probe_token_id"]):
        raise ValueError("K2F probe token differs from K2D")
    token = torch.tensor([token_id], dtype=torch.long)

    conv, _ = model.initial_state(1)
    hidden = F.embedding(token, model.embedding)
    residual = torch.zeros_like(hidden, dtype=torch.float32)

    # Match K2D: preceding layers execute only in PyTorch CPU.
    _, ssm = model.initial_state(1)
    for index in range(trace_layer):
        hidden, residual, _, _ = model.layers[index](
            hidden,
            residual,
            conv[index],
            ssm[index],
        )

    layer = model.layers[trace_layer]
    prefix = VN97Mamba2DbxOperandPrefix(layer).cpu().eval()
    p_dt, p_b, p_x = prefix(
        hidden,
        residual,
        conv[trace_layer],
    )

    if output_dir.exists():
        if output_dir.is_symlink() or any(output_dir.iterdir()):
            raise ValueError("K2F output directory must be new or empty")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)

    prefix_path = output_dir / f"layer-{trace_layer}-dbx-operands.onnx"
    _export_onnx(
        prefix,
        (hidden, residual, conv[trace_layer]),
        prefix_path,
        input_names=["hidden", "residual", "conv_state"],
        output_names=["dt_value", "b_value", "x_heads"],
    )
    o_dt, o_b, o_x = _run_ort_cpu(
        prefix_path,
        inputs={
            "hidden": hidden.detach().numpy(),
            "residual": residual.detach().numpy(),
            "conv_state": conv[trace_layer].detach().numpy(),
        },
        outputs=["dt_value", "b_value", "x_heads"],
    )

    operand_metrics = {
        "dt_value": _metrics(p_dt, o_dt, epsilon=epsilon),
        "b_value": _metrics(p_b, o_b, epsilon=epsilon),
        "x_heads": _metrics(p_x, o_x, epsilon=epsilon),
    }

    reference = torch.einsum(
        "bh,bn,bhp->bhpn",
        p_dt,
        p_b,
        p_x,
    )

    o_dt_t = torch.from_numpy(o_dt)
    o_b_t = torch.from_numpy(o_b)
    o_x_t = torch.from_numpy(o_x)

    sensitivity = {
        "ort_dt_only": _metrics(
            reference,
            torch.einsum(
                "bh,bn,bhp->bhpn",
                o_dt_t,
                p_b,
                p_x,
            ),
            epsilon=epsilon,
        ),
        "ort_b_only": _metrics(
            reference,
            torch.einsum(
                "bh,bn,bhp->bhpn",
                p_dt,
                o_b_t,
                p_x,
            ),
            epsilon=epsilon,
        ),
        "ort_x_only": _metrics(
            reference,
            torch.einsum(
                "bh,bn,bhp->bhpn",
                p_dt,
                p_b,
                o_x_t,
            ),
            epsilon=epsilon,
        ),
        "all_ort_operands_pytorch_einsum": _metrics(
            reference,
            torch.einsum(
                "bh,bn,bhp->bhpn",
                o_dt_t,
                o_b_t,
                o_x_t,
            ),
            epsilon=epsilon,
        ),
    }

    micro_metrics: dict[str, dict[str, Any]] = {}
    for mode in (
        "einsum",
        "mul_dt_x_b",
        "mul_b_x_dt",
        "fp32_widened",
    ):
        micro = VN97Mamba2DbxMicro(mode).cpu().eval()
        pytorch_value = micro(p_dt, p_b, p_x)
        graph_path = output_dir / f"dbx-{mode}.onnx"
        _export_onnx(
            micro,
            (p_dt, p_b, p_x),
            graph_path,
            input_names=["dt_value", "b_value", "x_heads"],
            output_names=["d_b_x"],
        )
        ort_value = _run_ort_cpu(
            graph_path,
            inputs={
                "dt_value": p_dt.detach().numpy(),
                "b_value": p_b.detach().numpy(),
                "x_heads": p_x.detach().numpy(),
            },
            outputs=["d_b_x"],
        )[0]
        micro_metrics[mode] = {
            "pytorch_vs_source_einsum": _metrics(
                reference,
                pytorch_value,
                epsilon=epsilon,
            ),
            "ort_vs_source_einsum": _metrics(
                reference,
                ort_value,
                epsilon=epsilon,
            ),
            "ort_vs_same_mode_pytorch": _metrics(
                pytorch_value,
                ort_value,
                epsilon=epsilon,
            ),
        }
        del micro

    operator_einsum_units = _max_units(
        micro_metrics["einsum"]["ort_vs_same_mode_pytorch"]
    )
    propagated_units = _max_units(
        sensitivity["all_ort_operands_pytorch_einsum"]
    )

    if operator_einsum_units > diagnostic_reference_units:
        diagnosis = "ort_einsum_operator"
    elif propagated_units > diagnostic_reference_units:
        diagnosis = "upstream_operand_drift_amplification"
    else:
        diagnosis = "not_reproduced_in_isolated_dbx"

    body: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "MEASURED",
        "k2d_receipt_id": k2d["receipt_id"],
        "k2e_receipt_id": k2e["receipt_id"],
        "g04_manifest_id": manifest["manifest_id"],
        "capsule_id": manifest["capsule_id"],
        "source_weight_sha256": manifest["source_weight_sha256"],
        "trace_layer": trace_layer,
        "probe_token_id": token_id,
        "provider": "CPUExecutionProvider",
        "dtype": str(manifest["state_contract"]["dtype"]),
        "fp16_epsilon": epsilon,
        "diagnostic_reference_max_epsilon_units":
            diagnostic_reference_units,
        "k2e_no_improvement_confirmed": True,
        "operand_metrics": operand_metrics,
        "operand_sensitivity": sensitivity,
        "micro_operator_metrics": micro_metrics,
        "diagnosis": diagnosis,
        "acceptance_threshold_defined": False,
        "production_activation_authorized": False,
    }
    body["receipt_id"] = hashlib.sha256(
        SCHEMA.encode("ascii") + b"\0" + _canonical_json(body)
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_json(body) + b"\n")

    del prefix
    del layer
    del model
    del capsule
    gc.collect()
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capsule-root", required=True, type=Path)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--k2d-receipt", required=True, type=Path)
    parser.add_argument("--k2e-receipt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/vn97-k2f-dbx-operands.json"),
    )
    args = parser.parse_args()

    receipt = run_k2f(
        capsule_root=args.capsule_root.resolve(strict=True),
        bundle_dir=args.bundle_dir.resolve(strict=True),
        k2d_receipt_path=args.k2d_receipt.resolve(strict=True),
        k2e_receipt_path=args.k2e_receipt.resolve(strict=True),
        output_dir=args.output_dir,
        output_path=args.output,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
