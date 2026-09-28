from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .mamba2_g03_capsule import load_g03_capsule
from .mamba2_kaggle_k2a_ort import _canonical_json, _load_json, _prompt_cases
from .mamba2_kaggle_k2b_provider_split import _ort_actual, compare_cases
from .mamba2_onnx import (
    Mamba2OnnxConfig,
    VN97Mamba2StepOnnx,
    verify_mamba2_g04_bundle,
)


SCHEMA = "VN97M2K2CBACKEND1"
K2B_SCHEMA = "VN97M2K2BPROVIDER1"


@torch.inference_mode()
def _pytorch_expected_on_device(
    *,
    capsule_root: Path,
    token_cases: list[list[int]],
    device: torch.device,
) -> list[dict[str, Any]]:
    capsule = load_g03_capsule(
        capsule_root,
        verify_large_weight_sha256=False,
    )
    model = VN97Mamba2StepOnnx(
        Mamba2OnnxConfig.from_source_spec(capsule.spec),
        capsule.tensors,
    ).eval().to(device)

    outputs: list[dict[str, Any]] = []
    for case_index, token_ids in enumerate(token_cases):
        conv, ssm = model.initial_state(1)
        step_logits: list[np.ndarray] = []
        for token_id in token_ids:
            token = torch.tensor(
                [token_id],
                dtype=torch.long,
                device=device,
            )
            logits, conv, ssm = model(token, conv, ssm)
            step_logits.append(logits.detach().cpu().numpy().copy())
        outputs.append(
            {
                "case_index": case_index,
                "token_ids": token_ids,
                "logits": step_logits,
                "final_conv_state": conv.detach().cpu().numpy().copy(),
                "final_ssm_state": ssm.detach().cpu().numpy().copy(),
            }
        )

    del model
    del capsule
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return outputs


def _layerwise_state_metrics(
    reference_case: dict[str, Any],
    candidate_case: dict[str, Any],
    *,
    epsilon: float,
    diagnostic_reference_units: float,
) -> dict[str, Any]:
    ref_conv = np.asarray(reference_case["final_conv_state"])
    cand_conv = np.asarray(candidate_case["final_conv_state"])
    ref_ssm = np.asarray(reference_case["final_ssm_state"])
    cand_ssm = np.asarray(candidate_case["final_ssm_state"])
    if ref_conv.shape != cand_conv.shape or ref_ssm.shape != cand_ssm.shape:
        raise ValueError("K2C layerwise state shape mismatch")
    if ref_conv.ndim < 1 or ref_ssm.ndim < 1:
        raise ValueError("K2C layerwise state rank mismatch")
    if ref_conv.shape[0] != ref_ssm.shape[0]:
        raise ValueError("K2C layer count mismatch")

    layers: list[dict[str, Any]] = []
    first_over: int | None = None
    for index in range(ref_conv.shape[0]):
        conv_left = ref_conv[index].astype(np.float32, copy=False)
        conv_right = cand_conv[index].astype(np.float32, copy=False)
        ssm_left = ref_ssm[index].astype(np.float32, copy=False)
        ssm_right = cand_ssm[index].astype(np.float32, copy=False)

        conv_diff = np.abs(conv_left - conv_right)
        ssm_diff = np.abs(ssm_left - ssm_right)
        conv_scale = np.maximum(1.0, np.abs(conv_left))
        ssm_scale = np.maximum(1.0, np.abs(ssm_left))
        conv_units = conv_diff / (epsilon * conv_scale)
        ssm_units = ssm_diff / (epsilon * ssm_scale)

        record = {
            "layer": index,
            "conv_max_abs_error": float(conv_diff.max(initial=0.0)),
            "conv_max_epsilon_units": float(conv_units.max(initial=0.0)),
            "ssm_max_abs_error": float(ssm_diff.max(initial=0.0)),
            "ssm_max_epsilon_units": float(ssm_units.max(initial=0.0)),
        }
        record["max_epsilon_units"] = max(
            record["conv_max_epsilon_units"],
            record["ssm_max_epsilon_units"],
        )
        if (
            first_over is None
            and record["max_epsilon_units"] > diagnostic_reference_units
        ):
            first_over = index
        layers.append(record)

    return {
        "first_layer_over_reference": first_over,
        "max_layer_epsilon_units": max(
            (float(item["max_epsilon_units"]) for item in layers),
            default=0.0,
        ),
        "layers": layers,
    }


def _max_units(comparison: dict[str, Any]) -> float:
    metrics = comparison["metrics"]
    return max(
        float(metrics["logits"]["max_epsilon_units"]),
        float(metrics["conv_state"]["max_epsilon_units"]),
        float(metrics["ssm_state"]["max_epsilon_units"]),
    )


def _verify_k2b_receipt(path: Path) -> dict[str, Any]:
    receipt = _load_json(path, "K2B receipt")
    if receipt.get("schema") != K2B_SCHEMA:
        raise ValueError("K2C requires K2B provider-split receipt")
    if receipt.get("status") != "MEASURED":
        raise ValueError("K2C requires K2B status MEASURED")
    if receipt.get("acceptance_threshold_defined") is not False:
        raise ValueError("K2B must remain measurement-only")
    if receipt.get("production_activation_authorized") is not False:
        raise ValueError("K2B must not authorize production")
    receipt_id = receipt.get("receipt_id")
    if not isinstance(receipt_id, str) or len(receipt_id) != 64:
        raise ValueError("K2C K2B receipt identity invalid")
    body = dict(receipt)
    body.pop("receipt_id", None)
    expected = hashlib.sha256(
        K2B_SCHEMA.encode("ascii") + b"\0" + _canonical_json(body)
    ).hexdigest()
    if receipt_id != expected:
        raise ValueError("K2C K2B receipt identity mismatch")
    return receipt


def run_k2c(
    *,
    capsule_root: Path,
    bundle_dir: Path,
    k2b_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("K2C requires CUDA")
    gpu_name = torch.cuda.get_device_name(0)
    if "T4" not in gpu_name.upper():
        raise RuntimeError(f"K2C expects Kaggle T4, got {gpu_name!r}")

    k2b = _verify_k2b_receipt(k2b_receipt_path)
    manifest = verify_mamba2_g04_bundle(bundle_dir)
    for field in ("g04_manifest_id", "capsule_id", "source_weight_sha256"):
        manifest_field = (
            "manifest_id" if field == "g04_manifest_id" else field
        )
        if manifest.get(manifest_field) != k2b.get(field):
            raise ValueError(f"K2C {field} lineage mismatch")

    capsule_manifest = _load_json(
        capsule_root / "capsule.vn97m2g03.json",
        "G0.3 capsule manifest",
    )
    if capsule_manifest.get("capsule_id") != k2b.get("capsule_id"):
        raise ValueError("K2C loaded capsule identity mismatch")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(capsule_root / "tokenizer"),
        local_files_only=True,
        use_fast=True,
    )
    prompt_cases = _prompt_cases(tokenizer)
    if not prompt_cases or not prompt_cases[0]:
        raise RuntimeError("K2C diagnostic probe is empty")
    token_cases = [[int(prompt_cases[0][0])]]

    # One token is intentional. K2C is a backend-isolation diagnostic, not an
    # acceptance campaign. Keeping this to one recurrent step makes the full
    # 2.7B CPU reference practical on Kaggle.
    pytorch_cpu = _pytorch_expected_on_device(
        capsule_root=capsule_root,
        token_cases=token_cases,
        device=torch.device("cpu"),
    )
    pytorch_cuda = _pytorch_expected_on_device(
        capsule_root=capsule_root,
        token_cases=token_cases,
        device=torch.device("cuda:0"),
    )
    ort_cpu, cpu_provider = _ort_actual(
        graph_path=bundle_dir / "step.onnx",
        expected_cases=pytorch_cpu,
        provider_name="CPUExecutionProvider",
    )
    ort_cuda, cuda_provider = _ort_actual(
        graph_path=bundle_dir / "step.onnx",
        expected_cases=pytorch_cpu,
        provider_name="CUDAExecutionProvider",
    )

    epsilon = float(np.finfo(np.float16).eps)
    diagnostic_reference_units = 2.0

    pytorch_cuda_vs_cpu = compare_cases(
        pytorch_cpu,
        pytorch_cuda,
        epsilon=epsilon,
    )
    pytorch_cpu_vs_ort_cpu = compare_cases(
        pytorch_cpu,
        ort_cpu,
        epsilon=epsilon,
    )
    pytorch_cuda_vs_ort_cuda = compare_cases(
        pytorch_cuda,
        ort_cuda,
        epsilon=epsilon,
    )
    ort_cpu_vs_cuda = compare_cases(
        ort_cpu,
        ort_cuda,
        epsilon=epsilon,
    )

    layerwise = {
        "pytorch_cuda_vs_cpu": _layerwise_state_metrics(
            pytorch_cpu[0],
            pytorch_cuda[0],
            epsilon=epsilon,
            diagnostic_reference_units=diagnostic_reference_units,
        ),
        "pytorch_cpu_vs_ort_cpu": _layerwise_state_metrics(
            pytorch_cpu[0],
            ort_cpu[0],
            epsilon=epsilon,
            diagnostic_reference_units=diagnostic_reference_units,
        ),
        "ort_cpu_vs_cuda": _layerwise_state_metrics(
            ort_cpu[0],
            ort_cuda[0],
            epsilon=epsilon,
            diagnostic_reference_units=diagnostic_reference_units,
        ),
    }

    body: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "MEASURED",
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cpu_provider": cpu_provider,
        "cuda_provider": cuda_provider,
        "k2b_receipt_id": k2b["receipt_id"],
        "g04_manifest_id": manifest["manifest_id"],
        "capsule_id": manifest["capsule_id"],
        "source_weight_sha256": manifest["source_weight_sha256"],
        "dtype": str(manifest["state_contract"]["dtype"]),
        "fp16_epsilon": epsilon,
        "probe_cases": 1,
        "steps_compared": 1,
        "probe_token_id": token_cases[0][0],
        "diagnostic_reference_max_epsilon_units":
            diagnostic_reference_units,
        "pytorch_cuda_vs_cpu": pytorch_cuda_vs_cpu,
        "pytorch_cpu_vs_ort_cpu": pytorch_cpu_vs_ort_cpu,
        "pytorch_cuda_vs_ort_cuda": pytorch_cuda_vs_ort_cuda,
        "ort_cpu_vs_cuda": ort_cpu_vs_cuda,
        "layerwise_state": layerwise,
        "pytorch_backend_within_reference": (
            _max_units(pytorch_cuda_vs_cpu)
            <= diagnostic_reference_units
        ),
        "onnx_cpu_within_reference": (
            _max_units(pytorch_cpu_vs_ort_cpu)
            <= diagnostic_reference_units
        ),
        "ort_provider_within_reference": (
            _max_units(ort_cpu_vs_cuda)
            <= diagnostic_reference_units
        ),
        "acceptance_threshold_defined": False,
        "production_activation_authorized": False,
    }
    body["receipt_id"] = hashlib.sha256(
        SCHEMA.encode("ascii") + b"\0" + _canonical_json(body)
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_json(body) + b"\n")
    return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capsule-root", required=True, type=Path)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--k2b-receipt", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/vn97-k2c-backend-matrix.json"),
    )
    args = parser.parse_args()

    receipt = run_k2c(
        capsule_root=args.capsule_root.resolve(strict=True),
        bundle_dir=args.bundle_dir.resolve(strict=True),
        k2b_receipt_path=args.k2b_receipt.resolve(strict=True),
        output_path=args.output,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
