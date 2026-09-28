from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .mamba2_kaggle_k2a_ort import (
    _canonical_json,
    _load_json,
    _prompt_cases,
    _pytorch_expected,
)
from .mamba2_onnx import verify_mamba2_g04_bundle


SCHEMA = "VN97M2K2BPROVIDER1"


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


def _ort_actual(
    *,
    graph_path: Path,
    expected_cases: list[dict[str, Any]],
    provider_name: str,
) -> tuple[list[dict[str, Any]], str]:
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()

    available = ort.get_available_providers()
    if provider_name not in available:
        raise RuntimeError(
            f"K2B requires {provider_name}; available={available!r}"
        )

    provider_options: Any
    if provider_name == "CUDAExecutionProvider":
        provider_options = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kSameAsRequested",
                },
            )
        ]
    else:
        provider_options = ["CPUExecutionProvider"]

    session = ort.InferenceSession(
        str(graph_path),
        providers=provider_options,
    )
    active = session.get_providers()
    if not active or active[0] != provider_name:
        raise RuntimeError(
            f"K2B session did not activate {provider_name}: {active!r}"
        )

    input_names = {item.name for item in session.get_inputs()}
    output_names = {item.name for item in session.get_outputs()}
    if input_names != {"input_ids", "conv_state", "ssm_state"}:
        raise RuntimeError("K2B ONNX input contract mismatch")
    if output_names != {"logits", "next_conv_state", "next_ssm_state"}:
        raise RuntimeError("K2B ONNX output contract mismatch")

    results: list[dict[str, Any]] = []
    for expected in expected_cases:
        conv = np.zeros_like(expected["final_conv_state"])
        ssm = np.zeros_like(expected["final_ssm_state"])
        step_logits: list[np.ndarray] = []
        for token_id in expected["token_ids"]:
            logits, conv, ssm = session.run(
                ["logits", "next_conv_state", "next_ssm_state"],
                {
                    "input_ids": np.asarray([token_id], dtype=np.int64),
                    "conv_state": conv,
                    "ssm_state": ssm,
                },
            )
            step_logits.append(np.asarray(logits).copy())
            conv = np.asarray(conv).copy()
            ssm = np.asarray(ssm).copy()
        results.append(
            {
                "case_index": expected["case_index"],
                "token_ids": expected["token_ids"],
                "logits": step_logits,
                "final_conv_state": conv,
                "final_ssm_state": ssm,
            }
        )

    del session
    gc.collect()
    if provider_name == "CUDAExecutionProvider" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results, active[0]


def compare_cases(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    epsilon: float,
) -> dict[str, Any]:
    aggregate = {
        "logits": {
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "max_epsilon_units": 0.0,
        },
        "conv_state": {
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "max_epsilon_units": 0.0,
        },
        "ssm_state": {
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "max_epsilon_units": 0.0,
        },
    }
    logit_means: list[float] = []
    argmax_exact = True
    steps = 0

    if len(reference) != len(candidate):
        raise ValueError("K2B case count mismatch")

    for left, right in zip(reference, candidate, strict=True):
        if left["case_index"] != right["case_index"]:
            raise ValueError("K2B case identity mismatch")
        if left["token_ids"] != right["token_ids"]:
            raise ValueError("K2B token sequence mismatch")

        for ref_logits, cand_logits in zip(
            left["logits"],
            right["logits"],
            strict=True,
        ):
            metrics = _epsilon_metrics(
                ref_logits,
                cand_logits,
                epsilon=epsilon,
            )
            aggregate["logits"]["max_abs_error"] = max(
                aggregate["logits"]["max_abs_error"],
                metrics["max_abs_error"],
            )
            aggregate["logits"]["max_epsilon_units"] = max(
                aggregate["logits"]["max_epsilon_units"],
                metrics["max_epsilon_units"],
            )
            logit_means.append(metrics["mean_abs_error"])
            if int(ref_logits.argmax()) != int(cand_logits.argmax()):
                argmax_exact = False
            steps += 1

        for name, ref_value, cand_value in (
            (
                "conv_state",
                left["final_conv_state"],
                right["final_conv_state"],
            ),
            (
                "ssm_state",
                left["final_ssm_state"],
                right["final_ssm_state"],
            ),
        ):
            metrics = _epsilon_metrics(
                ref_value,
                cand_value,
                epsilon=epsilon,
            )
            aggregate[name]["max_abs_error"] = max(
                aggregate[name]["max_abs_error"],
                metrics["max_abs_error"],
            )
            aggregate[name]["mean_abs_error"] = max(
                aggregate[name]["mean_abs_error"],
                metrics["mean_abs_error"],
            )
            aggregate[name]["max_epsilon_units"] = max(
                aggregate[name]["max_epsilon_units"],
                metrics["max_epsilon_units"],
            )

    aggregate["logits"]["mean_abs_error"] = (
        float(sum(logit_means) / len(logit_means))
        if logit_means
        else 0.0
    )
    return {
        "argmax_exact": argmax_exact,
        "steps_compared": steps,
        "metrics": aggregate,
    }


def _max_units(comparison: dict[str, Any]) -> float:
    metrics = comparison["metrics"]
    return max(
        float(metrics["logits"]["max_epsilon_units"]),
        float(metrics["conv_state"]["max_epsilon_units"]),
        float(metrics["ssm_state"]["max_epsilon_units"]),
    )


def run_k2b(
    *,
    capsule_root: Path,
    bundle_dir: Path,
    k2a_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("K2B requires CUDA")
    gpu_name = torch.cuda.get_device_name(0)
    if "T4" not in gpu_name.upper():
        raise RuntimeError(f"K2B expects Kaggle T4, got {gpu_name!r}")

    k2a = _load_json(k2a_receipt_path, "K2A receipt")
    if k2a.get("schema") != "VN97M2K2AORT1":
        raise ValueError("K2B requires K2A measurement receipt")
    if k2a.get("status") != "MEASURED":
        raise ValueError("K2B requires K2A status MEASURED")
    if k2a.get("ort_provider") != "CUDAExecutionProvider":
        raise ValueError("K2B requires real K2A CUDA provider evidence")
    if k2a.get("production_activation_authorized") is not False:
        raise ValueError("K2A measurement must not self-authorize production")
    k2a_receipt_id = k2a.get("receipt_id")
    if not isinstance(k2a_receipt_id, str) or len(k2a_receipt_id) != 64:
        raise ValueError("K2B K2A receipt identity invalid")

    manifest = verify_mamba2_g04_bundle(bundle_dir)
    if manifest.get("manifest_id") != k2a.get("g04_manifest_id"):
        raise ValueError("K2B G0.4 manifest/K2A mismatch")
    if manifest.get("capsule_id") != k2a.get("capsule_id"):
        raise ValueError("K2B capsule/K2A mismatch")
    if manifest.get("source_weight_sha256") != k2a.get(
        "source_weight_sha256"
    ):
        raise ValueError("K2B source-weight/K2A mismatch")

    capsule_manifest = _load_json(
        capsule_root / "capsule.vn97m2g03.json",
        "G0.3 capsule manifest",
    )
    if capsule_manifest.get("capsule_id") != k2a.get("capsule_id"):
        raise ValueError("K2B loaded capsule identity mismatch")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(capsule_root / "tokenizer"),
        local_files_only=True,
        use_fast=True,
    )
    token_cases = _prompt_cases(tokenizer)

    reference = _pytorch_expected(
        capsule_root=capsule_root,
        token_cases=token_cases,
    )

    cpu_actual, cpu_provider = _ort_actual(
        graph_path=bundle_dir / "step.onnx",
        expected_cases=reference,
        provider_name="CPUExecutionProvider",
    )
    pytorch_vs_cpu = compare_cases(
        reference,
        cpu_actual,
        epsilon=float(np.finfo(np.float16).eps),
    )

    cuda_actual, cuda_provider = _ort_actual(
        graph_path=bundle_dir / "step.onnx",
        expected_cases=reference,
        provider_name="CUDAExecutionProvider",
    )
    pytorch_vs_cuda = compare_cases(
        reference,
        cuda_actual,
        epsilon=float(np.finfo(np.float16).eps),
    )
    cpu_vs_cuda = compare_cases(
        cpu_actual,
        cuda_actual,
        epsilon=float(np.finfo(np.float16).eps),
    )

    semantic_reference_units = 2.0
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "MEASURED",
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cpu_provider": cpu_provider,
        "cuda_provider": cuda_provider,
        "k2a_receipt_id": k2a_receipt_id,
        "g04_manifest_id": manifest["manifest_id"],
        "capsule_id": manifest["capsule_id"],
        "source_weight_sha256": manifest["source_weight_sha256"],
        "dtype": str(manifest["state_contract"]["dtype"]),
        "fp16_epsilon": float(np.finfo(np.float16).eps),
        "probe_cases": len(token_cases),
        "pytorch_vs_cpu": pytorch_vs_cpu,
        "cpu_vs_cuda": cpu_vs_cuda,
        "pytorch_vs_cuda": pytorch_vs_cuda,
        "diagnostic_reference_max_epsilon_units": semantic_reference_units,
        "pytorch_vs_cpu_within_reference": (
            _max_units(pytorch_vs_cpu) <= semantic_reference_units
        ),
        "cpu_vs_cuda_within_reference": (
            _max_units(cpu_vs_cuda) <= semantic_reference_units
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
    parser.add_argument("--k2a-receipt", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/vn97-k2b-provider-split.json"),
    )
    args = parser.parse_args()

    receipt = run_k2b(
        capsule_root=args.capsule_root.resolve(strict=True),
        bundle_dir=args.bundle_dir.resolve(strict=True),
        k2a_receipt_path=args.k2a_receipt.resolve(strict=True),
        output_path=args.output,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
