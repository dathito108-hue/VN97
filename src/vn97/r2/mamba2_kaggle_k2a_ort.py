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
from .mamba2_onnx import (
    Mamba2OnnxConfig,
    VN97Mamba2StepOnnx,
    verify_mamba2_g04_bundle,
)


SCHEMA = "VN97M2K2AORT1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


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


def _prompt_cases(tokenizer: Any) -> list[list[int]]:
    prompts = (
        "The quick brown fox",
        "State space models",
        "Mobile inference",
    )
    cases: list[list[int]] = []
    for prompt in prompts:
        ids = tokenizer(
            prompt,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        case = [int(value) for value in ids[:4]]
        if case:
            cases.append(case)
    if len(cases) != len(prompts):
        raise RuntimeError("K2A prompt tokenization unexpectedly empty")
    return cases


@torch.inference_mode()
def _pytorch_expected(
    *,
    capsule_root: Path,
    token_cases: list[list[int]],
) -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("K2A PyTorch reference requires CUDA")
    device = torch.device("cuda:0")
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
            step_logits.append(
                logits.detach().cpu().numpy().copy()
            )
        outputs.append(
            {
                "case_index": case_index,
                "token_ids": token_ids,
                "logits": step_logits,
                "final_conv_state":
                    conv.detach().cpu().numpy().copy(),
                "final_ssm_state":
                    ssm.detach().cpu().numpy().copy(),
            }
        )

    del model
    del capsule
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


def _ort_actual(
    *,
    graph_path: Path,
    expected_cases: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    import onnxruntime as ort

    # K2A runs in the same process after torch has initialized the Kaggle
    # CUDA runtime. Explicitly preload the CUDA/cuDNN libraries before ORT
    # creates its provider so it resolves the same runtime family.
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError(
            "K2A requires ONNX Runtime CUDAExecutionProvider; available="
            + repr(providers)
        )

    session = ort.InferenceSession(
        str(graph_path),
        providers=[
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kSameAsRequested",
                },
            )
        ],
    )
    active = session.get_providers()
    if not active or active[0] != "CUDAExecutionProvider":
        raise RuntimeError(
            "K2A ORT session did not activate CUDAExecutionProvider: "
            + repr(active)
        )

    input_names = {item.name for item in session.get_inputs()}
    output_names = {item.name for item in session.get_outputs()}
    if input_names != {"input_ids", "conv_state", "ssm_state"}:
        raise RuntimeError("K2A ONNX input contract mismatch")
    if output_names != {"logits", "next_conv_state", "next_ssm_state"}:
        raise RuntimeError("K2A ONNX output contract mismatch")

    results: list[dict[str, Any]] = []
    for expected in expected_cases:
        first_conv = expected["final_conv_state"]
        first_ssm = expected["final_ssm_state"]
        conv = np.zeros_like(first_conv)
        ssm = np.zeros_like(first_ssm)
        step_logits: list[np.ndarray] = []
        for token_id in expected["token_ids"]:
            values = session.run(
                ["logits", "next_conv_state", "next_ssm_state"],
                {
                    "input_ids": np.asarray([token_id], dtype=np.int64),
                    "conv_state": conv,
                    "ssm_state": ssm,
                },
            )
            logits, conv, ssm = values
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
    return results, active[0]


def run_k2a(
    *,
    capsule_root: Path,
    bundle_dir: Path,
    k1i_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("K2A requires CUDA")
    gpu_name = torch.cuda.get_device_name(0)
    if "T4" not in gpu_name.upper():
        raise RuntimeError(f"K2A expects Kaggle T4, got {gpu_name!r}")

    k1i = _load_json(k1i_receipt_path, "K1I receipt")
    if k1i.get("schema") != "VN97M2K1ISEMANTIC1":
        raise ValueError("K2A requires K1I semantic parity receipt")
    if k1i.get("status") != "PASS":
        raise ValueError("K2A requires K1I PASS")
    if k1i.get("same_input_all_layers_passed") is not True:
        raise ValueError("K2A requires all-layer K1I parity PASS")
    k1i_receipt_id = k1i.get("receipt_id")
    if (
        not isinstance(k1i_receipt_id, str)
        or len(k1i_receipt_id) != 64
    ):
        raise ValueError("K2A K1I receipt identity invalid")

    manifest = verify_mamba2_g04_bundle(bundle_dir)
    if manifest.get("capsule_id") != k1i.get("capsule_id"):
        raise ValueError("K2A ONNX bundle/K1I capsule mismatch")
    if manifest.get("source_weight_sha256") != k1i.get(
        "source_weight_sha256"
    ):
        raise ValueError("K2A ONNX bundle/K1I source-weight mismatch")
    if manifest.get("same_weights_semantics") is not True:
        raise ValueError("K2A requires same-weight ONNX graph")
    if manifest.get("quantization_used") is not False:
        raise ValueError("K2A dense baseline must not be quantized")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(capsule_root / "tokenizer"),
        local_files_only=True,
        use_fast=True,
    )
    token_cases = _prompt_cases(tokenizer)

    expected = _pytorch_expected(
        capsule_root=capsule_root,
        token_cases=token_cases,
    )
    free_after_reference, total = torch.cuda.mem_get_info()
    if free_after_reference < 10 * 1024**3:
        raise RuntimeError(
            "K2A did not release enough T4 VRAM before ORT session creation"
        )

    actual, provider = _ort_actual(
        graph_path=bundle_dir / "step.onnx",
        expected_cases=expected,
    )

    epsilon = float(np.finfo(np.float16).eps)
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
    logit_mean_values: list[float] = []
    argmax_exact = True
    step_count = 0

    for ref_case, ort_case in zip(expected, actual, strict=True):
        if ref_case["token_ids"] != ort_case["token_ids"]:
            raise RuntimeError("K2A case token mismatch")
        for ref_logits, ort_logits in zip(
            ref_case["logits"],
            ort_case["logits"],
            strict=True,
        ):
            metrics = _epsilon_metrics(
                ref_logits,
                ort_logits,
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
            logit_mean_values.append(metrics["mean_abs_error"])
            if int(ref_logits.argmax()) != int(ort_logits.argmax()):
                argmax_exact = False
            step_count += 1

        for name, ref_value, ort_value in (
            (
                "conv_state",
                ref_case["final_conv_state"],
                ort_case["final_conv_state"],
            ),
            (
                "ssm_state",
                ref_case["final_ssm_state"],
                ort_case["final_ssm_state"],
            ),
        ):
            metrics = _epsilon_metrics(
                ref_value,
                ort_value,
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
        float(sum(logit_mean_values) / len(logit_mean_values))
        if logit_mean_values
        else 0.0
    )

    body: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "MEASURED",
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "ort_provider": provider,
        "k1i_receipt_id": k1i_receipt_id,
        "g04_manifest_id": manifest["manifest_id"],
        "capsule_id": manifest["capsule_id"],
        "source_weight_sha256": manifest["source_weight_sha256"],
        "dtype": str(manifest["state_contract"]["dtype"]),
        "fp16_epsilon": epsilon,
        "probe_cases": len(token_cases),
        "steps_compared": step_count,
        "argmax_exact": argmax_exact,
        "metrics": aggregate,
        "gpu_total_bytes": total,
        "gpu_free_bytes_before_ort": free_after_reference,
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
    parser.add_argument("--k1i-receipt", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/vn97-k2a-ort-parity.json"),
    )
    args = parser.parse_args()

    receipt = run_k2a(
        capsule_root=args.capsule_root.resolve(strict=True),
        bundle_dir=args.bundle_dir.resolve(strict=True),
        k1i_receipt_path=args.k1i_receipt.resolve(strict=True),
        output_path=args.output,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
