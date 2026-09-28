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
from .mamba2_kaggle_k2a_ort import (
    _canonical_json,
    _epsilon_metrics,
    _load_json,
)
from .mamba2_kaggle_k2f_dbx_operands import _verify_receipt
from .mamba2_kaggle_k2h_projection_conv import SCHEMA as K2H_SCHEMA
from .mamba2_onnx import (
    Mamba2OnnxConfig,
    VN97Mamba2StepOnnx,
    verify_mamba2_g04_bundle,
)


SCHEMA = "VN97M2K2ITRAJ1"
CHECKPOINT_STEPS = (1, 2, 4, 8, 16, 32)
TOKEN_COUNT = 32
PROMPT = (
    "VN97 validates recurrent numerical stability across execution backends "
    "before production activation, preserving state semantics, deterministic "
    "reasoning, and efficient mobile inference over long sequential context."
)


def _token_sequence(tokenizer: Any) -> list[int]:
    ids = tokenizer(
        PROMPT,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    values = [int(value) for value in ids]
    if len(values) < TOKEN_COUNT:
        raise RuntimeError(
            f"K2I prompt produced only {len(values)} tokens; "
            f"need {TOKEN_COUNT}"
        )
    return values[:TOKEN_COUNT]


def _top1_margin(logits: np.ndarray) -> float:
    values = logits.astype(np.float32, copy=False).reshape(-1)
    if values.size < 2:
        raise ValueError("K2I logits must contain at least two entries")
    indices = np.argpartition(values, -2)[-2:]
    top = np.sort(values[indices])
    return float(top[-1] - top[-2])


@torch.inference_mode()
def _pytorch_reference(
    *,
    capsule_root: Path,
    token_ids: list[int],
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("K2I PyTorch reference requires CUDA")

    device = torch.device("cuda:0")
    capsule = load_g03_capsule(
        capsule_root,
        verify_large_weight_sha256=False,
    )
    model = VN97Mamba2StepOnnx(
        Mamba2OnnxConfig.from_source_spec(capsule.spec),
        capsule.tensors,
    ).eval().to(device)

    conv, ssm = model.initial_state(1)
    logits_steps: list[np.ndarray] = []
    checkpoints: dict[int, dict[str, np.ndarray]] = {}

    for step, token_id in enumerate(token_ids, start=1):
        token = torch.tensor(
            [token_id],
            dtype=torch.long,
            device=device,
        )
        logits, conv, ssm = model(token, conv, ssm)
        logits_steps.append(logits.detach().cpu().numpy().copy())
        if step in CHECKPOINT_STEPS:
            checkpoints[step] = {
                "conv_state": conv.detach().cpu().numpy().copy(),
                "ssm_state": ssm.detach().cpu().numpy().copy(),
            }

    del model
    del capsule
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "token_ids": token_ids,
        "logits": logits_steps,
        "checkpoints": checkpoints,
    }


def _aggregate_max(
    current: dict[str, float],
    metrics: dict[str, float],
) -> None:
    current["max_abs_error"] = max(
        current["max_abs_error"],
        float(metrics["max_abs_error"]),
    )
    current["max_epsilon_units"] = max(
        current["max_epsilon_units"],
        float(metrics["max_epsilon_units"]),
    )
    current["mean_abs_error"] = max(
        current["mean_abs_error"],
        float(metrics["mean_abs_error"]),
    )


def _blank_metrics() -> dict[str, float]:
    return {
        "max_abs_error": 0.0,
        "mean_abs_error": 0.0,
        "max_epsilon_units": 0.0,
    }


def run_k2i(
    *,
    capsule_root: Path,
    bundle_dir: Path,
    k2h_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("K2I requires CUDA")
    gpu_name = torch.cuda.get_device_name(0)
    if "T4" not in gpu_name.upper():
        raise RuntimeError(f"K2I expects Kaggle T4, got {gpu_name!r}")

    k2h = _verify_receipt(
        k2h_receipt_path,
        schema=K2H_SCHEMA,
        label="K2H receipt",
    )
    if k2h.get("diagnosis") != "in_proj_candidate_only":
        raise ValueError("K2I requires K2H in_proj_candidate_only evidence")
    if k2h.get("in_proj_candidate") != "fp16_linear":
        raise ValueError("K2I requires canonical FP16 linear as in-proj candidate")
    if k2h.get("conv_affine_candidate") is not None:
        raise ValueError("K2I expects no bounded conv-affine micro candidate")
    if k2h.get("production_graph_changed") is not False:
        raise ValueError("K2I requires unchanged production graph")

    manifest = verify_mamba2_g04_bundle(bundle_dir)
    for field in ("g04_manifest_id", "capsule_id", "source_weight_sha256"):
        manifest_field = "manifest_id" if field == "g04_manifest_id" else field
        if manifest.get(manifest_field) != k2h.get(field):
            raise ValueError(f"K2I {field} lineage mismatch")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(capsule_root / "tokenizer"),
        local_files_only=True,
        use_fast=True,
    )
    token_ids = _token_sequence(tokenizer)

    reference = _pytorch_reference(
        capsule_root=capsule_root,
        token_ids=token_ids,
    )
    free_after_reference, total = torch.cuda.mem_get_info()
    if free_after_reference < 10 * 1024**3:
        raise RuntimeError(
            "K2I did not release enough T4 VRAM before ORT session creation"
        )

    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    session = ort.InferenceSession(
        str(bundle_dir / "step.onnx"),
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
            "K2I ORT session did not activate CUDAExecutionProvider"
        )

    first = reference["checkpoints"][1]
    conv = np.zeros_like(first["conv_state"])
    ssm = np.zeros_like(first["ssm_state"])
    epsilon = float(np.finfo(np.float16).eps)

    aggregate = {
        "logits": _blank_metrics(),
        "conv_state": _blank_metrics(),
        "ssm_state": _blank_metrics(),
    }
    steps: list[dict[str, Any]] = []
    checkpoint_metrics: dict[str, dict[str, Any]] = {}
    argmax_exact_all = True
    argmax_certified_all = True

    for step, token_id in enumerate(token_ids, start=1):
        logits, conv, ssm = session.run(
            ["logits", "next_conv_state", "next_ssm_state"],
            {
                "input_ids": np.asarray([token_id], dtype=np.int64),
                "conv_state": conv,
                "ssm_state": ssm,
            },
        )
        logits = np.asarray(logits)
        conv = np.asarray(conv)
        ssm = np.asarray(ssm)

        ref_logits = reference["logits"][step - 1]
        logit_metrics = _epsilon_metrics(
            ref_logits,
            logits,
            epsilon=epsilon,
        )
        _aggregate_max(aggregate["logits"], logit_metrics)

        ref_top1 = int(ref_logits.argmax())
        ort_top1 = int(logits.argmax())
        exact = ref_top1 == ort_top1
        margin = _top1_margin(ref_logits)
        certified = (
            margin
            > 2.0 * float(logit_metrics["max_abs_error"])
        )
        argmax_exact_all = argmax_exact_all and exact
        argmax_certified_all = argmax_certified_all and certified

        step_record: dict[str, Any] = {
            "step": step,
            "token_id": token_id,
            "logits": logit_metrics,
            "reference_top1": ref_top1,
            "ort_top1": ort_top1,
            "argmax_exact": exact,
            "top1_margin": margin,
            "argmax_certified_by_margin": certified,
        }

        if step in CHECKPOINT_STEPS:
            ref_state = reference["checkpoints"][step]
            conv_metrics = _epsilon_metrics(
                ref_state["conv_state"],
                conv,
                epsilon=epsilon,
            )
            ssm_metrics = _epsilon_metrics(
                ref_state["ssm_state"],
                ssm,
                epsilon=epsilon,
            )
            _aggregate_max(aggregate["conv_state"], conv_metrics)
            _aggregate_max(aggregate["ssm_state"], ssm_metrics)
            checkpoint_metrics[str(step)] = {
                "conv_state": conv_metrics,
                "ssm_state": ssm_metrics,
            }
            step_record["checkpoint_state"] = {
                "conv_state": conv_metrics,
                "ssm_state": ssm_metrics,
            }

        steps.append(step_record)

    del session
    gc.collect()

    state_growth = {
        "conv_state_max_epsilon_units": [
            {
                "step": step,
                "value": float(
                    checkpoint_metrics[str(step)]["conv_state"][
                        "max_epsilon_units"
                    ]
                ),
            }
            for step in CHECKPOINT_STEPS
        ],
        "ssm_state_max_epsilon_units": [
            {
                "step": step,
                "value": float(
                    checkpoint_metrics[str(step)]["ssm_state"][
                        "max_epsilon_units"
                    ]
                ),
            }
            for step in CHECKPOINT_STEPS
        ],
    }

    body: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "MEASURED",
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "ort_provider": active[0],
        "k2h_receipt_id": k2h["receipt_id"],
        "g04_manifest_id": manifest["manifest_id"],
        "capsule_id": manifest["capsule_id"],
        "source_weight_sha256": manifest["source_weight_sha256"],
        "dtype": str(manifest["state_contract"]["dtype"]),
        "fp16_epsilon": epsilon,
        "token_count": len(token_ids),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "argmax_exact_all_steps": argmax_exact_all,
        "argmax_certified_by_margin_all_steps": argmax_certified_all,
        "aggregate_metrics": aggregate,
        "state_growth": state_growth,
        "steps": steps,
        "production_graph_changed": False,
        "acceptance_threshold_defined": False,
        "production_activation_authorized": False,
        "gpu_total_bytes": total,
        "gpu_free_bytes_before_ort": free_after_reference,
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
    parser.add_argument("--k2h-receipt", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/vn97-k2i-trajectory.json"),
    )
    args = parser.parse_args()

    receipt = run_k2i(
        capsule_root=args.capsule_root.resolve(strict=True),
        bundle_dir=args.bundle_dir.resolve(strict=True),
        k2h_receipt_path=args.k2h_receipt.resolve(strict=True),
        output_path=args.output,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
