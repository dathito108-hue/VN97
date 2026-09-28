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
)
from .mamba2_kaggle_k2f_dbx_operands import _verify_receipt
from .mamba2_kaggle_k2j_closed_loop import (
    TOP_K,
    _topk_overlap,
)
from .mamba2_kaggle_k2m_tie_sensitivity import SCHEMA as K2M_SCHEMA
from .mamba2_onnx import (
    Mamba2OnnxConfig,
    VN97Mamba2StepOnnx,
    verify_mamba2_g04_bundle,
)


SCHEMA = "VN97M2K2NULP1"
GENERATED_TOKENS = 64
PROMPTS = (
    "State space models can reason about",
    "A mobile assistant should carefully",
    "The fastest reliable inference path is",
    "Explain why recurrent state matters for",
    "Write a short Python function that",
    "Given a sequence of observations, infer",
    "Một trợ lý di động tự chủ cần",
    "Trong mô hình trạng thái tuần tự,",
)


def _fp16_downward_ulp(value: float) -> float:
    current = np.float16(value)
    lower = np.nextafter(
        current,
        np.float16(-np.inf),
        dtype=np.float16,
    )
    spacing = abs(float(current) - float(lower))
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("K2N cannot resolve FP16 ULP for candidate logit")
    return spacing


def _candidate_gap_ulp(
    logits: np.ndarray,
    *,
    reference_top1: int,
    candidate_token: int,
) -> dict[str, float | bool]:
    values = logits.astype(np.float32, copy=False).reshape(-1)
    reference_value = float(values[reference_top1])
    candidate_value = float(values[candidate_token])
    gap = float(reference_value - candidate_value)
    if gap < -1.0e-7:
        raise ValueError("K2N reference top1 is not maximal")
    gap = max(gap, 0.0)
    ulp = _fp16_downward_ulp(reference_value)
    units = float(gap / ulp)
    return {
        "reference_top1_logit": reference_value,
        "candidate_logit": candidate_value,
        "reference_candidate_gap": gap,
        "reference_top1_downward_ulp": ulp,
        "reference_candidate_gap_ulp": units,
        "within_one_reference_ulp": units <= 1.000001,
    }


def _blank_metrics() -> dict[str, float]:
    return {
        "max_abs_error": 0.0,
        "mean_abs_error": 0.0,
        "max_epsilon_units": 0.0,
    }


def _aggregate_max(
    aggregate: dict[str, float],
    metrics: dict[str, float],
) -> None:
    aggregate["max_abs_error"] = max(
        aggregate["max_abs_error"],
        float(metrics["max_abs_error"]),
    )
    aggregate["mean_abs_error"] = max(
        aggregate["mean_abs_error"],
        float(metrics["mean_abs_error"]),
    )
    aggregate["max_epsilon_units"] = max(
        aggregate["max_epsilon_units"],
        float(metrics["max_epsilon_units"]),
    )


def _tokenize_prompts(tokenizer: Any) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, prompt in enumerate(PROMPTS):
        ids = tokenizer(
            prompt,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        token_ids = [int(value) for value in ids]
        if not token_ids:
            raise RuntimeError(f"K2N prompt {index} tokenized empty")
        cases.append(
            {
                "prompt_index": index,
                "prompt": prompt,
                "token_ids": token_ids,
            }
        )
    return cases


@torch.inference_mode()
def _pytorch_reference(
    *,
    capsule_root: Path,
    prompt_cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("K2N PyTorch reference requires CUDA")

    device = torch.device("cuda:0")
    capsule = load_g03_capsule(
        capsule_root,
        verify_large_weight_sha256=False,
    )
    model = VN97Mamba2StepOnnx(
        Mamba2OnnxConfig.from_source_spec(capsule.spec),
        capsule.tensors,
    ).eval().to(device)

    references: list[dict[str, Any]] = []
    for case in prompt_cases:
        conv, ssm = model.initial_state(1)
        logits: torch.Tensor | None = None

        for token_id in case["token_ids"]:
            token = torch.tensor(
                [token_id],
                dtype=torch.long,
                device=device,
            )
            logits, conv, ssm = model(token, conv, ssm)

        if logits is None:
            raise RuntimeError("K2N prompt priming produced no logits")

        generated: list[int] = []
        prediction_logits: list[np.ndarray] = []

        for _ in range(GENERATED_TOKENS):
            current = logits.detach().cpu().numpy().copy()
            next_token = int(current.argmax())
            prediction_logits.append(current)
            generated.append(next_token)

            token = torch.tensor(
                [next_token],
                dtype=torch.long,
                device=device,
            )
            logits, conv, ssm = model(token, conv, ssm)

        references.append(
            {
                "prompt_index": int(case["prompt_index"]),
                "prompt_token_ids": list(case["token_ids"]),
                "generated_token_ids": generated,
                "prediction_logits": prediction_logits,
            }
        )

    del model
    del capsule
    gc.collect()
    torch.cuda.empty_cache()
    return references


def _state_shapes(manifest: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    contract = manifest["state_contract"]
    conv_shape = tuple(
        int(value) for value in contract["conv_state"]["shape"]
    )
    ssm_shape = tuple(
        int(value) for value in contract["ssm_state"]["shape"]
    )
    if not conv_shape or not ssm_shape:
        raise ValueError("K2N invalid state shapes")
    return conv_shape, ssm_shape


def _diagnose(
    *,
    mismatch_count: int,
    mismatches_outside_one_ulp: int,
    top5_overlap_min: int,
) -> str:
    if mismatch_count == 0:
        return "exact_same_input_decision_parity"
    if mismatches_outside_one_ulp == 0 and top5_overlap_min >= 4:
        return "only_fp16_near_tie_decision_divergences"
    if mismatches_outside_one_ulp == 0:
        return "near_tie_divergences_with_rank_neighborhood_change"
    return "material_same_input_decision_divergence_present"


def run_k2n(
    *,
    capsule_root: Path,
    bundle_dir: Path,
    k2m_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("K2N requires CUDA")
    gpu_name = torch.cuda.get_device_name(0)
    if "T4" not in gpu_name.upper():
        raise RuntimeError(f"K2N expects Kaggle T4, got {gpu_name!r}")

    k2m = _verify_receipt(
        k2m_receipt_path,
        schema=K2M_SCHEMA,
        label="K2M receipt",
    )
    if k2m.get("diagnosis") != "distributed_near_tie_sensitivity":
        raise ValueError(
            "K2N requires K2M distributed_near_tie_sensitivity evidence"
        )
    if k2m.get("production_graph_changed") is not False:
        raise ValueError("K2N requires unchanged G0.4 production graph")

    manifest = verify_mamba2_g04_bundle(bundle_dir)
    for field in ("g04_manifest_id", "capsule_id", "source_weight_sha256"):
        manifest_field = "manifest_id" if field == "g04_manifest_id" else field
        if manifest.get(manifest_field) != k2m.get(field):
            raise ValueError(f"K2N {field} lineage mismatch")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(capsule_root / "tokenizer"),
        local_files_only=True,
        use_fast=True,
    )
    prompt_cases = _tokenize_prompts(tokenizer)

    references = _pytorch_reference(
        capsule_root=capsule_root,
        prompt_cases=prompt_cases,
    )

    free_after_reference, total = torch.cuda.mem_get_info()
    if free_after_reference < 10 * 1024**3:
        raise RuntimeError(
            "K2N did not release enough T4 VRAM before ORT session creation"
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
            "K2N ORT session did not activate CUDAExecutionProvider"
        )

    conv_shape, ssm_shape = _state_shapes(manifest)
    epsilon = float(np.finfo(np.float16).eps)
    aggregate = _blank_metrics()
    mismatch_records: list[dict[str, Any]] = []
    prompt_summaries: list[dict[str, Any]] = []
    total_decisions = 0
    exact_decisions = 0
    mismatch_count = 0
    mismatches_outside_one_ulp = 0
    top5_overlap_min = TOP_K
    top5_overlap_sum = 0

    for reference in references:
        conv = np.zeros(conv_shape, dtype=np.float16)
        ssm = np.zeros(ssm_shape, dtype=np.float16)
        logits: np.ndarray | None = None

        for token_id in reference["prompt_token_ids"]:
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

        if logits is None:
            raise RuntimeError("K2N ORT prompt priming produced no logits")

        prompt_mismatches = 0
        prompt_outside_one_ulp = 0
        prompt_top5_min = TOP_K
        prompt_top5_sum = 0
        prompt_metrics = _blank_metrics()

        for step in range(1, GENERATED_TOKENS + 1):
            ref_logits = reference["prediction_logits"][step - 1]
            ref_top1 = int(reference["generated_token_ids"][step - 1])
            ort_top1 = int(logits.argmax())

            metrics = _epsilon_metrics(
                ref_logits,
                logits,
                epsilon=epsilon,
            )
            _aggregate_max(aggregate, metrics)
            _aggregate_max(prompt_metrics, metrics)

            overlap = _topk_overlap(
                ref_logits,
                logits,
                TOP_K,
            )
            top5_overlap_min = min(top5_overlap_min, overlap)
            top5_overlap_sum += overlap
            prompt_top5_min = min(prompt_top5_min, overlap)
            prompt_top5_sum += overlap

            total_decisions += 1
            exact = ref_top1 == ort_top1
            exact_decisions += int(exact)

            if not exact:
                mismatch_count += 1
                prompt_mismatches += 1
                gap = _candidate_gap_ulp(
                    ref_logits,
                    reference_top1=ref_top1,
                    candidate_token=ort_top1,
                )
                outside = not bool(gap["within_one_reference_ulp"])
                mismatches_outside_one_ulp += int(outside)
                prompt_outside_one_ulp += int(outside)

                ort_values = logits.astype(np.float32, copy=False).reshape(-1)
                mismatch_records.append(
                    {
                        "prompt_index": int(reference["prompt_index"]),
                        "step": step,
                        "reference_top1": ref_top1,
                        "ort_top1": ort_top1,
                        **gap,
                        "ort_reference_logit": float(
                            ort_values[ref_top1]
                        ),
                        "ort_candidate_logit": float(
                            ort_values[ort_top1]
                        ),
                        "ort_reference_minus_candidate": float(
                            ort_values[ref_top1] - ort_values[ort_top1]
                        ),
                        "top5_overlap": overlap,
                        "logit_metrics": metrics,
                    }
                )

            # Teacher-force the PyTorch reference token so every K2N
            # comparison remains same-input even after a decision mismatch.
            logits, conv, ssm = session.run(
                ["logits", "next_conv_state", "next_ssm_state"],
                {
                    "input_ids": np.asarray([ref_top1], dtype=np.int64),
                    "conv_state": conv,
                    "ssm_state": ssm,
                },
            )
            logits = np.asarray(logits)
            conv = np.asarray(conv)
            ssm = np.asarray(ssm)

        prompt_summaries.append(
            {
                "prompt_index": int(reference["prompt_index"]),
                "prompt_token_ids": reference["prompt_token_ids"],
                "reference_generated_token_ids":
                    reference["generated_token_ids"],
                "decision_count": GENERATED_TOKENS,
                "mismatch_count": prompt_mismatches,
                "mismatches_outside_one_reference_ulp":
                    prompt_outside_one_ulp,
                "top5_overlap_min": prompt_top5_min,
                "top5_overlap_mean": float(
                    prompt_top5_sum / GENERATED_TOKENS
                ),
                "aggregate_logit_metrics": prompt_metrics,
            }
        )

    del session
    gc.collect()

    diagnosis = _diagnose(
        mismatch_count=mismatch_count,
        mismatches_outside_one_ulp=mismatches_outside_one_ulp,
        top5_overlap_min=top5_overlap_min,
    )

    body: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "MEASURED",
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "ort_provider": active[0],
        "k2m_receipt_id": k2m["receipt_id"],
        "g04_manifest_id": manifest["manifest_id"],
        "capsule_id": manifest["capsule_id"],
        "source_weight_sha256": manifest["source_weight_sha256"],
        "dtype": str(manifest["state_contract"]["dtype"]),
        "fp16_epsilon": epsilon,
        "prompt_count": len(references),
        "generated_tokens_per_prompt": GENERATED_TOKENS,
        "total_same_input_decisions": total_decisions,
        "exact_decisions": exact_decisions,
        "exact_decision_fraction": float(
            exact_decisions / total_decisions
        ),
        "mismatch_count": mismatch_count,
        "mismatches_outside_one_reference_ulp":
            mismatches_outside_one_ulp,
        "top5_overlap_min": top5_overlap_min,
        "top5_overlap_mean": float(
            top5_overlap_sum / total_decisions
        ),
        "aggregate_logit_metrics": aggregate,
        "prompt_summaries": prompt_summaries,
        "mismatches": mismatch_records,
        "diagnosis": diagnosis,
        "one_ulp_is_diagnostic_not_acceptance": True,
        "weights_changed": False,
        "architecture_changed": False,
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
    parser.add_argument("--k2m-receipt", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/vn97-k2n-ulp-audit.json"),
    )
    args = parser.parse_args()

    receipt = run_k2n(
        capsule_root=args.capsule_root.resolve(strict=True),
        bundle_dir=args.bundle_dir.resolve(strict=True),
        k2m_receipt_path=args.k2m_receipt.resolve(strict=True),
        output_path=args.output,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
