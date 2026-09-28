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
    _epsilon_metrics,
)
from .mamba2_kaggle_k2f_dbx_operands import _verify_receipt
from .mamba2_kaggle_k2j_closed_loop import (
    TOP_K,
    _topk_overlap,
)
from .mamba2_kaggle_k2n_ulp_audit import (
    SCHEMA as K2N_SCHEMA,
    GENERATED_TOKENS,
    _aggregate_max,
    _blank_metrics,
    _candidate_gap_ulp,
    _pytorch_reference,
    _state_shapes,
)
from .mamba2_onnx import verify_mamba2_g04_bundle


SCHEMA = "VN97M2K2OHOLDOUT1"

MAX_REFERENCE_GAP_ULP_FOR_MISMATCH = 1.0
MIN_TOP5_OVERLAP_FOR_MISMATCH = 4

HOLDOUT_PROMPTS = (
    "A robust recurrent model should preserve",
    "When two floating point implementations disagree slightly,",
    "Write pseudocode for a bounded retry loop that",
    "Compare sequential and parallel execution when",
    "For mobile inference, memory bandwidth matters because",
    "Given prices 101, 103, 102, reason about",
    "In a game agent, simultaneous actions require",
    "A safe autonomous planner should request approval before",
    "Một mô hình chạy trên điện thoại cần cân bằng",
    "Khi hai phép tính số thực chỉ lệch rất nhỏ,",
    "Hãy mô tả cách bộ nhớ dài hạn hỗ trợ",
    "Trong giao dịch theo chuỗi thời gian, cần chú ý",
)


def _holdout_cases(tokenizer: Any) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, prompt in enumerate(HOLDOUT_PROMPTS):
        ids = tokenizer(
            prompt,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        token_ids = [int(value) for value in ids]
        if not token_ids:
            raise RuntimeError(f"K2O holdout prompt {index} tokenized empty")
        cases.append(
            {
                "prompt_index": index,
                "prompt": prompt,
                "token_ids": token_ids,
            }
        )
    return cases


def _gate(
    *,
    nonfinite_count: int,
    mismatch_records: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if nonfinite_count != 0:
        failures.append("nonfinite_logits")

    for item in mismatch_records:
        if (
            float(item["reference_candidate_gap_ulp"])
            > MAX_REFERENCE_GAP_ULP_FOR_MISMATCH + 1.0e-6
        ):
            failures.append(
                "mismatch_exceeds_one_reference_fp16_ulp"
            )
            break

    for item in mismatch_records:
        if int(item["top5_overlap"]) < MIN_TOP5_OVERLAP_FOR_MISMATCH:
            failures.append("mismatch_top5_overlap_below_four")
            break

    return (len(failures) == 0, failures)


def run_k2o(
    *,
    capsule_root: Path,
    bundle_dir: Path,
    k2n_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("K2O requires CUDA")
    gpu_name = torch.cuda.get_device_name(0)
    if "T4" not in gpu_name.upper():
        raise RuntimeError(f"K2O expects Kaggle T4, got {gpu_name!r}")

    k2n = _verify_receipt(
        k2n_receipt_path,
        schema=K2N_SCHEMA,
        label="K2N receipt",
    )
    if k2n.get("diagnosis") not in {
        "exact_same_input_decision_parity",
        "only_fp16_near_tie_decision_divergences",
    }:
        raise ValueError(
            "K2O requires K2N exact parity or only FP16 near-tie divergences"
        )
    if k2n.get("mismatches_outside_one_reference_ulp") != 0:
        raise ValueError("K2O requires zero K2N mismatches outside one ULP")
    if k2n.get("production_graph_changed") is not False:
        raise ValueError("K2O requires unchanged G0.4 production graph")

    manifest = verify_mamba2_g04_bundle(bundle_dir)
    for field in ("g04_manifest_id", "capsule_id", "source_weight_sha256"):
        manifest_field = "manifest_id" if field == "g04_manifest_id" else field
        if manifest.get(manifest_field) != k2n.get(field):
            raise ValueError(f"K2O {field} lineage mismatch")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(capsule_root / "tokenizer"),
        local_files_only=True,
        use_fast=True,
    )
    prompt_cases = _holdout_cases(tokenizer)

    references = _pytorch_reference(
        capsule_root=capsule_root,
        prompt_cases=prompt_cases,
    )

    free_after_reference, total = torch.cuda.mem_get_info()
    if free_after_reference < 10 * 1024**3:
        raise RuntimeError(
            "K2O did not release enough T4 VRAM before ORT session creation"
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
            "K2O ORT session did not activate CUDAExecutionProvider"
        )

    conv_shape, ssm_shape = _state_shapes(manifest)
    epsilon = float(np.finfo(np.float16).eps)
    aggregate = _blank_metrics()
    mismatch_records: list[dict[str, Any]] = []
    prompt_summaries: list[dict[str, Any]] = []
    total_decisions = 0
    exact_decisions = 0
    nonfinite_count = 0
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
            raise RuntimeError("K2O ORT prompt priming produced no logits")

        prompt_mismatches = 0
        prompt_metrics = _blank_metrics()
        prompt_top5_min = TOP_K
        prompt_top5_sum = 0

        for step in range(1, GENERATED_TOKENS + 1):
            ref_logits = reference["prediction_logits"][step - 1]
            ref_top1 = int(reference["generated_token_ids"][step - 1])
            ort_top1 = int(logits.argmax())

            if not np.isfinite(ref_logits).all():
                nonfinite_count += 1
            if not np.isfinite(logits).all():
                nonfinite_count += 1

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
                prompt_mismatches += 1
                gap = _candidate_gap_ulp(
                    ref_logits,
                    reference_top1=ref_top1,
                    candidate_token=ort_top1,
                )
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

            # Holdout gate is same-input by construction: ORT always receives
            # the PyTorch reference token, never its own divergent top-1.
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
                "top5_overlap_min": prompt_top5_min,
                "top5_overlap_mean": float(
                    prompt_top5_sum / GENERATED_TOKENS
                ),
                "aggregate_logit_metrics": prompt_metrics,
            }
        )

    del session
    gc.collect()

    gate_passed, gate_failures = _gate(
        nonfinite_count=nonfinite_count,
        mismatch_records=mismatch_records,
    )

    body: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS" if gate_passed else "FAIL",
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "ort_provider": active[0],
        "k2n_receipt_id": k2n["receipt_id"],
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
        "mismatch_count": len(mismatch_records),
        "nonfinite_count": nonfinite_count,
        "top5_overlap_min": top5_overlap_min,
        "top5_overlap_mean": float(
            top5_overlap_sum / total_decisions
        ),
        "aggregate_logit_metrics": aggregate,
        "prompt_summaries": prompt_summaries,
        "mismatches": mismatch_records,
        "acceptance_policy": {
            "predeclared_before_holdout_measurement": True,
            "require_finite_logits": True,
            "max_reference_gap_ulp_for_any_mismatch":
                MAX_REFERENCE_GAP_ULP_FOR_MISMATCH,
            "min_top5_overlap_for_any_mismatch":
                MIN_TOP5_OVERLAP_FOR_MISMATCH,
            "teacher_forced_same_input": True,
        },
        "gate_passed": gate_passed,
        "gate_failures": gate_failures,
        "weights_changed": False,
        "architecture_changed": False,
        "production_graph_changed": False,
        "acceptance_threshold_defined": True,
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
    parser.add_argument("--k2n-receipt", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/vn97-k2o-holdout-gate.json"),
    )
    args = parser.parse_args()

    receipt = run_k2o(
        capsule_root=args.capsule_root.resolve(strict=True),
        bundle_dir=args.bundle_dir.resolve(strict=True),
        k2n_receipt_path=args.k2n_receipt.resolve(strict=True),
        output_path=args.output,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
