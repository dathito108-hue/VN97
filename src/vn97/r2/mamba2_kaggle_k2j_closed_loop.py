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
from .mamba2_kaggle_k2i_trajectory import (
    SCHEMA as K2I_SCHEMA,
    _top1_margin,
)
from .mamba2_onnx import (
    Mamba2OnnxConfig,
    VN97Mamba2StepOnnx,
    verify_mamba2_g04_bundle,
)


SCHEMA = "VN97M2K2JCLOSED1"
GENERATED_TOKENS = 32
CHECKPOINT_STEPS = (1, 2, 4, 8, 16, 32)
TOP_K = 5
PROMPTS = (
    "State space models can reason about",
    "A mobile assistant should carefully",
    "The fastest reliable inference path is",
)


def _topk_ids(logits: np.ndarray, k: int = TOP_K) -> list[int]:
    values = logits.astype(np.float32, copy=False).reshape(-1)
    if k <= 0 or k > values.size:
        raise ValueError("K2J top-k request is out of range")
    indices = np.argpartition(values, -k)[-k:]
    ordered = indices[np.argsort(values[indices])[::-1]]
    return [int(value) for value in ordered]


def _topk_overlap(left: np.ndarray, right: np.ndarray, k: int = TOP_K) -> int:
    return len(set(_topk_ids(left, k)) & set(_topk_ids(right, k)))


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    count = 0
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        count += 1
    return count


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
    aggregate["max_epsilon_units"] = max(
        aggregate["max_epsilon_units"],
        float(metrics["max_epsilon_units"]),
    )
    aggregate["mean_abs_error"] = max(
        aggregate["mean_abs_error"],
        float(metrics["mean_abs_error"]),
    )


def _tokenize_prompts(tokenizer: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index, prompt in enumerate(PROMPTS):
        ids = tokenizer(
            prompt,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        token_ids = [int(value) for value in ids]
        if not token_ids:
            raise RuntimeError(f"K2J prompt {index} tokenized empty")
        items.append(
            {
                "prompt_index": index,
                "prompt": prompt,
                "token_ids": token_ids,
            }
        )
    return items


@torch.inference_mode()
def _pytorch_closed_loop(
    *,
    capsule_root: Path,
    prompt_cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("K2J PyTorch reference requires CUDA")
    device = torch.device("cuda:0")
    capsule = load_g03_capsule(
        capsule_root,
        verify_large_weight_sha256=False,
    )
    model = VN97Mamba2StepOnnx(
        Mamba2OnnxConfig.from_source_spec(capsule.spec),
        capsule.tensors,
    ).eval().to(device)

    results: list[dict[str, Any]] = []
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
            raise RuntimeError("K2J prompt priming produced no logits")

        generated: list[int] = []
        prediction_logits: list[np.ndarray] = []
        checkpoints: dict[int, dict[str, np.ndarray]] = {}

        for step in range(1, GENERATED_TOKENS + 1):
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
            if step in CHECKPOINT_STEPS:
                checkpoints[step] = {
                    "conv_state": conv.detach().cpu().numpy().copy(),
                    "ssm_state": ssm.detach().cpu().numpy().copy(),
                }

        results.append(
            {
                "prompt_index": case["prompt_index"],
                "prompt_token_ids": case["token_ids"],
                "generated_token_ids": generated,
                "prediction_logits": prediction_logits,
                "checkpoints": checkpoints,
            }
        )

    del model
    del capsule
    gc.collect()
    torch.cuda.empty_cache()
    return results


def _run_ort_closed_loop(
    *,
    graph_path: Path,
    references: list[dict[str, Any]],
    epsilon: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
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
            "K2J ORT session did not activate CUDAExecutionProvider"
        )

    aggregate = {
        "logits": _blank_metrics(),
        "conv_state": _blank_metrics(),
        "ssm_state": _blank_metrics(),
    }
    prompt_results: list[dict[str, Any]] = []
    compared_steps = 0
    margin_certified_steps = 0
    topk_overlap_sum = 0
    topk_overlap_min = TOP_K

    for reference in references:
        first_checkpoint = reference["checkpoints"][1]
        conv = np.zeros_like(first_checkpoint["conv_state"])
        ssm = np.zeros_like(first_checkpoint["ssm_state"])
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
            raise RuntimeError("K2J ORT prompt priming produced no logits")

        ort_generated: list[int] = []
        step_records: list[dict[str, Any]] = []
        first_divergence: int | None = None
        common_prefix = True

        for step in range(1, GENERATED_TOKENS + 1):
            ort_top1 = int(logits.argmax())
            ref_top1 = int(reference["generated_token_ids"][step - 1])
            ort_generated.append(ort_top1)

            record: dict[str, Any] = {
                "step": step,
                "reference_top1": ref_top1,
                "ort_top1": ort_top1,
                "argmax_exact": ort_top1 == ref_top1,
            }

            if common_prefix:
                ref_logits = reference["prediction_logits"][step - 1]
                metrics = _epsilon_metrics(
                    ref_logits,
                    logits,
                    epsilon=epsilon,
                )
                _aggregate_max(aggregate["logits"], metrics)
                margin = _top1_margin(ref_logits)
                certified = (
                    margin > 2.0 * float(metrics["max_abs_error"])
                )
                overlap = _topk_overlap(
                    ref_logits,
                    logits,
                    TOP_K,
                )
                compared_steps += 1
                margin_certified_steps += int(certified)
                topk_overlap_sum += overlap
                topk_overlap_min = min(topk_overlap_min, overlap)

                record.update(
                    {
                        "logits": metrics,
                        "reference_top1_margin": margin,
                        "argmax_certified_by_margin": certified,
                        "top5_overlap": overlap,
                    }
                )

                if ort_top1 != ref_top1:
                    first_divergence = step
                    common_prefix = False

            token = np.asarray([ort_top1], dtype=np.int64)
            logits, conv, ssm = session.run(
                ["logits", "next_conv_state", "next_ssm_state"],
                {
                    "input_ids": token,
                    "conv_state": conv,
                    "ssm_state": ssm,
                },
            )
            logits = np.asarray(logits)
            conv = np.asarray(conv)
            ssm = np.asarray(ssm)

            if common_prefix and step in CHECKPOINT_STEPS:
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
                record["checkpoint_state"] = {
                    "conv_state": conv_metrics,
                    "ssm_state": ssm_metrics,
                }

            step_records.append(record)

        prefix = _common_prefix_length(
            reference["generated_token_ids"],
            ort_generated,
        )
        prompt_results.append(
            {
                "prompt_index": reference["prompt_index"],
                "prompt_token_ids": reference["prompt_token_ids"],
                "reference_generated_token_ids":
                    reference["generated_token_ids"],
                "ort_generated_token_ids": ort_generated,
                "common_prefix_length": prefix,
                "first_divergence_step": first_divergence,
                "closed_loop_exact": prefix == GENERATED_TOKENS,
                "steps": step_records,
            }
        )

    del session
    gc.collect()

    summary = {
        "aggregate_metrics": aggregate,
        "compared_common_prefix_steps": compared_steps,
        "margin_certified_common_prefix_steps": margin_certified_steps,
        "top5_overlap_min": topk_overlap_min if compared_steps else 0,
        "top5_overlap_mean": (
            float(topk_overlap_sum / compared_steps)
            if compared_steps
            else 0.0
        ),
    }
    return prompt_results, summary, active[0]


def run_k2j(
    *,
    capsule_root: Path,
    bundle_dir: Path,
    k2i_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("K2J requires CUDA")
    gpu_name = torch.cuda.get_device_name(0)
    if "T4" not in gpu_name.upper():
        raise RuntimeError(f"K2J expects Kaggle T4, got {gpu_name!r}")

    k2i = _verify_receipt(
        k2i_receipt_path,
        schema=K2I_SCHEMA,
        label="K2I receipt",
    )
    if k2i.get("argmax_exact_all_steps") is not True:
        raise ValueError("K2J requires K2I exact 32-step argmax evidence")
    if k2i.get("production_graph_changed") is not False:
        raise ValueError("K2J requires unchanged G0.4 production graph")

    manifest = verify_mamba2_g04_bundle(bundle_dir)
    for field in ("g04_manifest_id", "capsule_id", "source_weight_sha256"):
        manifest_field = "manifest_id" if field == "g04_manifest_id" else field
        if manifest.get(manifest_field) != k2i.get(field):
            raise ValueError(f"K2J {field} lineage mismatch")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(capsule_root / "tokenizer"),
        local_files_only=True,
        use_fast=True,
    )
    prompt_cases = _tokenize_prompts(tokenizer)

    references = _pytorch_closed_loop(
        capsule_root=capsule_root,
        prompt_cases=prompt_cases,
    )
    free_after_reference, total = torch.cuda.mem_get_info()
    if free_after_reference < 10 * 1024**3:
        raise RuntimeError(
            "K2J did not release enough T4 VRAM before ORT session creation"
        )

    epsilon = float(np.finfo(np.float16).eps)
    prompt_results, summary, provider = _run_ort_closed_loop(
        graph_path=bundle_dir / "step.onnx",
        references=references,
        epsilon=epsilon,
    )

    exact_prompts = sum(
        int(item["closed_loop_exact"])
        for item in prompt_results
    )
    minimum_prefix = min(
        int(item["common_prefix_length"])
        for item in prompt_results
    )
    first_divergences = [
        {
            "prompt_index": item["prompt_index"],
            "step": item["first_divergence_step"],
        }
        for item in prompt_results
        if item["first_divergence_step"] is not None
    ]

    body: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "MEASURED",
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "ort_provider": provider,
        "k2i_receipt_id": k2i["receipt_id"],
        "g04_manifest_id": manifest["manifest_id"],
        "capsule_id": manifest["capsule_id"],
        "source_weight_sha256": manifest["source_weight_sha256"],
        "dtype": str(manifest["state_contract"]["dtype"]),
        "fp16_epsilon": epsilon,
        "prompt_count": len(prompt_results),
        "generated_tokens_per_prompt": GENERATED_TOKENS,
        "total_generated_tokens": len(prompt_results) * GENERATED_TOKENS,
        "closed_loop_exact_prompts": exact_prompts,
        "closed_loop_exact_all_prompts": exact_prompts == len(prompt_results),
        "minimum_common_prefix_length": minimum_prefix,
        "first_divergences": first_divergences,
        **summary,
        "prompts": prompt_results,
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
    parser.add_argument("--k2i-receipt", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/vn97-k2j-closed-loop.json"),
    )
    args = parser.parse_args()

    receipt = run_k2j(
        capsule_root=args.capsule_root.resolve(strict=True),
        bundle_dir=args.bundle_dir.resolve(strict=True),
        k2i_receipt_path=args.k2i_receipt.resolve(strict=True),
        output_path=args.output,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
