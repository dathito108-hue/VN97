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
from .mamba2_kaggle_k2j_closed_loop import (
    TOP_K,
    _topk_overlap,
)
from .mamba2_kaggle_k2n_ulp_audit import _candidate_gap_ulp
from .mamba2_kaggle_k2o_holdout_gate import SCHEMA as K2O_SCHEMA
from .mamba2_onnx import (
    Mamba2OnnxConfig,
    VN97Mamba2StepOnnx,
)
from .mamba2_parallel_onnx import verify_g05_bundle


SCHEMA = "VN97M2K2PG05BRIDGE1"
CHUNK_SIZE = 8
DECODE_TOKENS = 32
CONTINUATION_TOKENS = 8
MAX_REFERENCE_GAP_ULP_FOR_MISMATCH = 1.0
MIN_TOP5_OVERLAP_FOR_MISMATCH = 4


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"K2P {label} must be lowercase SHA-256")
    return value


def _verify_k2o_receipt(path: Path) -> dict[str, Any]:
    try:
        receipt = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("K2P K2O receipt is unreadable") from exc
    if not isinstance(receipt, dict):
        raise ValueError("K2P K2O receipt must be an object")
    if receipt.get("schema") != K2O_SCHEMA:
        raise ValueError("K2P K2O receipt schema mismatch")
    if receipt.get("status") != "PASS":
        raise ValueError("K2P K2O receipt must be PASS")
    if receipt.get("gate_passed") is not True:
        raise ValueError("K2P K2O gate must have passed")
    if receipt.get("gate_failures") != []:
        raise ValueError("K2P K2O gate failures must be empty")
    if receipt.get("acceptance_threshold_defined") is not True:
        raise ValueError("K2P K2O receipt must carry its predeclared gate")
    policy = receipt.get("acceptance_policy")
    if not isinstance(policy, dict):
        raise ValueError("K2P K2O acceptance policy missing")
    if policy.get("predeclared_before_holdout_measurement") is not True:
        raise ValueError("K2P K2O gate was not predeclared")
    if receipt.get("production_graph_changed") is not False:
        raise ValueError("K2P requires unchanged K2O G0.4 graph")
    if receipt.get("production_activation_authorized") is not False:
        raise ValueError("K2P K2O receipt must not authorize production")

    _require_sha256(receipt.get("capsule_id"), "K2O capsule ID")
    _require_sha256(
        receipt.get("source_weight_sha256"),
        "K2O source weight SHA-256",
    )
    _require_sha256(
        receipt.get("g04_manifest_id"),
        "K2O G0.4 manifest ID",
    )
    receipt_id = _require_sha256(
        receipt.get("receipt_id"),
        "K2O receipt ID",
    )
    body = dict(receipt)
    body.pop("receipt_id", None)
    expected = hashlib.sha256(
        K2O_SCHEMA.encode("ascii") + b"\0" + _canonical_json(body)
    ).hexdigest()
    if receipt_id != expected:
        raise ValueError("K2P K2O receipt identity mismatch")
    return receipt

DECODE_PROMPTS = (
    "A recurrent mobile model preserves state by",
    "Explain the difference between decode and prefill",
    "A safe tool-using assistant should verify",
    "For low latency inference on a phone,",
    "Trong suy luận tuần tự trên điện thoại,",
    "Một trợ lý tự chủ phải kiểm tra quyền trước khi",
)

PREFILL_CASES = (
    ("Parallel prefill must preserve recurrent state exactly enough for", 1),
    ("Chunked prompt processing should produce the same continuation as", 2),
    ("Mobile inference benefits from bounded prompt chunks when state remains", 4),
    ("A single shared weight graph should handle prompt prefill and recurrent decode without changing model semantics", 8),
)


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


def _gate(
    *,
    nonfinite_count: int,
    mismatches: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if nonfinite_count != 0:
        failures.append("nonfinite_logits")

    for item in mismatches:
        if (
            float(item["reference_candidate_gap_ulp"])
            > MAX_REFERENCE_GAP_ULP_FOR_MISMATCH + 1.0e-6
        ):
            failures.append("mismatch_exceeds_one_reference_fp16_ulp")
            break

    for item in mismatches:
        if int(item["top5_overlap"]) < MIN_TOP5_OVERLAP_FOR_MISMATCH:
            failures.append("mismatch_top5_overlap_below_four")
            break

    return len(failures) == 0, failures


def _tokenize_decode_prompts(tokenizer: Any) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, prompt in enumerate(DECODE_PROMPTS):
        ids = tokenizer(
            prompt,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        token_ids = [int(value) for value in ids]
        if not token_ids:
            raise RuntimeError(f"K2P decode prompt {index} tokenized empty")
        cases.append(
            {
                "prompt_index": index,
                "token_ids": token_ids,
            }
        )
    return cases


def _tokenize_prefill_cases(tokenizer: Any) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, (prompt, valid_length) in enumerate(PREFILL_CASES):
        ids = tokenizer(
            prompt,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        token_ids = [int(value) for value in ids]
        if len(token_ids) < valid_length:
            raise RuntimeError(
                f"K2P prefill case {index} has only {len(token_ids)} tokens"
            )
        cases.append(
            {
                "case_index": index,
                "valid_length": valid_length,
                "token_ids": token_ids[:valid_length],
            }
        )
    return cases


@torch.inference_mode()
def _pytorch_reference(
    *,
    capsule_root: Path,
    decode_cases: list[dict[str, Any]],
    prefill_cases: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not torch.cuda.is_available():
        raise RuntimeError("K2P PyTorch reference requires CUDA")

    device = torch.device("cuda:0")
    capsule = load_g03_capsule(
        capsule_root,
        verify_large_weight_sha256=False,
    )
    model = VN97Mamba2StepOnnx(
        Mamba2OnnxConfig.from_source_spec(capsule.spec),
        capsule.tensors,
    ).eval().to(device)

    decode_refs: list[dict[str, Any]] = []
    for case in decode_cases:
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
            raise RuntimeError("K2P decode priming produced no logits")

        generated: list[int] = []
        prediction_logits: list[np.ndarray] = []
        for _ in range(DECODE_TOKENS):
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

        decode_refs.append(
            {
                "prompt_index": int(case["prompt_index"]),
                "prompt_token_ids": list(case["token_ids"]),
                "generated_token_ids": generated,
                "prediction_logits": prediction_logits,
            }
        )

    prefill_refs: list[dict[str, Any]] = []
    for case in prefill_cases:
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
            raise RuntimeError("K2P prefill reference produced no logits")

        final_prefill_logits = logits.detach().cpu().numpy().copy()
        conv_after_prefill = conv.detach().cpu().numpy().copy()
        ssm_after_prefill = ssm.detach().cpu().numpy().copy()

        continuation_ids: list[int] = []
        continuation_logits: list[np.ndarray] = []
        for _ in range(CONTINUATION_TOKENS):
            current = logits.detach().cpu().numpy().copy()
            next_token = int(current.argmax())
            continuation_logits.append(current)
            continuation_ids.append(next_token)
            token = torch.tensor(
                [next_token],
                dtype=torch.long,
                device=device,
            )
            logits, conv, ssm = model(token, conv, ssm)

        prefill_refs.append(
            {
                "case_index": int(case["case_index"]),
                "valid_length": int(case["valid_length"]),
                "prompt_token_ids": list(case["token_ids"]),
                "final_prefill_logits": final_prefill_logits,
                "conv_state": conv_after_prefill,
                "ssm_state": ssm_after_prefill,
                "continuation_token_ids": continuation_ids,
                "continuation_logits": continuation_logits,
            }
        )

    del model
    del capsule
    gc.collect()
    torch.cuda.empty_cache()
    return decode_refs, prefill_refs


def _g05_shapes(manifest: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    state = manifest["state_contract"]
    conv = tuple(int(value) for value in state["conv_state_shape"])
    ssm = tuple(int(value) for value in state["ssm_state_shape"])
    if not conv or not ssm:
        raise ValueError("K2P invalid G0.5 state shapes")
    return conv, ssm


def _run_chunk(
    session: Any,
    *,
    chunk_size: int,
    token_ids: list[int],
    valid_length: int,
    conv: np.ndarray,
    ssm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 1 <= valid_length <= chunk_size:
        raise ValueError("K2P valid_length outside G0.5 chunk")
    if len(token_ids) != valid_length:
        raise ValueError("K2P token count must equal valid_length")

    padded = np.zeros((1, chunk_size), dtype=np.int64)
    padded[0, :valid_length] = np.asarray(token_ids, dtype=np.int64)

    outputs = session.run(
        ["logits", "next_conv_state", "next_ssm_state"],
        {
            "input_ids": padded,
            "valid_length": np.asarray([valid_length], dtype=np.int64),
            "conv_state": conv,
            "ssm_state": ssm,
        },
    )
    logits = np.asarray(outputs[0]).copy()
    next_conv = np.asarray(outputs[1]).copy()
    next_ssm = np.asarray(outputs[2]).copy()
    return logits, next_conv, next_ssm


def _final_logits(logits: np.ndarray, valid_length: int) -> np.ndarray:
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("K2P G0.5 logits must be [1,chunk,vocab]")
    if not 1 <= valid_length <= logits.shape[1]:
        raise ValueError("K2P final-logit index outside graph output")
    return logits[:, valid_length - 1, :]


def _compare_decision(
    *,
    reference_logits: np.ndarray,
    ort_logits: np.ndarray,
    prompt_index: int,
    step: int,
    phase: str,
    epsilon: float,
) -> tuple[bool, dict[str, Any] | None, dict[str, float], int]:
    if not np.isfinite(reference_logits).all() or not np.isfinite(ort_logits).all():
        return False, None, _blank_metrics(), 0

    metrics = _epsilon_metrics(
        reference_logits,
        ort_logits,
        epsilon=epsilon,
    )
    reference_top1 = int(reference_logits.argmax())
    ort_top1 = int(ort_logits.argmax())
    overlap = _topk_overlap(reference_logits, ort_logits, TOP_K)

    if reference_top1 == ort_top1:
        return True, None, metrics, overlap

    gap = _candidate_gap_ulp(
        reference_logits,
        reference_top1=reference_top1,
        candidate_token=ort_top1,
    )
    mismatch = {
        "phase": phase,
        "prompt_index": prompt_index,
        "step": step,
        "reference_top1": reference_top1,
        "ort_top1": ort_top1,
        **gap,
        "top5_overlap": overlap,
        "logit_metrics": metrics,
    }
    return False, mismatch, metrics, overlap


def run_k2p(
    *,
    capsule_root: Path,
    g05_bundle_dir: Path,
    k2o_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("K2P requires CUDA")
    gpu_name = torch.cuda.get_device_name(0)
    if "T4" not in gpu_name.upper():
        raise RuntimeError(f"K2P expects Kaggle T4, got {gpu_name!r}")

    k2o = _verify_k2o_receipt(k2o_receipt_path)

    manifest = verify_g05_bundle(g05_bundle_dir)
    if int(manifest["max_chunk_size"]) != CHUNK_SIZE:
        raise ValueError(
            f"K2P currently requires G0.5 chunk size {CHUNK_SIZE}"
        )
    if manifest.get("state_contract", {}).get("dtype") != "float16":
        raise ValueError("K2P real-transfer bridge requires FP16 G0.5 state")
    if manifest.get("capsule_id") != k2o.get("capsule_id"):
        raise ValueError("K2P G0.5/K2O capsule lineage mismatch")
    if manifest.get("source_weight_sha256") != k2o.get("source_weight_sha256"):
        raise ValueError("K2P G0.5/K2O source-weight lineage mismatch")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(capsule_root / "tokenizer"),
        local_files_only=True,
        use_fast=True,
    )
    decode_cases = _tokenize_decode_prompts(tokenizer)
    prefill_cases = _tokenize_prefill_cases(tokenizer)

    decode_refs, prefill_refs = _pytorch_reference(
        capsule_root=capsule_root,
        decode_cases=decode_cases,
        prefill_cases=prefill_cases,
    )

    free_after_reference, total = torch.cuda.mem_get_info()
    if free_after_reference < 9 * 1024**3:
        raise RuntimeError(
            "K2P did not release enough T4 VRAM before G0.5 ORT session"
        )

    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()

    graph = g05_bundle_dir / f"recurrent-{CHUNK_SIZE}.onnx"
    session = ort.InferenceSession(
        str(graph),
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    active = session.get_providers()
    if not active or active[0] != "CUDAExecutionProvider":
        raise RuntimeError(
            "K2P G0.5 ORT session did not activate CUDAExecutionProvider"
        )

    conv_shape, ssm_shape = _g05_shapes(manifest)
    epsilon = float(np.finfo(np.float16).eps)

    aggregate = _blank_metrics()
    state_aggregate = {
        "conv_state": _blank_metrics(),
        "ssm_state": _blank_metrics(),
    }
    mismatches: list[dict[str, Any]] = []
    total_decisions = 0
    exact_decisions = 0
    nonfinite_count = 0
    top5_overlap_min = TOP_K
    top5_overlap_sum = 0
    decode_summaries: list[dict[str, Any]] = []
    prefill_summaries: list[dict[str, Any]] = []

    for reference in decode_refs:
        conv = np.zeros(conv_shape, dtype=np.float16)
        ssm = np.zeros(ssm_shape, dtype=np.float16)
        ort_decision: np.ndarray | None = None

        for token_id in reference["prompt_token_ids"]:
            out, conv, ssm = _run_chunk(
                session,
                chunk_size=CHUNK_SIZE,
                token_ids=[int(token_id)],
                valid_length=1,
                conv=conv,
                ssm=ssm,
            )
            ort_decision = _final_logits(out, 1)

        if ort_decision is None:
            raise RuntimeError("K2P decode priming produced no logits")

        prompt_mismatches = 0
        for step in range(1, DECODE_TOKENS + 1):
            ref_logits = reference["prediction_logits"][step - 1]
            if not np.isfinite(ref_logits).all() or not np.isfinite(ort_decision).all():
                nonfinite_count += 1

            exact, mismatch, metrics, overlap = _compare_decision(
                reference_logits=ref_logits,
                ort_logits=ort_decision,
                prompt_index=int(reference["prompt_index"]),
                step=step,
                phase="decode",
                epsilon=epsilon,
            )
            _aggregate_max(aggregate, metrics)
            total_decisions += 1
            exact_decisions += int(exact)
            top5_overlap_min = min(top5_overlap_min, overlap)
            top5_overlap_sum += overlap
            if mismatch is not None:
                mismatches.append(mismatch)
                prompt_mismatches += 1

            reference_token = int(reference["generated_token_ids"][step - 1])
            out, conv, ssm = _run_chunk(
                session,
                chunk_size=CHUNK_SIZE,
                token_ids=[reference_token],
                valid_length=1,
                conv=conv,
                ssm=ssm,
            )
            ort_decision = _final_logits(out, 1)

        decode_summaries.append(
            {
                "prompt_index": int(reference["prompt_index"]),
                "decision_count": DECODE_TOKENS,
                "mismatch_count": prompt_mismatches,
            }
        )

    for reference in prefill_refs:
        conv = np.zeros(conv_shape, dtype=np.float16)
        ssm = np.zeros(ssm_shape, dtype=np.float16)
        valid_length = int(reference["valid_length"])
        logits, conv, ssm = _run_chunk(
            session,
            chunk_size=CHUNK_SIZE,
            token_ids=[int(v) for v in reference["prompt_token_ids"]],
            valid_length=valid_length,
            conv=conv,
            ssm=ssm,
        )
        ort_decision = _final_logits(logits, valid_length)

        conv_metrics = _epsilon_metrics(
            reference["conv_state"],
            conv,
            epsilon=epsilon,
        )
        ssm_metrics = _epsilon_metrics(
            reference["ssm_state"],
            ssm,
            epsilon=epsilon,
        )
        _aggregate_max(state_aggregate["conv_state"], conv_metrics)
        _aggregate_max(state_aggregate["ssm_state"], ssm_metrics)

        case_mismatches = 0
        for step in range(1, CONTINUATION_TOKENS + 1):
            ref_logits = reference["continuation_logits"][step - 1]
            if not np.isfinite(ref_logits).all() or not np.isfinite(ort_decision).all():
                nonfinite_count += 1

            exact, mismatch, metrics, overlap = _compare_decision(
                reference_logits=ref_logits,
                ort_logits=ort_decision,
                prompt_index=int(reference["case_index"]),
                step=step,
                phase=f"prefill_v{valid_length}",
                epsilon=epsilon,
            )
            _aggregate_max(aggregate, metrics)
            total_decisions += 1
            exact_decisions += int(exact)
            top5_overlap_min = min(top5_overlap_min, overlap)
            top5_overlap_sum += overlap
            if mismatch is not None:
                mismatches.append(mismatch)
                case_mismatches += 1

            token_id = int(reference["continuation_token_ids"][step - 1])
            out, conv, ssm = _run_chunk(
                session,
                chunk_size=CHUNK_SIZE,
                token_ids=[token_id],
                valid_length=1,
                conv=conv,
                ssm=ssm,
            )
            ort_decision = _final_logits(out, 1)

        prefill_summaries.append(
            {
                "case_index": int(reference["case_index"]),
                "valid_length": valid_length,
                "continuation_decisions": CONTINUATION_TOKENS,
                "mismatch_count": case_mismatches,
                "conv_state_metrics_after_prefill": conv_metrics,
                "ssm_state_metrics_after_prefill": ssm_metrics,
            }
        )

    del session
    gc.collect()

    gate_passed, gate_failures = _gate(
        nonfinite_count=nonfinite_count,
        mismatches=mismatches,
    )

    body: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS" if gate_passed else "FAIL",
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "ort_providers": active,
        "k2o_receipt_id": k2o["receipt_id"],
        "g05_manifest_id": manifest["manifest_id"],
        "capsule_id": manifest["capsule_id"],
        "source_weight_sha256": manifest["source_weight_sha256"],
        "chunk_size": CHUNK_SIZE,
        "dtype": manifest["state_contract"]["dtype"],
        "decode_prompt_count": len(decode_refs),
        "decode_tokens_per_prompt": DECODE_TOKENS,
        "prefill_case_count": len(prefill_refs),
        "prefill_continuation_tokens": CONTINUATION_TOKENS,
        "total_same_input_decisions": total_decisions,
        "exact_decisions": exact_decisions,
        "exact_decision_fraction": float(exact_decisions / total_decisions),
        "mismatch_count": len(mismatches),
        "nonfinite_count": nonfinite_count,
        "top5_overlap_min": top5_overlap_min,
        "top5_overlap_mean": float(top5_overlap_sum / total_decisions),
        "aggregate_logit_metrics": aggregate,
        "prefill_state_metrics_diagnostic_only": state_aggregate,
        "decode_summaries": decode_summaries,
        "prefill_summaries": prefill_summaries,
        "mismatches": mismatches,
        "acceptance_policy": {
            "predeclared_before_measurement": True,
            "require_finite_logits": True,
            "max_reference_gap_ulp_for_any_mismatch":
                MAX_REFERENCE_GAP_ULP_FOR_MISMATCH,
            "min_top5_overlap_for_any_mismatch":
                MIN_TOP5_OVERLAP_FOR_MISMATCH,
            "teacher_forced_same_input": True,
            "prefill_state_error_is_diagnostic_only": True,
        },
        "gate_passed": gate_passed,
        "gate_failures": gate_failures,
        "g04_to_g05_mobile_graph_bridge_passed": gate_passed,
        "android_execution_measured": False,
        "weights_changed": False,
        "architecture_changed": False,
        "quantization_used": False,
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
    parser.add_argument("--g05-bundle-dir", required=True, type=Path)
    parser.add_argument("--k2o-receipt", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/vn97-k2p-g05-bridge.json"),
    )
    args = parser.parse_args()

    receipt = run_k2p(
        capsule_root=args.capsule_root.resolve(strict=True),
        g05_bundle_dir=args.g05_bundle_dir.resolve(strict=True),
        k2o_receipt_path=args.k2o_receipt.resolve(strict=True),
        output_path=args.output,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
