from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .mamba2_kaggle_k2a_ort import _canonical_json
from .mamba2_kaggle_k2f_dbx_operands import _verify_receipt
from .mamba2_kaggle_k2j_closed_loop import SCHEMA as K2J_SCHEMA
from .mamba2_kaggle_k2k_state_swap import (
    SCHEMA as K2K_SCHEMA,
    _capture_ort_prestate,
    _capture_pytorch_prestate,
    _ort_session,
)
from .mamba2_kaggle_k2l_layer_influence import (
    SCHEMA as K2L_SCHEMA,
    _record,
    _run_logits,
)
from .mamba2_onnx import verify_mamba2_g04_bundle


SCHEMA = "VN97M2K2MTIE1"
ALPHAS = (0.0, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0)
CANDIDATES_PER_MODE = 2


def _select_candidates(k2l: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for mode in ("conv", "ssm", "both"):
        records = k2l["top_by_mode"][mode]
        for item in records[:CANDIDATES_PER_MODE]:
            key = (int(item["layer"]), str(item["mode"]))
            if key in seen:
                continue
            seen.add(key)
            selected.append(
                {
                    "layer": key[0],
                    "mode": key[1],
                    "k2l_gap_influence": float(item["gap_influence"]),
                    "k2l_gap": float(item["reference_minus_ort_logit"]),
                }
            )
    if not selected:
        raise ValueError("K2M K2L candidate list is empty")
    return selected


def _interpolate(
    ort_value: np.ndarray,
    py_value: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    mixed = (
        ort_value.astype(np.float32)
        + np.float32(alpha)
        * (
            py_value.astype(np.float32)
            - ort_value.astype(np.float32)
        )
    )
    return mixed.astype(ort_value.dtype)


def _perturbation_summary(
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    left = baseline.astype(np.float32, copy=False)
    right = candidate.astype(np.float32, copy=False)
    diff = np.abs(right - left)
    changed = int(np.count_nonzero(candidate != baseline))
    return {
        "changed_fraction": float(changed / baseline.size),
        "max_abs_delta": float(diff.max(initial=0.0)),
        "mean_abs_delta": float(diff.mean()) if diff.size else 0.0,
    }


def _diagnose(candidates: list[dict[str, Any]]) -> str:
    repaired = [
        item
        for item in candidates
        if item["minimal_alpha_to_reference"] is not None
    ]
    if not repaired:
        return "no_interpolation_repairs_tie"

    low = [
        item
        for item in repaired
        if float(item["minimal_alpha_to_reference"]) <= 0.25
    ]
    low_layers = {int(item["layer"]) for item in low}
    low_modes = {str(item["mode"]) for item in low}

    if len(low) >= 3 and len(low_layers) >= 2:
        return "distributed_near_tie_sensitivity"
    if len(low) >= 2 and len(low_modes) >= 2:
        return "multi_component_near_tie_sensitivity"

    ranked = sorted(
        repaired,
        key=lambda item: (
            float(item["minimal_alpha_to_reference"]),
            -float(item["k2l_gap_influence"]),
        ),
    )
    if (
        len(ranked) == 1
        or (
            float(ranked[0]["minimal_alpha_to_reference"]) <= 0.25
            and float(ranked[1]["minimal_alpha_to_reference"]) >= 0.75
        )
    ):
        return "localized_state_sensitivity"
    return "broad_near_tie_sensitivity"


def run_k2m(
    *,
    bundle_dir: Path,
    k2j_receipt_path: Path,
    k2k_receipt_path: Path,
    k2l_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("K2M requires CUDA")
    gpu_name = torch.cuda.get_device_name(0)
    if "T4" not in gpu_name.upper():
        raise RuntimeError(f"K2M expects Kaggle T4, got {gpu_name!r}")

    k2j = _verify_receipt(
        k2j_receipt_path,
        schema=K2J_SCHEMA,
        label="K2J receipt",
    )
    k2k = _verify_receipt(
        k2k_receipt_path,
        schema=K2K_SCHEMA,
        label="K2K receipt",
    )
    k2l = _verify_receipt(
        k2l_receipt_path,
        schema=K2L_SCHEMA,
        label="K2L receipt",
    )

    if k2k.get("k2j_receipt_id") != k2j.get("receipt_id"):
        raise ValueError("K2M K2J/K2K lineage mismatch")
    if k2l.get("k2j_receipt_id") != k2j.get("receipt_id"):
        raise ValueError("K2M K2J/K2L lineage mismatch")
    if k2l.get("k2k_receipt_id") != k2k.get("receipt_id"):
        raise ValueError("K2M K2K/K2L lineage mismatch")
    if k2k.get("diagnosis") != "mixed_runtime_and_state":
        raise ValueError("K2M requires K2K mixed runtime/state evidence")
    if k2l.get("diagnosis") not in {
        "single_layer_state_interaction_localized",
        "component_layer_interaction_localized",
    }:
        raise ValueError("K2M requires K2L layer-local repair evidence")

    manifest = verify_mamba2_g04_bundle(bundle_dir)
    for field in ("g04_manifest_id", "capsule_id", "source_weight_sha256"):
        manifest_field = "manifest_id" if field == "g04_manifest_id" else field
        if manifest.get(manifest_field) != k2l.get(field):
            raise ValueError(f"K2M {field} lineage mismatch")

    prompt_index = int(k2l["prompt_index"])
    divergence_step = int(k2l["divergence_step"])
    prompt = next(
        (
            item
            for item in k2j["prompts"]
            if int(item["prompt_index"]) == prompt_index
        ),
        None,
    )
    if prompt is None:
        raise ValueError("K2M target prompt missing")

    prompt_token_ids = [int(v) for v in prompt["prompt_token_ids"]]
    reference_generated = [
        int(v) for v in prompt["reference_generated_token_ids"]
    ]
    before_current = reference_generated[: divergence_step - 2]
    current_token = int(k2l["current_input_token"])
    reference_token = int(k2l["reference_next_token"])
    ort_token = int(k2l["ort_next_token"])

    capsule_root = bundle_dir.parent / "g03-capsule"
    py_conv, py_ssm = _capture_pytorch_prestate(
        capsule_root=capsule_root,
        prompt_token_ids=prompt_token_ids,
        common_generated_before_current=before_current,
    )
    ort_conv, ort_ssm, provider = _capture_ort_prestate(
        graph_path=bundle_dir / "step.onnx",
        prompt_token_ids=prompt_token_ids,
        common_generated_before_current=before_current,
        zero_conv=py_conv,
        zero_ssm=py_ssm,
    )

    session, provider2 = _ort_session(bundle_dir / "step.onnx")
    if provider2 != provider:
        raise RuntimeError("K2M ORT provider changed between phases")

    baseline_logits = _run_logits(
        session,
        current_token=current_token,
        conv=ort_conv,
        ssm=ort_ssm,
    )
    baseline_gap = float(k2l["baseline"]["reference_minus_ort_logit"])
    baseline = _record(
        baseline_logits,
        reference_token=reference_token,
        ort_token=ort_token,
        baseline_gap=baseline_gap,
    )
    if int(baseline["top1"]) != ort_token:
        raise RuntimeError("K2M failed to reproduce divergent ORT baseline")

    candidates = _select_candidates(k2l)
    measured: list[dict[str, Any]] = []

    for candidate in candidates:
        layer = int(candidate["layer"])
        mode = str(candidate["mode"])
        curve: list[dict[str, Any]] = []
        minimal_alpha: float | None = None

        for alpha in ALPHAS:
            conv = ort_conv
            ssm = ort_ssm
            conv_summary = {
                "changed_fraction": 0.0,
                "max_abs_delta": 0.0,
                "mean_abs_delta": 0.0,
            }
            ssm_summary = dict(conv_summary)

            if mode in {"conv", "both"}:
                conv = ort_conv.copy()
                mixed = _interpolate(
                    ort_conv[layer],
                    py_conv[layer],
                    alpha=alpha,
                )
                conv[layer] = mixed
                conv_summary = _perturbation_summary(
                    ort_conv[layer],
                    mixed,
                )

            if mode in {"ssm", "both"}:
                ssm = ort_ssm.copy()
                mixed = _interpolate(
                    ort_ssm[layer],
                    py_ssm[layer],
                    alpha=alpha,
                )
                ssm[layer] = mixed
                ssm_summary = _perturbation_summary(
                    ort_ssm[layer],
                    mixed,
                )

            logits = _run_logits(
                session,
                current_token=current_token,
                conv=conv,
                ssm=ssm,
            )
            record = _record(
                logits,
                reference_token=reference_token,
                ort_token=ort_token,
                baseline_gap=baseline_gap,
            )
            record.update(
                {
                    "alpha": alpha,
                    "conv_perturbation": conv_summary,
                    "ssm_perturbation": ssm_summary,
                }
            )
            curve.append(record)
            if (
                minimal_alpha is None
                and alpha > 0.0
                and record["top1_class"] == "reference"
            ):
                minimal_alpha = alpha

        measured.append(
            {
                **candidate,
                "minimal_alpha_to_reference": minimal_alpha,
                "curve": curve,
            }
        )

    del session
    gc.collect()

    diagnosis = _diagnose(measured)
    ranked_thresholds = sorted(
        [
            {
                "layer": int(item["layer"]),
                "mode": item["mode"],
                "minimal_alpha_to_reference":
                    item["minimal_alpha_to_reference"],
                "k2l_gap_influence": float(item["k2l_gap_influence"]),
            }
            for item in measured
        ],
        key=lambda item: (
            2.0
            if item["minimal_alpha_to_reference"] is None
            else float(item["minimal_alpha_to_reference"]),
            -float(item["k2l_gap_influence"]),
        ),
    )

    body: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "MEASURED",
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "ort_provider": provider,
        "k2j_receipt_id": k2j["receipt_id"],
        "k2k_receipt_id": k2k["receipt_id"],
        "k2l_receipt_id": k2l["receipt_id"],
        "g04_manifest_id": manifest["manifest_id"],
        "capsule_id": manifest["capsule_id"],
        "source_weight_sha256": manifest["source_weight_sha256"],
        "dtype": str(manifest["state_contract"]["dtype"]),
        "prompt_index": prompt_index,
        "divergence_step": divergence_step,
        "current_input_token": current_token,
        "reference_next_token": reference_token,
        "ort_next_token": ort_token,
        "alphas": list(ALPHAS),
        "baseline": baseline,
        "candidates": measured,
        "ranked_thresholds": ranked_thresholds,
        "diagnosis": diagnosis,
        "weights_changed": False,
        "architecture_changed": False,
        "production_graph_changed": False,
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
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--k2j-receipt", required=True, type=Path)
    parser.add_argument("--k2k-receipt", required=True, type=Path)
    parser.add_argument("--k2l-receipt", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/vn97-k2m-tie-sensitivity.json"),
    )
    args = parser.parse_args()

    receipt = run_k2m(
        bundle_dir=args.bundle_dir.resolve(strict=True),
        k2j_receipt_path=args.k2j_receipt.resolve(strict=True),
        k2k_receipt_path=args.k2k_receipt.resolve(strict=True),
        k2l_receipt_path=args.k2l_receipt.resolve(strict=True),
        output_path=args.output,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
