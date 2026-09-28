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
from .mamba2_kaggle_k2j_closed_loop import (
    SCHEMA as K2J_SCHEMA,
    TOP_K,
    _topk_ids,
)
from .mamba2_kaggle_k2k_state_swap import (
    SCHEMA as K2K_SCHEMA,
    _capture_ort_prestate,
    _capture_pytorch_prestate,
    _classify_top1,
    _ort_session,
)
from .mamba2_onnx import verify_mamba2_g04_bundle


SCHEMA = "VN97M2K2LLAYER1"
GROUP_SIZE = 8
TOP_GROUPS_TO_EXPAND = 3
TOP_INFLUENCERS_TO_REPORT = 10


def _gap(
    logits: np.ndarray,
    *,
    reference_token: int,
    ort_token: int,
) -> float:
    values = logits.astype(np.float32, copy=False).reshape(-1)
    return float(values[reference_token] - values[ort_token])


def _record(
    logits: np.ndarray,
    *,
    reference_token: int,
    ort_token: int,
    baseline_gap: float,
) -> dict[str, Any]:
    values = logits.astype(np.float32, copy=False).reshape(-1)
    top1 = int(values.argmax())
    gap = _gap(
        logits,
        reference_token=reference_token,
        ort_token=ort_token,
    )
    return {
        "top1": top1,
        "top1_class": _classify_top1(
            top1,
            reference_token=reference_token,
            ort_token=ort_token,
        ),
        "reference_minus_ort_logit": gap,
        "gap_influence": float(gap - baseline_gap),
        "top5_ids": _topk_ids(logits, TOP_K),
    }


def _rank_records(
    records: list[dict[str, Any]],
    *,
    limit: int = TOP_INFLUENCERS_TO_REPORT,
) -> list[dict[str, Any]]:
    ranked = sorted(
        records,
        key=lambda item: (
            -float(item["gap_influence"]),
            int(item.get("layer_start", item.get("layer", 0))),
        ),
    )
    return ranked[:limit]


def _selected_group_starts(
    group_records: list[dict[str, Any]],
    *,
    top_n: int = TOP_GROUPS_TO_EXPAND,
) -> list[int]:
    by_group: dict[int, float] = {}
    for item in group_records:
        start = int(item["layer_start"])
        influence = float(item["gap_influence"])
        by_group[start] = max(by_group.get(start, float("-inf")), influence)
    ordered = sorted(
        by_group.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return [start for start, _ in ordered[:top_n]]


def _diagnose(
    *,
    group_records: list[dict[str, Any]],
    layer_records: list[dict[str, Any]],
) -> str:
    if any(
        item["mode"] == "both"
        and item["top1_class"] == "reference"
        for item in layer_records
    ):
        return "single_layer_state_interaction_localized"
    if any(
        item["top1_class"] == "reference"
        for item in layer_records
    ):
        return "component_layer_interaction_localized"
    if any(
        item["mode"] == "both"
        and item["top1_class"] == "reference"
        for item in group_records
    ):
        return "multi_layer_state_interaction_localized"
    return "distributed_state_runtime_interaction"


def _run_logits(
    session: Any,
    *,
    current_token: int,
    conv: np.ndarray,
    ssm: np.ndarray,
) -> np.ndarray:
    logits, _, _ = session.run(
        ["logits", "next_conv_state", "next_ssm_state"],
        {
            "input_ids": np.asarray([current_token], dtype=np.int64),
            "conv_state": conv,
            "ssm_state": ssm,
        },
    )
    return np.asarray(logits).copy()


def _scan_range(
    session: Any,
    *,
    current_token: int,
    ort_conv: np.ndarray,
    ort_ssm: np.ndarray,
    py_conv: np.ndarray,
    py_ssm: np.ndarray,
    layer_start: int,
    layer_end: int,
    mode: str,
    reference_token: int,
    ort_token: int,
    baseline_gap: float,
) -> dict[str, Any]:
    if mode not in {"conv", "ssm", "both"}:
        raise ValueError(f"unsupported K2L scan mode: {mode}")

    conv = ort_conv
    ssm = ort_ssm

    if mode in {"conv", "both"}:
        conv = ort_conv.copy()
        conv[layer_start:layer_end] = py_conv[layer_start:layer_end]
    if mode in {"ssm", "both"}:
        ssm = ort_ssm.copy()
        ssm[layer_start:layer_end] = py_ssm[layer_start:layer_end]

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
            "mode": mode,
            "layer_start": layer_start,
            "layer_end_exclusive": layer_end,
            "layer_count": layer_end - layer_start,
        }
    )
    return record


def run_k2l(
    *,
    bundle_dir: Path,
    k2j_receipt_path: Path,
    k2k_receipt_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("K2L requires CUDA")
    gpu_name = torch.cuda.get_device_name(0)
    if "T4" not in gpu_name.upper():
        raise RuntimeError(f"K2L expects Kaggle T4, got {gpu_name!r}")

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
    if k2k.get("k2j_receipt_id") != k2j.get("receipt_id"):
        raise ValueError("K2L K2J/K2K lineage mismatch")
    if k2k.get("diagnosis") != "mixed_runtime_and_state":
        raise ValueError("K2L requires K2K mixed_runtime_and_state evidence")
    if k2k.get("production_graph_changed") is not False:
        raise ValueError("K2L requires unchanged G0.4 production graph")

    matrix = k2k.get("matrix")
    if not isinstance(matrix, dict):
        raise ValueError("K2L K2K matrix missing")
    if matrix["ort_oo"]["top1_class"] != "ort":
        raise ValueError("K2L requires ORT+ORT-state divergent baseline")
    for key in ("ort_pp", "ort_po", "ort_op", "py_pp", "py_po", "py_op", "py_oo"):
        if matrix[key]["top1_class"] != "reference":
            raise ValueError(
                "K2L requires only ORT+full-ORT-state to diverge"
            )

    manifest = verify_mamba2_g04_bundle(bundle_dir)
    for field in ("g04_manifest_id", "capsule_id", "source_weight_sha256"):
        manifest_field = "manifest_id" if field == "g04_manifest_id" else field
        if manifest.get(manifest_field) != k2k.get(field):
            raise ValueError(f"K2L {field} lineage mismatch")

    prompt_index = int(k2k["prompt_index"])
    divergence_step = int(k2k["divergence_step"])
    prompt = next(
        (
            item for item in k2j["prompts"]
            if int(item["prompt_index"]) == prompt_index
        ),
        None,
    )
    if prompt is None:
        raise ValueError("K2L target prompt missing from K2J")
    if int(prompt["first_divergence_step"]) != divergence_step:
        raise ValueError("K2L K2J/K2K divergence-step mismatch")

    prompt_token_ids = [int(v) for v in prompt["prompt_token_ids"]]
    reference_generated = [
        int(v) for v in prompt["reference_generated_token_ids"]
    ]
    ort_generated = [int(v) for v in prompt["ort_generated_token_ids"]]

    before_current = reference_generated[: divergence_step - 2]
    current_token = reference_generated[divergence_step - 2]
    reference_token = reference_generated[divergence_step - 1]
    ort_token = ort_generated[divergence_step - 1]

    if current_token != int(k2k["current_input_token"]):
        raise ValueError("K2L current token mismatch")
    if reference_token != int(k2k["reference_next_token"]):
        raise ValueError("K2L reference token mismatch")
    if ort_token != int(k2k["ort_next_token"]):
        raise ValueError("K2L ORT token mismatch")

    # K2K already established the exact state semantics at this boundary.
    # Recompute the two pre-states so K2L remains independently auditable.
    capsule_root = bundle_dir.parent / "g03-capsule"
    if not capsule_root.exists():
        raise ValueError(
            "K2L expected sibling G0.3 capsule at "
            f"{capsule_root}"
        )

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

    if py_conv.shape != ort_conv.shape:
        raise ValueError("K2L conv-state shape mismatch")
    if py_ssm.shape != ort_ssm.shape:
        raise ValueError("K2L SSM-state shape mismatch")
    if py_conv.shape[0] != py_ssm.shape[0]:
        raise ValueError("K2L layer count mismatch")
    layer_count = int(py_conv.shape[0])
    manifest_layers = int(
        manifest["state_contract"]["conv_state"]["shape"][0]
    )
    if layer_count != manifest_layers:
        raise ValueError("K2L state/manifest layer count mismatch")

    session, provider2 = _ort_session(bundle_dir / "step.onnx")
    if provider2 != provider:
        raise RuntimeError("K2L ORT provider changed between phases")

    baseline_logits = _run_logits(
        session,
        current_token=current_token,
        conv=ort_conv,
        ssm=ort_ssm,
    )
    baseline = _record(
        baseline_logits,
        reference_token=reference_token,
        ort_token=ort_token,
        baseline_gap=float(matrix["ort_oo"]["reference_minus_ort_logit"]),
    )
    expected_gap = float(matrix["ort_oo"]["reference_minus_ort_logit"])
    actual_gap = float(baseline["reference_minus_ort_logit"])
    if int(baseline["top1"]) != ort_token:
        raise RuntimeError("K2L baseline did not reproduce K2K ORT token")
    if abs(actual_gap - expected_gap) > 1.0e-6:
        raise RuntimeError("K2L baseline gap did not reproduce K2K")

    baseline_gap = actual_gap

    group_records: list[dict[str, Any]] = []
    for start in range(0, layer_count, GROUP_SIZE):
        end = min(start + GROUP_SIZE, layer_count)
        for mode in ("conv", "ssm", "both"):
            group_records.append(
                _scan_range(
                    session,
                    current_token=current_token,
                    ort_conv=ort_conv,
                    ort_ssm=ort_ssm,
                    py_conv=py_conv,
                    py_ssm=py_ssm,
                    layer_start=start,
                    layer_end=end,
                    mode=mode,
                    reference_token=reference_token,
                    ort_token=ort_token,
                    baseline_gap=baseline_gap,
                )
            )

    selected_starts = _selected_group_starts(group_records)

    layer_records: list[dict[str, Any]] = []
    for start in selected_starts:
        end = min(start + GROUP_SIZE, layer_count)
        for layer in range(start, end):
            for mode in ("conv", "ssm", "both"):
                record = _scan_range(
                    session,
                    current_token=current_token,
                    ort_conv=ort_conv,
                    ort_ssm=ort_ssm,
                    py_conv=py_conv,
                    py_ssm=py_ssm,
                    layer_start=layer,
                    layer_end=layer + 1,
                    mode=mode,
                    reference_token=reference_token,
                    ort_token=ort_token,
                    baseline_gap=baseline_gap,
                )
                record["layer"] = layer
                layer_records.append(record)

    del session
    gc.collect()

    top_by_mode = {
        mode: _rank_records(
            [item for item in layer_records if item["mode"] == mode]
        )
        for mode in ("conv", "ssm", "both")
    }
    group_top_by_mode = {
        mode: _rank_records(
            [item for item in group_records if item["mode"] == mode]
        )
        for mode in ("conv", "ssm", "both")
    }

    repairing_layers = [
        {
            "layer": int(item["layer"]),
            "mode": item["mode"],
            "gap_influence": float(item["gap_influence"]),
            "reference_minus_ort_logit": float(
                item["reference_minus_ort_logit"]
            ),
        }
        for item in layer_records
        if item["top1_class"] == "reference"
    ]
    repairing_groups = [
        {
            "layer_start": int(item["layer_start"]),
            "layer_end_exclusive": int(item["layer_end_exclusive"]),
            "mode": item["mode"],
            "gap_influence": float(item["gap_influence"]),
            "reference_minus_ort_logit": float(
                item["reference_minus_ort_logit"]
            ),
        }
        for item in group_records
        if item["top1_class"] == "reference"
    ]

    diagnosis = _diagnose(
        group_records=group_records,
        layer_records=layer_records,
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
        "g04_manifest_id": manifest["manifest_id"],
        "capsule_id": manifest["capsule_id"],
        "source_weight_sha256": manifest["source_weight_sha256"],
        "dtype": str(manifest["state_contract"]["dtype"]),
        "prompt_index": prompt_index,
        "divergence_step": divergence_step,
        "current_input_token": current_token,
        "reference_next_token": reference_token,
        "ort_next_token": ort_token,
        "layer_count": layer_count,
        "group_size": GROUP_SIZE,
        "selected_group_starts": selected_starts,
        "baseline": baseline,
        "group_records": group_records,
        "layer_records": layer_records,
        "group_top_by_mode": group_top_by_mode,
        "top_by_mode": top_by_mode,
        "repairing_groups": repairing_groups,
        "repairing_layers": repairing_layers,
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/vn97-k2l-layer-influence.json"),
    )
    args = parser.parse_args()

    receipt = run_k2l(
        bundle_dir=args.bundle_dir.resolve(strict=True),
        k2j_receipt_path=args.k2j_receipt.resolve(strict=True),
        k2k_receipt_path=args.k2k_receipt.resolve(strict=True),
        output_path=args.output,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
