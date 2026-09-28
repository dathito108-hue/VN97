from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .mamba2_kaggle_k2a_ort import _canonical_json, _load_json
from .mamba2_kaggle_k2d_stage_trace import (
    SCHEMA as K2D_SCHEMA,
    run_k2d,
)


SCHEMA = "VN97M2K2EDBX1"


def _verify_k2d_receipt(path: Path) -> dict[str, Any]:
    receipt = _load_json(path, "K2D receipt")
    if receipt.get("schema") != K2D_SCHEMA:
        raise ValueError("K2E requires K2D stage-trace receipt")
    if receipt.get("status") != "MEASURED":
        raise ValueError("K2E requires K2D status MEASURED")
    if receipt.get("acceptance_threshold_defined") is not False:
        raise ValueError("K2D must remain diagnostic-only")
    if receipt.get("production_activation_authorized") is not False:
        raise ValueError("K2D must not authorize production")

    receipt_id = receipt.get("receipt_id")
    if not isinstance(receipt_id, str) or len(receipt_id) != 64:
        raise ValueError("K2E K2D receipt identity invalid")
    body = dict(receipt)
    body.pop("receipt_id", None)
    expected = hashlib.sha256(
        K2D_SCHEMA.encode("ascii") + b"\0" + _canonical_json(body)
    ).hexdigest()
    if receipt_id != expected:
        raise ValueError("K2E K2D receipt identity mismatch")

    first = receipt.get("first_stage_over_reference")
    if not isinstance(first, dict):
        raise ValueError("K2E requires a concrete K2D divergent stage")
    if first.get("stage") != "d_b_x":
        raise ValueError("K2E is only valid for K2D d_b_x divergence")
    return receipt


def run_k2e(
    *,
    capsule_root: Path,
    bundle_dir: Path,
    k2c_receipt_path: Path,
    k2d_receipt_path: Path,
    trace_output_dir: Path,
    post_trace_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    before = _verify_k2d_receipt(k2d_receipt_path)

    post = run_k2d(
        capsule_root=capsule_root,
        bundle_dir=bundle_dir,
        k2c_receipt_path=k2c_receipt_path,
        output_dir=trace_output_dir,
        output_path=post_trace_path,
    )

    for field in (
        "k2c_receipt_id",
        "g04_manifest_id",
        "capsule_id",
        "source_weight_sha256",
        "trace_layer",
        "probe_token_id",
        "provider",
        "dtype",
        "fp16_epsilon",
        "diagnostic_reference_max_epsilon_units",
    ):
        if post.get(field) != before.get(field):
            raise ValueError(f"K2E post-repair K2D lineage mismatch: {field}")

    before_metrics = before["stage_metrics"]["d_b_x"]
    after_metrics = post["stage_metrics"]["d_b_x"]
    reference = float(before["diagnostic_reference_max_epsilon_units"])
    before_units = float(before_metrics["max_epsilon_units"])
    after_units = float(after_metrics["max_epsilon_units"])

    if before_units <= reference:
        raise ValueError("K2E requires pre-repair d_b_x above reference")

    reduction = (
        before_units / after_units
        if after_units > 0.0
        else float("inf")
    )

    body: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "MEASURED",
        "k2d_before_receipt_id": before["receipt_id"],
        "k2d_after_receipt_id": post["receipt_id"],
        "k2c_receipt_id": before["k2c_receipt_id"],
        "g04_manifest_id": before["g04_manifest_id"],
        "capsule_id": before["capsule_id"],
        "source_weight_sha256": before["source_weight_sha256"],
        "trace_layer": before["trace_layer"],
        "probe_token_id": before["probe_token_id"],
        "provider": before["provider"],
        "repair": "explicit_left_associated_broadcast_dBx",
        "operation_order": "(dt_value*x_heads)*b_value",
        "activation_dtype_preserved": True,
        "weights_changed": False,
        "architecture_changed": False,
        "before_d_b_x": before_metrics,
        "after_d_b_x": after_metrics,
        "before_first_stage_over_reference":
            before["first_stage_over_reference"],
        "after_first_stage_over_reference":
            post["first_stage_over_reference"],
        "diagnostic_reference_max_epsilon_units": reference,
        "d_b_x_within_reference_after": after_units <= reference,
        "d_b_x_epsilon_reduction_factor": (
            reduction if reduction != float("inf") else None
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
    parser.add_argument("--k2c-receipt", required=True, type=Path)
    parser.add_argument("--k2d-receipt", required=True, type=Path)
    parser.add_argument("--trace-output-dir", required=True, type=Path)
    parser.add_argument("--post-trace-output", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/vn97-k2e-dbx-repair.json"),
    )
    args = parser.parse_args()

    receipt = run_k2e(
        capsule_root=args.capsule_root.resolve(strict=True),
        bundle_dir=args.bundle_dir.resolve(strict=True),
        k2c_receipt_path=args.k2c_receipt.resolve(strict=True),
        k2d_receipt_path=args.k2d_receipt.resolve(strict=True),
        trace_output_dir=args.trace_output_dir,
        post_trace_path=args.post_trace_output,
        output_path=args.output,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
