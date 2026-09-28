from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from vn97.quantization import ternary_symbols_and_scales
from .mamba2_g03_capsule import load_g03_capsule
from .mamba2_transfer import expected_unique_parameter_count


SCHEMA = "VN97M2K2TMULTIPLANE1"
DEFAULT_PLANES = 4
DEFAULT_THRESHOLD = 0.5
TILE_ROWS = 16
TILE_COLS = 16


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_receipt(body: dict[str, Any]) -> str:
    return hashlib.sha256(
        SCHEMA.encode("ascii")
        + b"\0"
        + _canonical_json(body)
    ).hexdigest()


def _projection_keys(n_layers: int) -> list[tuple[str, str, int]]:
    result: list[tuple[str, str, int]] = []
    for layer in range(n_layers):
        prefix = f"vn97.core.layers.{layer}.mixer"
        result.append(
            (f"{prefix}.in_proj.weight", "in_proj", layer)
        )
        result.append(
            (f"{prefix}.out_proj.weight", "out_proj", layer)
        )
    return result


def _packed_plane_bytes(
    rows: int,
    cols: int,
    *,
    tile_rows: int = TILE_ROWS,
    tile_cols: int = TILE_COLS,
) -> int:
    padded_rows = (
        (rows + tile_rows - 1) // tile_rows
    ) * tile_rows
    padded_cols = (
        (cols + tile_cols - 1) // tile_cols
    ) * tile_cols
    symbols = padded_rows * padded_cols
    packed = (symbols + 3) // 4
    scales = rows * 4
    header = 40
    return header + scales + packed


def _multiplane_metrics(
    weight: torch.Tensor,
    *,
    plane_count: int,
    threshold: float,
) -> dict[str, Any]:
    if weight.ndim != 2:
        raise ValueError("K2T requires rank-2 projection weights")
    if plane_count <= 0:
        raise ValueError("plane_count must be positive")

    source = weight.detach().float()
    residual = source.clone()
    reconstruction = torch.zeros_like(source)

    source_sq = float(source.square().sum().item())
    if not math.isfinite(source_sq) or source_sq <= 0.0:
        raise ValueError("projection source norm is invalid")

    planes: list[dict[str, Any]] = []
    for plane in range(1, plane_count + 1):
        symbols, scales = ternary_symbols_and_scales(
            residual,
            threshold=threshold,
        )
        component = (
            symbols.to(dtype=source.dtype)
            * scales.unsqueeze(1)
        )
        reconstruction.add_(component)
        residual.sub_(component)

        err_sq = float(residual.square().sum().item())
        recon_sq = float(reconstruction.square().sum().item())
        dot = float((source * reconstruction).sum().item())
        rel_rmse = math.sqrt(err_sq / source_sq)
        cosine = dot / math.sqrt(
            max(source_sq * recon_sq, 1.0e-30)
        )
        max_abs_error = float(residual.abs().max().item())
        zero_fraction = float(
            (symbols == 0).sum().item() / symbols.numel()
        )
        planes.append(
            {
                "plane_count": plane,
                "error_sq": err_sq,
                "reconstruction_sq": recon_sq,
                "source_dot_reconstruction": dot,
                "relative_rmse": rel_rmse,
                "cosine_similarity": cosine,
                "max_abs_error": max_abs_error,
                "zero_fraction_last_plane": zero_fraction,
            }
        )
        del symbols, scales, component
    return {
        "source_sq": source_sq,
        "planes": planes,
    }


def _empty_aggregate(plane_count: int) -> dict[str, Any]:
    return {
        "source_sq": 0.0,
        "planes": [
            {
                "error_sq": 0.0,
                "reconstruction_sq": 0.0,
                "source_dot_reconstruction": 0.0,
                "max_abs_error": 0.0,
            }
            for _ in range(plane_count)
        ],
    }


def _accumulate(
    aggregate: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    aggregate["source_sq"] += float(metrics["source_sq"])
    for dst, src in zip(
        aggregate["planes"],
        metrics["planes"],
        strict=True,
    ):
        dst["error_sq"] += float(src["error_sq"])
        dst["reconstruction_sq"] += float(
            src["reconstruction_sq"]
        )
        dst["source_dot_reconstruction"] += float(
            src["source_dot_reconstruction"]
        )
        dst["max_abs_error"] = max(
            float(dst["max_abs_error"]),
            float(src["max_abs_error"]),
        )


def _finalize_aggregate(
    aggregate: dict[str, Any],
) -> list[dict[str, Any]]:
    source_sq = float(aggregate["source_sq"])
    if source_sq <= 0.0:
        raise ValueError("K2T aggregate source norm is empty")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(
        aggregate["planes"],
        start=1,
    ):
        err_sq = float(item["error_sq"])
        recon_sq = float(item["reconstruction_sq"])
        dot = float(item["source_dot_reconstruction"])
        result.append(
            {
                "plane_count": index,
                "relative_rmse": math.sqrt(err_sq / source_sq),
                "cosine_similarity": dot
                / math.sqrt(
                    max(source_sq * recon_sq, 1.0e-30)
                ),
                "max_abs_error": float(
                    item["max_abs_error"]
                ),
            }
        )
    return result


def run_k2t(
    *,
    capsule_root: Path,
    output_path: Path,
    plane_count: int = DEFAULT_PLANES,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("K2T requires CUDA")
    gpu_name = torch.cuda.get_device_name(0)
    if "T4" not in gpu_name.upper():
        raise RuntimeError(
            f"K2T expects Kaggle T4, got {gpu_name!r}"
        )
    if torch.__version__ != "2.10.0+cu128":
        raise RuntimeError(
            "K2T requires torch 2.10.0+cu128 to match the "
            f"locked R2 campaign, got {torch.__version__}"
        )
    if torch.version.cuda != "12.8":
        raise RuntimeError(
            f"K2T requires CUDA 12.8, got {torch.version.cuda}"
        )
    if plane_count not in {1, 2, 3, 4}:
        raise ValueError("K2T plane_count must be in 1..4")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("K2T threshold must be in [0,1]")

    capsule = load_g03_capsule(
        capsule_root,
        verify_large_weight_sha256=True,
    )
    spec = capsule.spec
    keys = _projection_keys(spec.n_layers)

    global_aggregate = _empty_aggregate(plane_count)
    type_aggregates = {
        "in_proj": _empty_aggregate(plane_count),
        "out_proj": _empty_aggregate(plane_count),
    }
    matrix_records: list[dict[str, Any]] = []

    projection_source_bytes = 0
    candidate_projection_bytes = [
        0 for _ in range(plane_count)
    ]
    projection_elements = 0

    for matrix_index, (key, kind, layer) in enumerate(keys):
        source = capsule.tensors[key]
        if source.ndim != 2:
            raise RuntimeError(
                f"K2T projection is not rank-2: {key}"
            )
        rows, cols = (
            int(source.shape[0]),
            int(source.shape[1]),
        )
        projection_elements += rows * cols
        projection_source_bytes += rows * cols * 2

        plane_blob_bytes = _packed_plane_bytes(
            rows,
            cols,
        )
        for index in range(plane_count):
            candidate_projection_bytes[index] += (
                plane_blob_bytes * (index + 1)
            )

        weight = source.to(
            device="cuda",
            dtype=torch.float32,
            non_blocking=False,
        )
        metrics = _multiplane_metrics(
            weight,
            plane_count=plane_count,
            threshold=threshold,
        )
        _accumulate(global_aggregate, metrics)
        _accumulate(type_aggregates[kind], metrics)

        matrix_records.append(
            {
                "matrix_index": matrix_index,
                "key": key,
                "layer": layer,
                "projection": kind,
                "rows": rows,
                "cols": cols,
                "elements": rows * cols,
                "source_bytes_fp16": rows * cols * 2,
                "packed_bytes_per_plane": plane_blob_bytes,
                "planes": [
                    {
                        "plane_count": item["plane_count"],
                        "relative_rmse": item["relative_rmse"],
                        "cosine_similarity":
                            item["cosine_similarity"],
                        "max_abs_error":
                            item["max_abs_error"],
                        "zero_fraction_last_plane":
                            item["zero_fraction_last_plane"],
                    }
                    for item in metrics["planes"]
                ],
            }
        )
        del weight, metrics
        torch.cuda.empty_cache()
        gc.collect()

    source_total_bytes = (
        expected_unique_parameter_count(spec) * 2
    )
    retained_exact_bytes = (
        source_total_bytes - projection_source_bytes
    )
    if retained_exact_bytes <= 0:
        raise RuntimeError(
            "K2T retained exact byte accounting is invalid"
        )

    global_planes = _finalize_aggregate(
        global_aggregate
    )
    type_planes = {
        key: _finalize_aggregate(value)
        for key, value in type_aggregates.items()
    }

    candidate_sizes: list[dict[str, Any]] = []
    for index in range(plane_count):
        packed = candidate_projection_bytes[index]
        total = retained_exact_bytes + packed
        candidate_sizes.append(
            {
                "plane_count": index + 1,
                "projection_bytes": packed,
                "retained_exact_bytes": retained_exact_bytes,
                "estimated_total_bytes": total,
                "estimated_total_gib": total / (1024**3),
                "compression_ratio_vs_fp16_unique":
                    source_total_bytes / total,
                "weight_bits_per_projection_symbol":
                    2 * (index + 1),
            }
        )

    worst_by_plane: dict[str, list[dict[str, Any]]] = {}
    for plane in range(1, plane_count + 1):
        ordered = sorted(
            matrix_records,
            key=lambda item: float(
                item["planes"][plane - 1][
                    "relative_rmse"
                ]
            ),
            reverse=True,
        )
        worst_by_plane[str(plane)] = [
            {
                "key": item["key"],
                "layer": item["layer"],
                "projection": item["projection"],
                "relative_rmse":
                    item["planes"][plane - 1][
                        "relative_rmse"
                    ],
                "cosine_similarity":
                    item["planes"][plane - 1][
                        "cosine_similarity"
                    ],
            }
            for item in ordered[:12]
        ]

    body: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "MEASURED",
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "capsule_id": capsule.manifest.capsule_id(),
        "source_weight_sha256":
            capsule.manifest.source_weight_sha256,
        "source_unique_parameter_count":
            expected_unique_parameter_count(spec),
        "source_unique_fp16_bytes": source_total_bytes,
        "projection_matrix_count": len(keys),
        "projection_elements": projection_elements,
        "projection_source_fp16_bytes":
            projection_source_bytes,
        "threshold": threshold,
        "tile_rows": TILE_ROWS,
        "tile_cols": TILE_COLS,
        "max_plane_count": plane_count,
        "global_planes": global_planes,
        "projection_type_planes": type_planes,
        "candidate_sizes": candidate_sizes,
        "worst_matrices_by_plane": worst_by_plane,
        "matrix_records": matrix_records,
        "quantization_semantics":
            "iterative_canonical_vn97t2_per_row_ternary_residual_planes",
        "diagnostic_only": True,
        "behavior_gate_required": True,
        "mobile_kernel_required": True,
        "source_lineage_unchanged": True,
        "weights_changed": True,
        "architecture_changed": False,
        "production_graph_changed": False,
        "production_activation_authorized": False,
        "acceptance_threshold_defined": False,
    }
    body["receipt_id"] = _sha256_receipt(body)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_bytes(
        _canonical_json(body) + b"\n"
    )
    return body


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit multi-plane VN97T2 residual ternary "
            "projection lowering for Mamba-2 2.7B."
        )
    )
    parser.add_argument(
        "--capsule-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--planes",
        type=int,
        default=DEFAULT_PLANES,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
    )
    args = parser.parse_args()

    result = run_k2t(
        capsule_root=args.capsule_root,
        output_path=args.output,
        plane_count=args.planes,
        threshold=args.threshold,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "receipt_id": result["receipt_id"],
                "capsule_id": result["capsule_id"],
                "projection_matrix_count":
                    result["projection_matrix_count"],
                "global_planes": result["global_planes"],
                "candidate_sizes": result["candidate_sizes"],
                "production_activation_authorized":
                    result[
                        "production_activation_authorized"
                    ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
