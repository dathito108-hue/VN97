from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import torch

from .model import VN97LanguageCore
from .p4_task_evaluation import P4_CATEGORIES, load_p4_task_suite
from .p5e3_heldout_validation_cli import _load_artifact
from .p5e3c_target_rank_diagnostic_cli import _score_target, _targets_for_task
from .quantization import set_float_shadow_mode
from .training_cli import _atomic_write


P5E4A_SCHEMA = "VN97P5E4A"
P5E4A_FLOAT_SCHEMA = "VN97P5E4AFLOAT1"
P5E4A_PROFILE_ID = "vn97-p5e4a-selective-delta-reblend-search-v1"

_LAYER_RE = re.compile(r"^layers\.(\d+)\.")

_DYNAMICS_SUFFIXES = (
    ".core.a_log",
    ".core.dt_proj.weight",
    ".core.dt_proj.bias",
    ".core.b_proj.weight",
    ".core.c_proj.weight",
)

_IO_SUFFIXES = (
    ".core.in_proj.weight",
    ".core.out_proj.weight",
)

# The original P5E1 graft used alpha=0.35 and hurt held-out latent signal.
# Search only conservative projections of that known delta. No teacher model
# or teacher generation is required for this repair.
_SEARCH_PROFILES: tuple[dict[str, object], ...] = (
    {"id": "dyn_all_a002", "group": "dynamics", "start_layer": 0, "alpha": 0.02},
    {"id": "dyn_all_a004", "group": "dynamics", "start_layer": 0, "alpha": 0.04},
    {"id": "dyn_all_a006", "group": "dynamics", "start_layer": 0, "alpha": 0.06},
    {"id": "dyn_all_a008", "group": "dynamics", "start_layer": 0, "alpha": 0.08},
    {"id": "dyn_tail16_a004", "group": "dynamics", "start_layer": 16, "alpha": 0.04},
    {"id": "dyn_tail16_a008", "group": "dynamics", "start_layer": 16, "alpha": 0.08},
    {"id": "dyn_tail16_a012", "group": "dynamics", "start_layer": 16, "alpha": 0.12},
    {"id": "dyn_tail8_a008", "group": "dynamics", "start_layer": 24, "alpha": 0.08},
    {"id": "dyn_tail8_a012", "group": "dynamics", "start_layer": 24, "alpha": 0.12},
    {"id": "dyn_tail8_a016", "group": "dynamics", "start_layer": 24, "alpha": 0.16},
    {"id": "dynio_tail8_a004", "group": "dynamics_io", "start_layer": 24, "alpha": 0.04},
    {"id": "dynio_tail8_a008", "group": "dynamics_io", "start_layer": 24, "alpha": 0.08},
)


class VN97P5E4AError(RuntimeError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search conservative selective reblends of the already-produced "
            "P5E1 direct SSM delta. The teacher is not loaded. Embedding, "
            "norms and unselected layers remain exactly at the P5D3B base."
        )
    )
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--p5e1-dir", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _parameter_selected(
    name: str,
    *,
    group: str,
    start_layer: int,
) -> bool:
    match = _LAYER_RE.match(name)
    if match is None:
        return False
    layer_index = int(match.group(1))
    if layer_index < start_layer:
        return False
    if any(name.endswith(suffix) for suffix in _DYNAMICS_SUFFIXES):
        return True
    if group == "dynamics_io" and any(
        name.endswith(suffix)
        for suffix in _IO_SUFFIXES
    ):
        return True
    return False


def _build_candidate(
    *,
    base_model: VN97LanguageCore,
    p5e1_model: VN97LanguageCore,
    source_blend: float,
    profile: dict[str, object],
) -> tuple[VN97LanguageCore, int, int]:
    if not 0.0 < source_blend <= 1.0:
        raise VN97P5E4AError("invalid P5E1 source blend")

    alpha = float(profile["alpha"])
    if not 0.0 < alpha <= source_blend:
        raise VN97P5E4AError("candidate alpha must be within P5E1 source blend")
    group = str(profile["group"])
    start_layer = int(profile["start_layer"])

    candidate = VN97LanguageCore(base_model.config)
    set_float_shadow_mode(candidate, True)
    candidate.load_state_dict(base_model.state_dict())

    base_state = base_model.state_dict()
    source_state = p5e1_model.state_dict()
    candidate_state = candidate.state_dict()
    selected_tensors = 0
    selected_parameters = 0
    ratio = alpha / source_blend

    with torch.no_grad():
        for name, base_tensor in base_state.items():
            if not _parameter_selected(
                name,
                group=group,
                start_layer=start_layer,
            ):
                continue
            if name not in source_state or name not in candidate_state:
                raise VN97P5E4AError(f"missing reblend tensor: {name}")
            source_tensor = source_state[name]
            target_tensor = candidate_state[name]
            if (
                tuple(base_tensor.shape) != tuple(source_tensor.shape)
                or tuple(base_tensor.shape) != tuple(target_tensor.shape)
            ):
                raise VN97P5E4AError(f"reblend shape mismatch: {name}")

            # P5E1 layer tensors were produced by:
            # base + source_blend * (mapped_teacher - base)
            # Therefore scaling the observed delta by alpha/source_blend
            # exactly reconstructs a lower-alpha graft for these tensors.
            value = (
                base_tensor.detach().float()
                + (
                    source_tensor.detach().float()
                    - base_tensor.detach().float()
                )
                * ratio
            )
            target_tensor.copy_(value.to(dtype=target_tensor.dtype))
            selected_tensors += 1
            selected_parameters += int(target_tensor.numel())

    if selected_tensors == 0:
        raise VN97P5E4AError("candidate profile selected no tensors")
    return candidate, selected_tensors, selected_parameters


@torch.inference_mode()
def _score_model(
    *,
    model: VN97LanguageCore,
    tokenizer,
    suite,
    device: str,
) -> dict[str, Any]:
    model.to(device)
    model.eval()
    set_float_shadow_mode(model, True)

    total_tokens = 0
    total_nll = 0.0
    total_top5 = 0
    first_rank_sum = 0
    first_targets = 0
    categories: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "targets": 0,
            "tokens": 0,
            "nll_sum": 0.0,
            "top5_tokens": 0,
            "first_rank_sum": 0,
        }
    )

    for task in suite.tasks:
        for target_text in _targets_for_task(task):
            metrics = _score_target(
                model=model,
                tokenizer=tokenizer,
                prompt=task.prompt,
                target_text=target_text,
                device=device,
            )
            tokens = int(metrics["token_count"])
            nll = float(metrics["nll_sum"])
            top5 = int(metrics["top5_tokens"])
            first_rank = int(metrics["first_rank"])

            total_tokens += tokens
            total_nll += nll
            total_top5 += top5
            first_rank_sum += first_rank
            first_targets += 1

            row = categories[task.category]
            row["targets"] = int(row["targets"]) + 1
            row["tokens"] = int(row["tokens"]) + tokens
            row["nll_sum"] = float(row["nll_sum"]) + nll
            row["top5_tokens"] = int(row["top5_tokens"]) + top5
            row["first_rank_sum"] = int(row["first_rank_sum"]) + first_rank

    if total_tokens <= 0 or first_targets <= 0:
        raise VN97P5E4AError("reblend diagnostic produced no target signal")

    normalized: dict[str, dict[str, float | int]] = {}
    for category in P4_CATEGORIES:
        row = categories.get(category)
        if row is None:
            normalized[category] = {
                "targets": 0,
                "tokens": 0,
                "mean_nll": 0.0,
                "token_top5_rate": 0.0,
                "mean_first_rank": 0.0,
            }
            continue
        tokens = int(row["tokens"])
        targets = int(row["targets"])
        normalized[category] = {
            "targets": targets,
            "tokens": tokens,
            "mean_nll": float(row["nll_sum"]) / max(tokens, 1),
            "token_top5_rate": int(row["top5_tokens"]) / max(tokens, 1),
            "mean_first_rank": int(row["first_rank_sum"]) / max(targets, 1),
        }

    model.cpu()
    torch.cuda.empty_cache()

    return {
        "categories": normalized,
        "mean_first_rank": first_rank_sum / first_targets,
        "mean_nll": total_nll / total_tokens,
        "target_tokens": total_tokens,
        "token_top5_rate": total_top5 / total_tokens,
    }


def _safe_against_base(
    *,
    base: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if float(candidate["mean_nll"]) > float(base["mean_nll"]) * 1.01:
        reasons.append("nll_over_1pct")
    if (
        float(candidate["token_top5_rate"])
        + 0.005
        < float(base["token_top5_rate"])
    ):
        reasons.append("top5_drop_over_0.5pp")
    if (
        float(candidate["mean_first_rank"])
        > float(base["mean_first_rank"]) * 1.05
    ):
        reasons.append("first_rank_over_5pct")

    for category in ("tool_intent", "authority_behavior"):
        base_cat = base["categories"][category]
        cand_cat = candidate["categories"][category]
        if (
            int(base_cat["tokens"]) > 0
            and float(cand_cat["mean_nll"])
            > float(base_cat["mean_nll"]) * 1.03
        ):
            reasons.append(f"{category}_nll_over_3pct")
    return not reasons, reasons


def _utility(
    *,
    base: dict[str, Any],
    candidate: dict[str, Any],
) -> float:
    base_nll = float(base["mean_nll"])
    base_top5 = float(base["token_top5_rate"])
    base_rank = float(base["mean_first_rank"])

    nll_gain = (
        base_nll - float(candidate["mean_nll"])
    ) / max(base_nll, 1e-12)
    top5_gain = float(candidate["token_top5_rate"]) - base_top5
    rank_gain = (
        base_rank - float(candidate["mean_first_rank"])
    ) / max(base_rank, 1e-12)
    return nll_gain + 0.5 * top5_gain + 0.2 * rank_gain


def _meaningful_gain(
    *,
    base: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    return (
        float(candidate["mean_nll"]) <= float(base["mean_nll"]) * 0.99
        or float(candidate["token_top5_rate"])
        >= float(base["token_top5_rate"]) + 0.01
        or float(candidate["mean_first_rank"])
        <= float(base["mean_first_rank"]) * 0.95
    )


def _write_torch_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.unlink(missing_ok=True)
    torch.save(payload, temp)
    temp.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise VN97P5E4AError("CUDA device requested but CUDA is unavailable")

    suite = load_p4_task_suite(Path(args.suite).resolve(strict=True))
    base_model, tokenizer, base_meta = _load_artifact(
        Path(args.base_dir),
        label="base",
    )
    p5e1_model, p5e1_tokenizer, p5e1_meta = _load_artifact(
        Path(args.p5e1_dir),
        label="p5e1",
    )
    del p5e1_tokenizer

    if base_meta["tokenizer_sha256"] != p5e1_meta["tokenizer_sha256"]:
        raise VN97P5E4AError("base and P5E1 tokenizer mismatch")
    if base_model.config != p5e1_model.config:
        raise VN97P5E4AError("base and P5E1 model config mismatch")

    p5e1_report_path = Path(args.p5e1_dir) / "p5e1-report.json"
    report = json.loads(p5e1_report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise VN97P5E4AError("P5E1 report must be a JSON object")
    source_blend = float(report.get("blend", 0.0))
    if abs(source_blend - 0.35) > 1e-9:
        raise VN97P5E4AError(
            "P5E4A expects the locked P5E1 source blend of 0.35"
        )

    print(
        "VN97 P5E4A BASELINE start "
        f"profiles={len(_SEARCH_PROFILES)} "
        f"source_blend={source_blend:.3f} "
        f"teacher_loaded=false",
        flush=True,
    )
    base_metrics = _score_model(
        model=base_model,
        tokenizer=tokenizer,
        suite=suite,
        device=args.device,
    )
    print(
        "VN97 P5E4A BASELINE "
        f"nll={base_metrics['mean_nll']:.6f} "
        f"top5={base_metrics['token_top5_rate']:.6f} "
        f"first_rank={base_metrics['mean_first_rank']:.2f}",
        flush=True,
    )

    candidate_rows: list[dict[str, Any]] = []
    best_profile: dict[str, object] | None = None
    best_metrics: dict[str, Any] | None = None
    best_utility = float("-inf")

    for profile in _SEARCH_PROFILES:
        candidate, selected_tensors, selected_parameters = _build_candidate(
            base_model=base_model,
            p5e1_model=p5e1_model,
            source_blend=source_blend,
            profile=profile,
        )
        metrics = _score_model(
            model=candidate,
            tokenizer=tokenizer,
            suite=suite,
            device=args.device,
        )
        safe, reasons = _safe_against_base(
            base=base_metrics,
            candidate=metrics,
        )
        utility = _utility(
            base=base_metrics,
            candidate=metrics,
        )
        row = {
            "alpha": float(profile["alpha"]),
            "group": str(profile["group"]),
            "id": str(profile["id"]),
            "metrics": metrics,
            "safe": safe,
            "safety_reasons": reasons,
            "selected_parameters": selected_parameters,
            "selected_tensors": selected_tensors,
            "start_layer": int(profile["start_layer"]),
            "utility": utility,
        }
        candidate_rows.append(row)

        print(
            "VN97 P5E4A CANDIDATE "
            f"id={row['id']} "
            f"safe={str(safe).lower()} "
            f"nll={metrics['mean_nll']:.6f} "
            f"top5={metrics['token_top5_rate']:.6f} "
            f"first_rank={metrics['mean_first_rank']:.2f} "
            f"utility={utility:.6f} "
            f"reasons={','.join(reasons) if reasons else 'none'}",
            flush=True,
        )

        if safe and utility > best_utility:
            best_profile = profile
            best_metrics = metrics
            best_utility = utility
        del candidate
        torch.cuda.empty_cache()

    output = Path(args.output_dir)
    if output.exists() and (
        output.is_symlink()
        or not output.is_dir()
        or any(output.iterdir())
    ):
        raise VN97P5E4AError("output-dir must be new or empty")
    output.mkdir(parents=True, exist_ok=True)

    if best_profile is None or best_metrics is None:
        status = "NO_SAFE_REBLEND"
        selected_id = None
        meaningful_gain = False
    else:
        meaningful_gain = _meaningful_gain(
            base=base_metrics,
            candidate=best_metrics,
        )
        status = (
            "SAFE_REBLEND_GAIN"
            if meaningful_gain
            else "SAFE_REBLEND_PRESERVED"
        )
        selected_id = str(best_profile["id"])

        selected_model, _, _ = _build_candidate(
            base_model=base_model,
            p5e1_model=p5e1_model,
            source_blend=source_blend,
            profile=best_profile,
        )
        tokenizer_bytes = (Path(args.base_dir) / "tokenizer.vn97tk1").read_bytes()
        payload = {
            "base_student_sha256": base_meta["student_sha256"],
            "config": asdict(selected_model.config),
            "diagnostic_suite_reused": True,
            "float_shadow_required": True,
            "fresh_holdout_required": True,
            "p5e1_student_sha256": p5e1_meta["student_sha256"],
            "profile": best_profile,
            "profile_id": P5E4A_PROFILE_ID,
            "schema": P5E4A_FLOAT_SCHEMA,
            "source_blend": source_blend,
            "state_dict": selected_model.state_dict(),
            "status": status,
            "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
        }
        _write_torch_atomic(
            output / "student-float.pt",
            payload,
        )
        _atomic_write(
            output / "tokenizer.vn97tk1",
            tokenizer_bytes,
        )
        del selected_model

    final_report = {
        "base": {
            "artifact": base_meta,
            "metrics": base_metrics,
        },
        "candidates": candidate_rows,
        "diagnostic_suite_reused": True,
        "fresh_holdout_required": True,
        "generation_gate_passed": False,
        "meaningful_gain": meaningful_gain,
        "p5e1": {
            "artifact": p5e1_meta,
            "source_blend": source_blend,
        },
        "profile_id": P5E4A_PROFILE_ID,
        "schema": P5E4A_SCHEMA,
        "selected_id": selected_id,
        "selected_metrics": best_metrics,
        "status": status,
        "suite_sha256": suite.suite_sha256,
        "teacher_loaded": False,
    }
    _atomic_write(
        output / "p5e4a-report.json",
        (
            json.dumps(
                final_report,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        ),
    )

    if (output / "student-float.pt").is_file():
        sums = {
            "p5e4a-report.json": _sha256(output / "p5e4a-report.json"),
            "student-float.pt": _sha256(output / "student-float.pt"),
            "tokenizer.vn97tk1": _sha256(output / "tokenizer.vn97tk1"),
        }
        _atomic_write(
            output / "SHA256SUMS",
            "".join(
                f"{digest}  {name}\n"
                for name, digest in sorted(sums.items())
            ).encode("ascii"),
        )

    print(
        "VN97P5E4A "
        f"status={status} "
        f"selected={selected_id or 'none'} "
        f"base_nll={base_metrics['mean_nll']:.6f} "
        f"selected_nll="
        f"{best_metrics['mean_nll']:.6f}"
        if best_metrics is not None
        else (
            "VN97P5E4A "
            f"status={status} selected=none "
            f"base_nll={base_metrics['mean_nll']:.6f}"
        ),
        flush=True,
    )
    print(f"P5E4A report: {output / 'p5e4a-report.json'}", flush=True)
    if best_profile is not None:
        print(f"P5E4A candidate: {output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-p5e4a-reblend: {exc}", file=sys.stderr)
        raise
