from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

from .chat_boundary import recover_chat_response_text
from .cognition_adapter import TorchVN97InferenceEngine, VN97InferenceLimits
from .p4_task_evaluation import P4_CATEGORIES
from .p5e3c_target_rank_diagnostic_cli import _score_target
from .p5e4b_fresh_validation_cli import _build_probes
from .p5e4d_fresh_decoder_validation_cli import _load_repaired
from .p5e4e_output_expansion_cli import ADAPTER_RANK, ResidualOutputAdapter
from .p5e4f_fresh_output_validation_cli import (
    VN97OutputExpandedModel,
    _load_expansion,
)
from .p5e4g_prefix_decoding_repair_cli import (
    P5E4G_ADAPTER_SCHEMA,
    P5E4G_SCHEMA,
)
from .training import VN97ChatMessage
from .training_cli import _atomic_write


P5E4H_SCHEMA = "VN97P5E4H"
P5E4H_PROFILE_ID = "vn97-p5e4h-fresh-post-prefix-validation-v1"
PER_CATEGORY = 16
QAT_MIN_GENERATION_RATE = 0.10


class VN97P5E4HError(RuntimeError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fresh one-shot validation after P5E4G prefix decoding repair. "
            "Compares the locked P5E4E parent output adapter against the P5E4G "
            "prefix-repaired adapter on a new deterministic 96-task manifest."
        )
    )
    parser.add_argument("--repaired-dir", required=True)
    parser.add_argument("--expanded-dir", required=True)
    parser.add_argument("--prefix-repaired-dir", required=True)
    parser.add_argument("--output", required=True)
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


def _load_prefix_repaired(
    root: Path,
    *,
    repaired_meta: dict[str, Any],
    parent_meta: dict[str, Any],
    d_model: int,
    vocab_size: int,
) -> tuple[ResidualOutputAdapter, dict[str, Any]]:
    resolved = root.resolve(strict=True)
    report_path = resolved / "p5e4g-report.json"
    adapter_path = resolved / "output-adapter.pt"
    tokenizer_path = resolved / "tokenizer.vn97tk1"
    for path in (report_path, adapter_path, tokenizer_path):
        if not path.is_file():
            raise VN97P5E4HError(
                f"P5E4G artifact missing {path.name}"
            )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict)
        or report.get("schema") != P5E4G_SCHEMA
        or report.get("status") != "PREFIX_DECODING_REPAIR_SIGNAL"
        or report.get("ready_for_fresh_validation") is not True
        or report.get("ready_for_qat") is not False
    ):
        raise VN97P5E4HError("P5E4G report contract mismatch")
    if (
        report.get("repaired_student_sha256")
        != repaired_meta.get("student_sha256")
    ):
        raise VN97P5E4HError("P5E4G/P5E4C checkpoint identity mismatch")

    payload = torch.load(
        adapter_path,
        map_location="cpu",
        weights_only=False,
    )
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != P5E4G_ADAPTER_SCHEMA
        or payload.get("status") != "PREFIX_DECODING_REPAIR_SIGNAL"
        or payload.get("ready_for_fresh_validation") is not True
        or payload.get("ready_for_qat") is not False
    ):
        raise VN97P5E4HError("P5E4G adapter contract mismatch")
    if int(payload.get("rank", -1)) != ADAPTER_RANK:
        raise VN97P5E4HError("P5E4G adapter rank mismatch")
    if int(payload.get("d_model", -1)) != d_model:
        raise VN97P5E4HError("P5E4G adapter d_model mismatch")
    if int(payload.get("vocab_size", -1)) != vocab_size:
        raise VN97P5E4HError("P5E4G adapter vocabulary mismatch")
    if (
        payload.get("repaired_student_sha256")
        != repaired_meta.get("student_sha256")
    ):
        raise VN97P5E4HError("P5E4G adapter parent checkpoint mismatch")
    if (
        payload.get("parent_adapter_sha256")
        != parent_meta.get("adapter_sha256")
    ):
        raise VN97P5E4HError("P5E4G parent adapter identity mismatch")
    if (
        payload.get("tokenizer_sha256")
        != repaired_meta.get("tokenizer_sha256")
    ):
        raise VN97P5E4HError("P5E4G tokenizer identity mismatch")
    tokenizer_sha = _sha256(tokenizer_path)
    if tokenizer_sha != repaired_meta.get("tokenizer_sha256"):
        raise VN97P5E4HError("P5E4G copied tokenizer hash mismatch")

    state = payload.get("adapter_state_dict")
    if not isinstance(state, dict):
        raise VN97P5E4HError("P5E4G adapter state missing")
    adapter = ResidualOutputAdapter(
        d_model=d_model,
        vocab_size=vocab_size,
        rank=ADAPTER_RANK,
    )
    adapter.load_state_dict(state)

    return adapter, {
        "root": str(resolved),
        "adapter_sha256": _sha256(adapter_path),
        "report_sha256": _sha256(report_path),
        "tokenizer_sha256": tokenizer_sha,
        "repaired_student_sha256": payload.get(
            "repaired_student_sha256"
        ),
        "parent_adapter_sha256": payload.get(
            "parent_adapter_sha256"
        ),
        "frozen_core_sha256": payload.get("frozen_core_sha256"),
        "status": payload.get("status"),
        "rank": int(payload["rank"]),
    }


def _fresh_seed(
    repaired_sha: str,
    parent_adapter_sha: str,
    repaired_adapter_sha: str,
) -> str:
    return hashlib.sha256(
        (
            P5E4H_PROFILE_ID
            + "\0"
            + repaired_sha
            + "\0"
            + parent_adapter_sha
            + "\0"
            + repaired_adapter_sha
        ).encode("utf-8")
    ).hexdigest()


@torch.inference_mode()
def _evaluate(
    *,
    label: str,
    model,
    tokenizer,
    probes,
    device: str,
) -> dict[str, Any]:
    model.to(device)
    model.eval()

    engine = TorchVN97InferenceEngine(
        model,
        tokenizer,
        limits=VN97InferenceLimits(
            max_prompt_tokens=4096,
            repetition_penalty=1.12,
            no_repeat_ngram_size=4,
        ),
    )

    totals = {
        "tokens": 0,
        "nll_sum": 0.0,
        "top1_tokens": 0,
        "top5_tokens": 0,
        "first_rank_sum": 0,
        "first_top1": 0,
        "generation_passed": 0,
        "generation_errors": 0,
    }
    categories: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "tasks": 0,
            "tokens": 0,
            "nll_sum": 0.0,
            "top1_tokens": 0,
            "top5_tokens": 0,
            "first_rank_sum": 0,
            "first_top1": 0,
            "generation_passed": 0,
            "generation_errors": 0,
        }
    )

    for index, probe in enumerate(probes, start=1):
        latent = _score_target(
            model=model,
            tokenizer=tokenizer,
            prompt=probe.prompt,
            target_text=probe.target,
            device=device,
        )

        error_type = ""
        try:
            raw = engine.generate_chat_completion(
                (
                    VN97ChatMessage(
                        role="user",
                        content=probe.prompt,
                    ),
                ),
                max_new_tokens=probe.max_new_tokens,
            )
            recovered = recover_chat_response_text(raw).strip()
            generation_ok = recovered == probe.target
        except Exception as exc:
            generation_ok = False
            error_type = type(exc).__name__

        tokens = int(latent["token_count"])
        nll = float(latent["nll_sum"])
        top1 = int(latent["top1_tokens"])
        top5 = int(latent["top5_tokens"])
        first_rank = int(latent["first_rank"])
        first_top1 = int(first_rank == 1)

        totals["tokens"] += tokens
        totals["nll_sum"] += nll
        totals["top1_tokens"] += top1
        totals["top5_tokens"] += top5
        totals["first_rank_sum"] += first_rank
        totals["first_top1"] += first_top1
        totals["generation_passed"] += int(generation_ok)
        totals["generation_errors"] += int(bool(error_type))

        row = categories[probe.category]
        row["tasks"] = int(row["tasks"]) + 1
        row["tokens"] = int(row["tokens"]) + tokens
        row["nll_sum"] = float(row["nll_sum"]) + nll
        row["top1_tokens"] = int(row["top1_tokens"]) + top1
        row["top5_tokens"] = int(row["top5_tokens"]) + top5
        row["first_rank_sum"] = int(row["first_rank_sum"]) + first_rank
        row["first_top1"] = int(row["first_top1"]) + first_top1
        row["generation_passed"] = int(
            row["generation_passed"]
        ) + int(generation_ok)
        row["generation_errors"] = int(
            row["generation_errors"]
        ) + int(bool(error_type))

        if index % 16 == 0 or index == len(probes):
            print(
                "VN97 P5E4H PROGRESS "
                f"model={label} tasks={index}/{len(probes)} "
                f"gen_passed={totals['generation_passed']} "
                f"mean_nll="
                f"{float(totals['nll_sum']) / max(int(totals['tokens']), 1):.6f} "
                f"top1="
                f"{int(totals['top1_tokens']) / max(int(totals['tokens']), 1):.6f} "
                f"first_top1="
                f"{int(totals['first_top1']) / index:.6f}",
                flush=True,
            )

    if int(totals["tokens"]) <= 0:
        raise VN97P5E4HError("fresh validation produced no targets")

    normalized: dict[str, dict[str, float | int]] = {}
    for category in P4_CATEGORIES:
        row = categories[category]
        tasks = int(row["tasks"])
        tokens = int(row["tokens"])
        normalized[category] = {
            "tasks": tasks,
            "tokens": tokens,
            "mean_nll": float(row["nll_sum"]) / max(tokens, 1),
            "token_top1_rate": int(row["top1_tokens"]) / max(tokens, 1),
            "token_top5_rate": int(row["top5_tokens"]) / max(tokens, 1),
            "mean_first_rank": int(row["first_rank_sum"]) / max(tasks, 1),
            "first_top1_rate": int(row["first_top1"]) / max(tasks, 1),
            "generation_passed": int(row["generation_passed"]),
            "generation_pass_rate": int(
                row["generation_passed"]
            ) / max(tasks, 1),
            "generation_errors": int(row["generation_errors"]),
        }

    result = {
        "categories": normalized,
        "generation_errors": int(totals["generation_errors"]),
        "generation_passed": int(totals["generation_passed"]),
        "generation_pass_rate": int(
            totals["generation_passed"]
        ) / len(probes),
        "mean_first_rank": int(totals["first_rank_sum"]) / len(probes),
        "first_top1_rate": int(totals["first_top1"]) / len(probes),
        "mean_nll": float(totals["nll_sum"]) / int(totals["tokens"]),
        "target_tokens": int(totals["tokens"]),
        "tasks": len(probes),
        "token_top1_rate": int(
            totals["top1_tokens"]
        ) / int(totals["tokens"]),
        "token_top5_rate": int(
            totals["top5_tokens"]
        ) / int(totals["tokens"]),
    }

    model.cpu()
    torch.cuda.empty_cache()
    return result


def _gate(
    *,
    parent: dict[str, Any],
    repaired: dict[str, Any],
) -> tuple[str, list[str], bool]:
    reasons: list[str] = []

    if float(repaired["mean_nll"]) > float(parent["mean_nll"]) * 1.015:
        reasons.append("fresh_nll_over_1.5pct")
    if (
        float(repaired["token_top5_rate"]) + 0.0075
        < float(parent["token_top5_rate"])
    ):
        reasons.append("fresh_top5_drop_over_0.75pp")
    if (
        float(repaired["mean_first_rank"])
        > float(parent["mean_first_rank"]) * 1.05
    ):
        reasons.append("fresh_first_rank_over_5pct")
    if (
        float(repaired["first_top1_rate"]) + 0.02
        < float(parent["first_top1_rate"])
    ):
        reasons.append("fresh_first_top1_drop_over_2pp")
    if int(repaired["generation_errors"]) > int(
        parent["generation_errors"]
    ):
        reasons.append("fresh_more_generation_errors")

    for category in ("tool_intent", "authority_behavior"):
        base_cat = parent["categories"][category]
        repaired_cat = repaired["categories"][category]
        if (
            float(repaired_cat["mean_nll"])
            > float(base_cat["mean_nll"]) * 1.03
        ):
            reasons.append(f"fresh_{category}_nll_over_3pct")

    if reasons:
        return "FRESH_PREFIX_REJECTED_REGRESSION", reasons, False

    generation_gain = int(repaired["generation_passed"]) > int(
        parent["generation_passed"]
    )
    qat_ready = (
        generation_gain
        and float(repaired["generation_pass_rate"])
        >= QAT_MIN_GENERATION_RATE
    )
    if generation_gain:
        return "FRESH_PREFIX_GENERATION_GAIN", [], qat_ready

    prefix_gain = (
        float(repaired["first_top1_rate"])
        >= float(parent["first_top1_rate"]) + 0.02
        or float(repaired["mean_first_rank"])
        <= float(parent["mean_first_rank"]) * 0.80
    )
    latent_gain = (
        float(repaired["mean_nll"])
        <= float(parent["mean_nll"]) * 0.99
        or float(repaired["token_top1_rate"])
        >= float(parent["token_top1_rate"]) + 0.01
        or float(repaired["token_top5_rate"])
        >= float(parent["token_top5_rate"]) + 0.01
    )

    if prefix_gain:
        return "FRESH_PREFIX_SIGNAL_GAIN", [], False
    if latent_gain:
        return "FRESH_PREFIX_LATENT_GAIN", [], False
    if (
        int(parent["generation_passed"]) == 0
        and int(repaired["generation_passed"]) == 0
    ):
        return "FRESH_PREFIX_ZERO_GENERATION", [], False
    return "FRESH_PREFIX_PRESERVED", [], False


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise VN97P5E4HError("CUDA device requested but CUDA is unavailable")

    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise VN97P5E4HError(
            "output already exists; P5E4H is a one-shot validation"
        )

    base_model, tokenizer, repaired_meta = _load_repaired(
        Path(args.repaired_dir)
    )
    parent_adapter, parent_meta = _load_expansion(
        Path(args.expanded_dir),
        repaired_meta=repaired_meta,
        d_model=base_model.config.d_model,
        vocab_size=base_model.config.vocab_size,
    )
    repaired_adapter, prefix_meta = _load_prefix_repaired(
        Path(args.prefix_repaired_dir),
        repaired_meta=repaired_meta,
        parent_meta=parent_meta,
        d_model=base_model.config.d_model,
        vocab_size=base_model.config.vocab_size,
    )

    parent_model = VN97OutputExpandedModel(
        base_model,
        parent_adapter,
    )
    repaired_model = VN97OutputExpandedModel(
        base_model,
        repaired_adapter,
    )

    seed_hex = _fresh_seed(
        str(repaired_meta["student_sha256"]),
        str(parent_meta["adapter_sha256"]),
        str(prefix_meta["adapter_sha256"]),
    )
    probes = _build_probes(
        seed_hex=seed_hex,
        per_category=PER_CATEGORY,
    )
    manifest_sha256 = hashlib.sha256(
        b"".join(
            (
                probe.probe_id
                + "\0"
                + probe.category
                + "\0"
                + probe.prompt
                + "\0"
                + probe.target
                + "\n"
            ).encode("utf-8")
            for probe in probes
        )
    ).hexdigest()

    print(
        "VN97 P5E4H FRESH "
        f"seed_sha256={seed_hex} "
        f"manifest_sha256={manifest_sha256} "
        f"tasks={len(probes)} "
        "reused_p4_suite=false "
        "reused_p5e4b_manifest=false "
        "reused_p5e4d_manifest=false "
        "reused_p5e4f_manifest=false "
        f"parent_adapter_sha256={parent_meta['adapter_sha256']} "
        f"repaired_adapter_sha256={prefix_meta['adapter_sha256']}",
        flush=True,
    )

    parent = _evaluate(
        label="p5e4e-parent",
        model=parent_model,
        tokenizer=tokenizer,
        probes=probes,
        device=args.device,
    )
    repaired = _evaluate(
        label="p5e4g-prefix-repaired",
        model=repaired_model,
        tokenizer=tokenizer,
        probes=probes,
        device=args.device,
    )

    status, reasons, qat_ready = _gate(
        parent=parent,
        repaired=repaired,
    )

    report = {
        "artifacts": {
            "repaired_core": repaired_meta,
            "parent_output": parent_meta,
            "prefix_repaired_output": prefix_meta,
        },
        "evaluations": {
            "parent": parent,
            "prefix_repaired": repaired,
        },
        "fresh_generation": {
            "manifest_sha256": manifest_sha256,
            "per_category": PER_CATEGORY,
            "reused_p4_suite": False,
            "reused_p5e4b_manifest": False,
            "reused_p5e4d_manifest": False,
            "reused_p5e4f_manifest": False,
            "seed_sha256": seed_hex,
            "task_count": len(probes),
        },
        "one_shot": True,
        "profile_id": P5E4H_PROFILE_ID,
        "qat_min_generation_rate": QAT_MIN_GENERATION_RATE,
        "ready_for_qat": qat_ready,
        "ready_for_next_decoding_stage": (
            not qat_ready
            and status
            in {
                "FRESH_PREFIX_SIGNAL_GAIN",
                "FRESH_PREFIX_LATENT_GAIN",
                "FRESH_PREFIX_ZERO_GENERATION",
                "FRESH_PREFIX_PRESERVED",
                "FRESH_PREFIX_GENERATION_GAIN",
            }
        ),
        "reasons": reasons,
        "schema": P5E4H_SCHEMA,
        "status": status,
    }

    _atomic_write(
        output,
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8") + b"\n",
    )

    print(
        "VN97P5E4H "
        f"status={status} "
        f"parent_nll={parent['mean_nll']:.6f} "
        f"repaired_nll={repaired['mean_nll']:.6f} "
        f"parent_top1={parent['token_top1_rate']:.6f} "
        f"repaired_top1={repaired['token_top1_rate']:.6f} "
        f"parent_top5={parent['token_top5_rate']:.6f} "
        f"repaired_top5={repaired['token_top5_rate']:.6f} "
        f"parent_first_top1={parent['first_top1_rate']:.6f} "
        f"repaired_first_top1={repaired['first_top1_rate']:.6f} "
        f"parent_first_rank={parent['mean_first_rank']:.2f} "
        f"repaired_first_rank={repaired['mean_first_rank']:.2f} "
        f"parent_generation="
        f"{parent['generation_passed']}/{parent['tasks']} "
        f"repaired_generation="
        f"{repaired['generation_passed']}/{repaired['tasks']} "
        f"ready_for_qat={str(qat_ready).lower()} "
        f"ready_for_next_decoding_stage="
        f"{str(report['ready_for_next_decoding_stage']).lower()} "
        f"reasons={','.join(reasons) if reasons else 'none'} "
        f"report={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-p5e4h-validate: {exc}", file=sys.stderr)
        raise
