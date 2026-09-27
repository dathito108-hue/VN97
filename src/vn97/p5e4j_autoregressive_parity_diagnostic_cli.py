from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

from .cognition_adapter import _no_repeat_banned_tokens
from .p4_task_evaluation import P4_CATEGORIES
from .p5e4b_fresh_validation_cli import _build_probes
from .p5e4d_fresh_decoder_validation_cli import _load_repaired
from .p5e4f_fresh_output_validation_cli import (
    VN97OutputExpandedModel,
    _load_expansion,
)
from .p5e4h_fresh_post_prefix_validation_cli import (
    P5E4H_SCHEMA,
    _load_prefix_repaired,
)
from .training import VN97ChatMessage, encode_chat_completion_prompt
from .training_cli import _atomic_write


P5E4J_SCHEMA = "VN97P5E4J"
P5E4J_PROFILE_ID = "vn97-p5e4j-autoregressive-parity-diagnostic-v1"
PER_CATEGORY = 12
REPETITION_PENALTY = 1.12
NO_REPEAT_NGRAM_SIZE = 4


class VN97P5E4JError(RuntimeError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "P5E4J diagnostic: locate the exact first autoregressive divergence "
            "and audit sequential-reference versus production parallel runtime "
            "parity before any further decoder training."
        )
    )
    parser.add_argument("--repaired-dir", required=True)
    parser.add_argument("--expanded-dir", required=True)
    parser.add_argument("--prefix-repaired-dir", required=True)
    parser.add_argument("--p5e4h-report", required=True)
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


def _verify_p5e4h(
    path: Path,
    *,
    repaired_meta: dict[str, Any],
    prefix_meta: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    resolved = path.resolve(strict=True)
    raw = resolved.read_bytes()
    report = json.loads(raw.decode("utf-8"))
    if not isinstance(report, dict) or report.get("schema") != P5E4H_SCHEMA:
        raise VN97P5E4JError("invalid P5E4H report schema")
    if report.get("one_shot") is not True:
        raise VN97P5E4JError("P5E4H evidence is not one-shot")
    if report.get("ready_for_qat") is not False:
        raise VN97P5E4JError("P5E4J is only valid before QAT")
    if report.get("ready_for_next_decoding_stage") is not True:
        raise VN97P5E4JError("P5E4H did not authorize further diagnosis")
    if report.get("status") not in {
        "FRESH_PREFIX_SIGNAL_GAIN",
        "FRESH_PREFIX_LATENT_GAIN",
        "FRESH_PREFIX_ZERO_GENERATION",
        "FRESH_PREFIX_PRESERVED",
        "FRESH_PREFIX_GENERATION_GAIN",
    }:
        raise VN97P5E4JError("P5E4H status is not diagnostic eligible")

    artifacts = report.get("artifacts")
    repaired = (
        artifacts.get("repaired_core")
        if isinstance(artifacts, dict)
        else None
    )
    prefix = (
        artifacts.get("prefix_repaired_output")
        if isinstance(artifacts, dict)
        else None
    )
    if not isinstance(repaired, dict) or not isinstance(prefix, dict):
        raise VN97P5E4JError("P5E4H artifact identities missing")
    if repaired.get("student_sha256") != repaired_meta.get("student_sha256"):
        raise VN97P5E4JError("P5E4H/P5E4C checkpoint identity mismatch")
    if prefix.get("adapter_sha256") != prefix_meta.get("adapter_sha256"):
        raise VN97P5E4JError("P5E4H/P5E4G adapter identity mismatch")
    return report, hashlib.sha256(raw).hexdigest()


def _adjust_scores(
    scores: torch.Tensor,
    generated: list[int],
) -> tuple[torch.Tensor, set[int]]:
    adjusted = scores.clone()
    if REPETITION_PENALTY > 1.0 and generated:
        for token_id in set(generated):
            value = adjusted[token_id]
            adjusted[token_id] = torch.where(
                value < 0,
                value * REPETITION_PENALTY,
                value / REPETITION_PENALTY,
            )

    banned = _no_repeat_banned_tokens(
        generated,
        NO_REPEAT_NGRAM_SIZE,
    )
    for token_id in banned:
        adjusted[token_id] = -torch.inf
    return adjusted, banned


def _rank(scores: torch.Tensor, target: int) -> int:
    target_score = scores[target]
    if not bool(torch.isfinite(target_score)):
        return int(scores.numel()) + 1
    return int((scores > target_score).sum().item()) + 1


@torch.inference_mode()
def _trace_probe(
    *,
    model,
    tokenizer,
    prompt: str,
    target_text: str,
    device: str,
) -> dict[str, Any]:
    prompt_tokens = list(
        encode_chat_completion_prompt(
            tokenizer,
            (
                VN97ChatMessage(
                    role="user",
                    content=prompt,
                ),
            ),
        )
    )
    target_tokens = list(tokenizer.encode(target_text))
    if not target_tokens:
        raise VN97P5E4JError("diagnostic target encoded to zero tokens")
    expected = [*target_tokens, tokenizer.eos_id]

    input_ids = torch.tensor(
        [prompt_tokens],
        dtype=torch.long,
        device=device,
    )

    seq_logits, seq_states = model.forward_sequential_reference(input_ids)
    prod_logits, prod_states = model(input_ids)

    generated_gold_prefix: list[int] = []
    first_seq_divergence: int | None = None
    first_prod_divergence: int | None = None
    seq_first_rank = 0
    prod_first_rank = 0
    seq_first_constrained_rank = 0
    prod_first_constrained_rank = 0
    raw_top1_agree = 0
    constrained_top1_agree = 0
    runtime_prod_lost_correct = 0
    runtime_prod_gained_correct = 0
    constraint_flip_seq = 0
    constraint_flip_prod = 0
    target_banned_seq = 0
    target_banned_prod = 0
    max_abs_logit_diff = 0.0
    mean_abs_logit_diff_sum = 0.0
    traced_steps = 0
    divergence_rows: list[dict[str, Any]] = []

    for step, target in enumerate(expected):
        seq_scores_raw = seq_logits[0, -1].float()
        prod_scores_raw = prod_logits[0, -1].float()

        diff = (seq_scores_raw - prod_scores_raw).abs()
        max_abs_logit_diff = max(
            max_abs_logit_diff,
            float(diff.max().item()),
        )
        mean_abs_logit_diff_sum += float(diff.mean().item())
        traced_steps += 1

        seq_raw_top1 = int(seq_scores_raw.argmax().item())
        prod_raw_top1 = int(prod_scores_raw.argmax().item())
        raw_top1_agree += int(seq_raw_top1 == prod_raw_top1)

        seq_adjusted, seq_banned = _adjust_scores(
            seq_scores_raw,
            generated_gold_prefix,
        )
        prod_adjusted, prod_banned = _adjust_scores(
            prod_scores_raw,
            generated_gold_prefix,
        )

        seq_selected = int(seq_adjusted.argmax().item())
        prod_selected = int(prod_adjusted.argmax().item())
        constrained_top1_agree += int(seq_selected == prod_selected)

        seq_raw_rank = _rank(seq_scores_raw, target)
        prod_raw_rank = _rank(prod_scores_raw, target)
        seq_constrained_rank = _rank(seq_adjusted, target)
        prod_constrained_rank = _rank(prod_adjusted, target)

        if step == 0:
            seq_first_rank = seq_raw_rank
            prod_first_rank = prod_raw_rank
            seq_first_constrained_rank = seq_constrained_rank
            prod_first_constrained_rank = prod_constrained_rank

        seq_correct = seq_selected == target
        prod_correct = prod_selected == target
        runtime_prod_lost_correct += int(seq_correct and not prod_correct)
        runtime_prod_gained_correct += int(prod_correct and not seq_correct)

        constraint_flip_seq += int(
            seq_raw_top1 == target and seq_selected != target
        )
        constraint_flip_prod += int(
            prod_raw_top1 == target and prod_selected != target
        )
        target_banned_seq += int(target in seq_banned)
        target_banned_prod += int(target in prod_banned)

        if not seq_correct and first_seq_divergence is None:
            first_seq_divergence = step
        if not prod_correct and first_prod_divergence is None:
            first_prod_divergence = step

        if (
            (not seq_correct or not prod_correct)
            and len(divergence_rows) < 4
        ):
            divergence_rows.append(
                {
                    "step": step,
                    "is_eos_target": target == tokenizer.eos_id,
                    "seq_raw_rank": seq_raw_rank,
                    "prod_raw_rank": prod_raw_rank,
                    "seq_constrained_rank": seq_constrained_rank,
                    "prod_constrained_rank": prod_constrained_rank,
                    "seq_selected_matches_target": seq_correct,
                    "prod_selected_matches_target": prod_correct,
                    "seq_target_banned": target in seq_banned,
                    "prod_target_banned": target in prod_banned,
                }
            )

        if step + 1 >= len(expected):
            break

        if target == tokenizer.eos_id:
            break

        generated_gold_prefix.append(target)
        next_token = torch.tensor(
            [[target]],
            dtype=torch.long,
            device=device,
        )
        seq_logits, seq_states = model.forward_sequential_reference(
            next_token,
            seq_states,
        )
        prod_logits, prod_states = model(
            next_token,
            prod_states,
        )

    seq_exact = first_seq_divergence is None
    prod_exact = first_prod_divergence is None

    return {
        "target_tokens_with_eos": len(expected),
        "seq_exact_gold_path": seq_exact,
        "prod_exact_gold_path": prod_exact,
        "seq_first_divergence": (
            first_seq_divergence
            if first_seq_divergence is not None
            else len(expected)
        ),
        "prod_first_divergence": (
            first_prod_divergence
            if first_prod_divergence is not None
            else len(expected)
        ),
        "seq_first_rank": seq_first_rank,
        "prod_first_rank": prod_first_rank,
        "seq_first_constrained_rank": seq_first_constrained_rank,
        "prod_first_constrained_rank": prod_first_constrained_rank,
        "raw_top1_agreement": raw_top1_agree / traced_steps,
        "constrained_top1_agreement": constrained_top1_agree / traced_steps,
        "runtime_prod_lost_correct": runtime_prod_lost_correct,
        "runtime_prod_gained_correct": runtime_prod_gained_correct,
        "constraint_flip_seq": constraint_flip_seq,
        "constraint_flip_prod": constraint_flip_prod,
        "target_banned_seq": target_banned_seq,
        "target_banned_prod": target_banned_prod,
        "max_abs_logit_diff": max_abs_logit_diff,
        "mean_abs_logit_diff": mean_abs_logit_diff_sum / traced_steps,
        "traced_steps": traced_steps,
        "divergences": divergence_rows,
    }


def _bucket(position: int, length: int) -> str:
    if position >= length:
        return "exact"
    if position == 0:
        return "token0"
    if position <= 3:
        return "token1_3"
    if position <= 15:
        return "token4_15"
    if position == length - 1:
        return "eos"
    return "token16_plus"


def _diagnose(summary: dict[str, Any]) -> tuple[str, str]:
    parity = float(summary["constrained_top1_agreement"])
    prod_lost = int(summary["runtime_prod_lost_correct"])
    constraint_flips = int(summary["constraint_flip_prod"])
    prod_banned = int(summary["target_banned_prod"])
    buckets = summary["prod_divergence_buckets"]
    probes = int(summary["probes"])

    if parity < 0.995 or prod_lost > 0:
        return (
            "RUNTIME_PARITY_GAP",
            "production_parallel_parity_repair",
        )
    if constraint_flips > 0 or prod_banned > 0:
        return (
            "DECODING_CONSTRAINT_INTERFERENCE",
            "generation_constraint_repair",
        )
    if int(buckets["token0"]) >= max(1, int(0.40 * probes)):
        return (
            "FIRST_TOKEN_DECISION_GAP",
            "first_token_logit_calibration",
        )
    early = (
        int(buckets["token0"])
        + int(buckets["token1_3"])
        + int(buckets["token4_15"])
    )
    if early >= max(1, int(0.70 * probes)):
        return (
            "EARLY_PREFIX_DECISION_GAP",
            "autoregressive_prefix_curriculum",
        )
    if int(buckets["eos"]) >= max(1, int(0.20 * probes)):
        return (
            "EOS_BOUNDARY_GAP",
            "eos_boundary_calibration",
        )
    return (
        "DISTRIBUTED_SEQUENCE_DECISION_GAP",
        "sequence_level_decoding_repair",
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise VN97P5E4JError("CUDA device requested but CUDA is unavailable")

    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise VN97P5E4JError(
            "output already exists; P5E4J is a one-shot diagnostic"
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
    del parent_adapter

    adapter, prefix_meta = _load_prefix_repaired(
        Path(args.prefix_repaired_dir),
        repaired_meta=repaired_meta,
        parent_meta=parent_meta,
        d_model=base_model.config.d_model,
        vocab_size=base_model.config.vocab_size,
    )
    p5e4h_report, p5e4h_sha = _verify_p5e4h(
        Path(args.p5e4h_report),
        repaired_meta=repaired_meta,
        prefix_meta=prefix_meta,
    )

    model = VN97OutputExpandedModel(
        base_model,
        adapter,
    ).to(args.device)
    model.eval()

    seed_hex = hashlib.sha256(
        (
            P5E4J_PROFILE_ID
            + "\0"
            + str(repaired_meta["student_sha256"])
            + "\0"
            + str(prefix_meta["adapter_sha256"])
            + "\0"
            + p5e4h_sha
        ).encode("utf-8")
    ).hexdigest()
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
        "VN97 P5E4J START "
        f"p5e4h_status={p5e4h_report['status']} "
        f"seed_sha256={seed_hex} "
        f"manifest_sha256={manifest_sha256} "
        f"tasks={len(probes)} "
        "audit=sequential_reference_vs_production_parallel "
        "ready_for_qat=false",
        flush=True,
    )

    buckets = defaultdict(int)
    category_buckets: dict[str, dict[str, int]] = {
        category: defaultdict(int)
        for category in P4_CATEGORIES
    }
    rows: list[dict[str, Any]] = []
    totals = {
        "traced_steps": 0,
        "raw_top1_agreement_sum": 0.0,
        "constrained_top1_agreement_sum": 0.0,
        "runtime_prod_lost_correct": 0,
        "runtime_prod_gained_correct": 0,
        "constraint_flip_seq": 0,
        "constraint_flip_prod": 0,
        "target_banned_seq": 0,
        "target_banned_prod": 0,
        "seq_exact": 0,
        "prod_exact": 0,
        "seq_first_divergence_sum": 0,
        "prod_first_divergence_sum": 0,
        "max_abs_logit_diff": 0.0,
        "mean_abs_logit_diff_sum": 0.0,
    }

    for index, probe in enumerate(probes, start=1):
        trace = _trace_probe(
            model=model,
            tokenizer=tokenizer,
            prompt=probe.prompt,
            target_text=probe.target,
            device=args.device,
        )
        length = int(trace["target_tokens_with_eos"])
        prod_bucket = _bucket(
            int(trace["prod_first_divergence"]),
            length,
        )
        seq_bucket = _bucket(
            int(trace["seq_first_divergence"]),
            length,
        )
        buckets[prod_bucket] += 1
        category_buckets[probe.category][prod_bucket] += 1

        totals["traced_steps"] += int(trace["traced_steps"])
        totals["raw_top1_agreement_sum"] += (
            float(trace["raw_top1_agreement"])
            * int(trace["traced_steps"])
        )
        totals["constrained_top1_agreement_sum"] += (
            float(trace["constrained_top1_agreement"])
            * int(trace["traced_steps"])
        )
        for key in (
            "runtime_prod_lost_correct",
            "runtime_prod_gained_correct",
            "constraint_flip_seq",
            "constraint_flip_prod",
            "target_banned_seq",
            "target_banned_prod",
        ):
            totals[key] += int(trace[key])
        totals["seq_exact"] += int(bool(trace["seq_exact_gold_path"]))
        totals["prod_exact"] += int(bool(trace["prod_exact_gold_path"]))
        totals["seq_first_divergence_sum"] += int(
            trace["seq_first_divergence"]
        )
        totals["prod_first_divergence_sum"] += int(
            trace["prod_first_divergence"]
        )
        totals["max_abs_logit_diff"] = max(
            float(totals["max_abs_logit_diff"]),
            float(trace["max_abs_logit_diff"]),
        )
        totals["mean_abs_logit_diff_sum"] += (
            float(trace["mean_abs_logit_diff"])
            * int(trace["traced_steps"])
        )

        rows.append(
            {
                "probe_id": probe.probe_id,
                "category": probe.category,
                "prompt_sha256": hashlib.sha256(
                    probe.prompt.encode("utf-8")
                ).hexdigest(),
                "target_sha256": hashlib.sha256(
                    probe.target.encode("utf-8")
                ).hexdigest(),
                "prod_divergence_bucket": prod_bucket,
                "seq_divergence_bucket": seq_bucket,
                **trace,
            }
        )

        if index % 12 == 0 or index == len(probes):
            steps = max(1, int(totals["traced_steps"]))
            print(
                "VN97 P5E4J PROGRESS "
                f"tasks={index}/{len(probes)} "
                f"prod_exact={totals['prod_exact']} "
                f"seq_exact={totals['seq_exact']} "
                f"runtime_agreement="
                f"{float(totals['constrained_top1_agreement_sum']) / steps:.6f} "
                f"prod_lost_correct={totals['runtime_prod_lost_correct']} "
                f"constraint_flips={totals['constraint_flip_prod']}",
                flush=True,
            )

    steps = max(1, int(totals["traced_steps"]))
    normalized_buckets = {
        name: int(buckets[name])
        for name in (
            "token0",
            "token1_3",
            "token4_15",
            "token16_plus",
            "eos",
            "exact",
        )
    }
    summary = {
        "probes": len(probes),
        "traced_steps": int(totals["traced_steps"]),
        "raw_top1_agreement": (
            float(totals["raw_top1_agreement_sum"]) / steps
        ),
        "constrained_top1_agreement": (
            float(totals["constrained_top1_agreement_sum"]) / steps
        ),
        "runtime_prod_lost_correct": int(
            totals["runtime_prod_lost_correct"]
        ),
        "runtime_prod_gained_correct": int(
            totals["runtime_prod_gained_correct"]
        ),
        "constraint_flip_seq": int(totals["constraint_flip_seq"]),
        "constraint_flip_prod": int(totals["constraint_flip_prod"]),
        "target_banned_seq": int(totals["target_banned_seq"]),
        "target_banned_prod": int(totals["target_banned_prod"]),
        "seq_exact_gold_path": int(totals["seq_exact"]),
        "prod_exact_gold_path": int(totals["prod_exact"]),
        "mean_seq_first_divergence": (
            int(totals["seq_first_divergence_sum"]) / len(probes)
        ),
        "mean_prod_first_divergence": (
            int(totals["prod_first_divergence_sum"]) / len(probes)
        ),
        "max_abs_logit_diff": float(totals["max_abs_logit_diff"]),
        "mean_abs_logit_diff": (
            float(totals["mean_abs_logit_diff_sum"]) / steps
        ),
        "prod_divergence_buckets": normalized_buckets,
        "category_prod_divergence_buckets": {
            category: {
                name: int(category_buckets[category][name])
                for name in (
                    "token0",
                    "token1_3",
                    "token4_15",
                    "token16_plus",
                    "eos",
                    "exact",
                )
            }
            for category in P4_CATEGORIES
        },
    }

    status, next_stage = _diagnose(summary)

    report = {
        "artifacts": {
            "repaired_core": repaired_meta,
            "parent_output": parent_meta,
            "prefix_repaired_output": prefix_meta,
        },
        "diagnostic": summary,
        "fresh_manifest": {
            "manifest_sha256": manifest_sha256,
            "per_category": PER_CATEGORY,
            "seed_sha256": seed_hex,
            "task_count": len(probes),
            "reused_p4_suite": False,
            "reused_p5e4b_manifest": False,
            "reused_p5e4d_manifest": False,
            "reused_p5e4f_manifest": False,
            "reused_p5e4h_manifest": False,
        },
        "one_shot": True,
        "p5e4h_report_sha256": p5e4h_sha,
        "p5e4h_status": p5e4h_report["status"],
        "profile_id": P5E4J_PROFILE_ID,
        "ready_for_qat": False,
        "recommended_next_stage": next_stage,
        "rows": rows,
        "schema": P5E4J_SCHEMA,
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
        "VN97P5E4J "
        f"status={status} "
        f"runtime_agreement={summary['constrained_top1_agreement']:.6f} "
        f"raw_runtime_agreement={summary['raw_top1_agreement']:.6f} "
        f"prod_lost_correct={summary['runtime_prod_lost_correct']} "
        f"constraint_flips={summary['constraint_flip_prod']} "
        f"target_banned={summary['target_banned_prod']} "
        f"seq_exact={summary['seq_exact_gold_path']}/{len(probes)} "
        f"prod_exact={summary['prod_exact_gold_path']}/{len(probes)} "
        f"mean_seq_divergence={summary['mean_seq_first_divergence']:.2f} "
        f"mean_prod_divergence={summary['mean_prod_first_divergence']:.2f} "
        f"token0={normalized_buckets['token0']} "
        f"token1_3={normalized_buckets['token1_3']} "
        f"token4_15={normalized_buckets['token4_15']} "
        f"token16_plus={normalized_buckets['token16_plus']} "
        f"eos={normalized_buckets['eos']} "
        f"exact={normalized_buckets['exact']} "
        f"max_abs_logit_diff={summary['max_abs_logit_diff']:.6f} "
        f"recommended_next_stage={next_stage} "
        "ready_for_qat=false "
        f"report={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-p5e4j-diagnose: {exc}", file=sys.stderr)
        raise
