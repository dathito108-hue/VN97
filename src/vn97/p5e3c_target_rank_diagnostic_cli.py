from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

from .p4_task_evaluation import P4_CATEGORIES, load_p4_task_suite
from .p5e3_heldout_validation_cli import _load_artifact
from .training import VN97ChatMessage, encode_chat_completion_prompt
from .training_cli import _atomic_write


P5E3C_SCHEMA = "VN97P5E3C"
P5E3C_PROFILE_ID = "vn97-p5e3c-heldout-target-rank-diagnostic-v1"


class VN97P5E3CError(RuntimeError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Teacher-forced target-rank diagnostic for the P5E3 zero-signal "
            "case. It compares base, P5E1 and P5E2 without changing weights."
        )
    )
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--p5e1-dir", required=True)
    parser.add_argument("--p5e2-dir", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _targets_for_task(task) -> tuple[str, ...]:
    if task.scoring_kind == "exact_text":
        if not isinstance(task.expected, str):
            raise VN97P5E3CError("exact_text task expected value is not text")
        return (task.expected,)
    if task.scoring_kind == "json_exact":
        return (_canonical_json(task.expected),)
    if task.scoring_kind == "contains_all":
        if not isinstance(task.expected, tuple) or not task.expected:
            raise VN97P5E3CError("contains_all task expected value is invalid")
        return tuple(str(item) for item in task.expected)
    raise VN97P5E3CError(f"unsupported scoring kind: {task.scoring_kind}")


@torch.inference_mode()
def _score_target(
    *,
    model,
    tokenizer,
    prompt: str,
    target_text: str,
    device: str,
) -> dict[str, Any]:
    prompt_tokens = encode_chat_completion_prompt(
        tokenizer,
        (
            VN97ChatMessage(
                role="user",
                content=prompt,
            ),
        ),
    )
    target_tokens = tuple(tokenizer.encode(target_text))
    if not target_tokens:
        raise VN97P5E3CError("diagnostic target encoded to zero tokens")

    input_tokens = (
        *prompt_tokens,
        *target_tokens[:-1],
    )
    input_ids = torch.tensor(
        [input_tokens],
        dtype=torch.long,
        device=device,
    )
    logits, _ = model.forward_sequential_reference(input_ids)

    start = len(prompt_tokens) - 1
    end = start + len(target_tokens)
    target_logits = logits[0, start:end, :].float()
    labels = torch.tensor(
        target_tokens,
        dtype=torch.long,
        device=device,
    )

    nll_sum = float(
        F.cross_entropy(
            target_logits,
            labels,
            reduction="sum",
        ).item()
    )

    label_logits = target_logits.gather(
        1,
        labels.unsqueeze(1),
    ).squeeze(1)
    ranks = (
        (target_logits > label_logits.unsqueeze(1)).sum(dim=1)
        + 1
    ).to(torch.int64)

    token_count = int(labels.numel())
    top1 = int((ranks <= 1).sum().item())
    top5 = int((ranks <= 5).sum().item())
    top20 = int((ranks <= 20).sum().item())
    first_rank = int(ranks[0].item())

    return {
        "first_rank": first_rank,
        "first_top1": first_rank <= 1,
        "first_top5": first_rank <= 5,
        "first_top20": first_rank <= 20,
        "nll_sum": nll_sum,
        "target_sha256": hashlib.sha256(
            target_text.encode("utf-8")
        ).hexdigest(),
        "target_utf8_bytes": len(target_text.encode("utf-8")),
        "token_count": token_count,
        "top1_tokens": top1,
        "top5_tokens": top5,
        "top20_tokens": top20,
    }


def _evaluate(
    *,
    label: str,
    model,
    tokenizer,
    suite,
    device: str,
) -> dict[str, Any]:
    model.to(device)
    model.eval()

    rows: list[dict[str, Any]] = []
    total_tokens = 0
    total_nll = 0.0
    total_top1 = 0
    total_top5 = 0
    total_top20 = 0
    first_ranks: list[int] = []
    first_top1 = 0
    first_top5 = 0
    first_top20 = 0
    by_category: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "targets": 0,
            "tokens": 0,
            "nll_sum": 0.0,
            "top1_tokens": 0,
            "top5_tokens": 0,
            "top20_tokens": 0,
            "first_rank_sum": 0,
            "first_top5": 0,
        }
    )

    started = time.monotonic()
    target_index = 0
    for task in suite.tasks:
        for variant_index, target_text in enumerate(
            _targets_for_task(task)
        ):
            target_index += 1
            metrics = _score_target(
                model=model,
                tokenizer=tokenizer,
                prompt=task.prompt,
                target_text=target_text,
                device=device,
            )
            metrics.update(
                {
                    "category": task.category,
                    "scoring_kind": task.scoring_kind,
                    "task_id": task.task_id,
                    "variant_index": variant_index,
                }
            )
            rows.append(metrics)

            token_count = int(metrics["token_count"])
            total_tokens += token_count
            total_nll += float(metrics["nll_sum"])
            total_top1 += int(metrics["top1_tokens"])
            total_top5 += int(metrics["top5_tokens"])
            total_top20 += int(metrics["top20_tokens"])
            rank = int(metrics["first_rank"])
            first_ranks.append(rank)
            first_top1 += int(bool(metrics["first_top1"]))
            first_top5 += int(bool(metrics["first_top5"]))
            first_top20 += int(bool(metrics["first_top20"]))

            category = by_category[task.category]
            category["targets"] = int(category["targets"]) + 1
            category["tokens"] = int(category["tokens"]) + token_count
            category["nll_sum"] = float(category["nll_sum"]) + float(
                metrics["nll_sum"]
            )
            category["top1_tokens"] = int(category["top1_tokens"]) + int(
                metrics["top1_tokens"]
            )
            category["top5_tokens"] = int(category["top5_tokens"]) + int(
                metrics["top5_tokens"]
            )
            category["top20_tokens"] = int(category["top20_tokens"]) + int(
                metrics["top20_tokens"]
            )
            category["first_rank_sum"] = int(
                category["first_rank_sum"]
            ) + rank
            category["first_top5"] = int(category["first_top5"]) + int(
                bool(metrics["first_top5"])
            )

            print(
                "VN97 P5E3C TARGET "
                f"model={label} "
                f"target={target_index} "
                f"task={task.task_id} "
                f"category={task.category} "
                f"tokens={token_count} "
                f"first_rank={rank} "
                f"top1={int(metrics['top1_tokens'])}/{token_count} "
                f"top5={int(metrics['top5_tokens'])}/{token_count} "
                f"mean_nll={float(metrics['nll_sum']) / token_count:.6f}",
                flush=True,
            )

    if total_tokens <= 0 or not first_ranks:
        raise VN97P5E3CError("diagnostic produced no target tokens")

    category_report: dict[str, dict[str, float | int]] = {}
    for category in P4_CATEGORIES:
        row = by_category.get(category)
        if row is None:
            category_report[category] = {
                "targets": 0,
                "tokens": 0,
                "mean_nll": 0.0,
                "top1_rate": 0.0,
                "top5_rate": 0.0,
                "top20_rate": 0.0,
                "mean_first_rank": 0.0,
                "first_top5_rate": 0.0,
            }
            continue
        tokens = int(row["tokens"])
        targets = int(row["targets"])
        category_report[category] = {
            "targets": targets,
            "tokens": tokens,
            "mean_nll": float(row["nll_sum"]) / max(tokens, 1),
            "top1_rate": int(row["top1_tokens"]) / max(tokens, 1),
            "top5_rate": int(row["top5_tokens"]) / max(tokens, 1),
            "top20_rate": int(row["top20_tokens"]) / max(tokens, 1),
            "mean_first_rank": int(row["first_rank_sum"]) / max(targets, 1),
            "first_top5_rate": int(row["first_top5"]) / max(targets, 1),
        }

    result = {
        "categories": category_report,
        "elapsed_s": time.monotonic() - started,
        "first_target_count": len(first_ranks),
        "first_top1_rate": first_top1 / len(first_ranks),
        "first_top5_rate": first_top5 / len(first_ranks),
        "first_top20_rate": first_top20 / len(first_ranks),
        "mean_first_rank": sum(first_ranks) / len(first_ranks),
        "mean_nll": total_nll / total_tokens,
        "perplexity": math.exp(min(20.0, total_nll / total_tokens)),
        "target_tokens": total_tokens,
        "token_top1_rate": total_top1 / total_tokens,
        "token_top5_rate": total_top5 / total_tokens,
        "token_top20_rate": total_top20 / total_tokens,
        "targets": rows,
    }

    print(
        "VN97 P5E3C MODEL "
        f"model={label} "
        f"mean_nll={result['mean_nll']:.6f} "
        f"token_top1={result['token_top1_rate']:.6f} "
        f"token_top5={result['token_top5_rate']:.6f} "
        f"token_top20={result['token_top20_rate']:.6f} "
        f"mean_first_rank={result['mean_first_rank']:.2f} "
        f"first_top5={result['first_top5_rate']:.6f}",
        flush=True,
    )

    model.cpu()
    torch.cuda.empty_cache()
    return result


def _status(
    base: dict[str, Any],
    p5e1: dict[str, Any],
    p5e2: dict[str, Any],
) -> tuple[str, list[str]]:
    reasons: list[str] = []

    base_nll = float(base["mean_nll"])
    p5e2_nll = float(p5e2["mean_nll"])
    base_top5 = float(base["token_top5_rate"])
    p5e2_top5 = float(p5e2["token_top5_rate"])
    base_first = float(base["mean_first_rank"])
    p5e2_first = float(p5e2["mean_first_rank"])

    if p5e2_nll > base_nll * 1.05:
        reasons.append("target_nll_regressed")
    if p5e2_top5 + 0.01 < base_top5:
        reasons.append("target_top5_regressed")
    if p5e2_first > base_first * 1.20:
        reasons.append("first_token_rank_regressed")

    for category in ("tool_intent", "authority_behavior"):
        base_cat = base["categories"][category]
        p5e2_cat = p5e2["categories"][category]
        if (
            int(base_cat["tokens"]) > 0
            and float(p5e2_cat["mean_nll"])
            > float(base_cat["mean_nll"]) * 1.08
        ):
            reasons.append(f"{category}_nll_regressed")

    if reasons:
        return "LATENT_SIGNAL_REGRESSION", reasons

    gain = (
        p5e2_nll < base_nll * 0.98
        or p5e2_top5 > base_top5 + 0.01
        or p5e2_first < base_first * 0.90
    )
    if gain:
        return "LATENT_SIGNAL_GAIN", []

    strongest_nll = min(
        base_nll,
        float(p5e1["mean_nll"]),
    )
    if p5e2_nll <= strongest_nll * 1.02:
        return "LATENT_SIGNAL_PRESERVED", []
    return "LATENT_SIGNAL_WEAK", []


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise VN97P5E3CError("CUDA device requested but CUDA is unavailable")

    suite = load_p4_task_suite(Path(args.suite).resolve(strict=True))
    artifact_dirs = {
        "base": Path(args.base_dir),
        "p5e1": Path(args.p5e1_dir),
        "p5e2": Path(args.p5e2_dir),
    }

    evaluations: dict[str, dict[str, Any]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    tokenizer_hash: str | None = None

    for label in ("base", "p5e1", "p5e2"):
        print(f"VN97 P5E3C LOAD model={label}", flush=True)
        model, tokenizer, meta = _load_artifact(
            artifact_dirs[label],
            label=label,
        )
        current_hash = str(meta["tokenizer_sha256"])
        if tokenizer_hash is None:
            tokenizer_hash = current_hash
        elif current_hash != tokenizer_hash:
            raise VN97P5E3CError(
                "P5E3C artifacts do not share one tokenizer"
            )

        metadata[label] = meta
        evaluations[label] = _evaluate(
            label=label,
            model=model,
            tokenizer=tokenizer,
            suite=suite,
            device=args.device,
        )
        del model, tokenizer
        torch.cuda.empty_cache()

    status, reasons = _status(
        evaluations["base"],
        evaluations["p5e1"],
        evaluations["p5e2"],
    )

    report = {
        "artifacts": metadata,
        "diagnostic_only": True,
        "evaluations": evaluations,
        "generation_gate_passed": False,
        "profile_id": P5E3C_PROFILE_ID,
        "reasons": reasons,
        "schema": P5E3C_SCHEMA,
        "status": status,
        "suite_sha256": suite.suite_sha256,
        "task_count": len(suite.tasks),
        "tokenizer_sha256": tokenizer_hash,
    }
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise VN97P5E3CError("output path must not already exist")
    _atomic_write(
        output,
        (
            json.dumps(
                report,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        ),
    )

    print(
        "VN97P5E3C "
        f"status={status} "
        f"base_nll={evaluations['base']['mean_nll']:.6f} "
        f"p5e1_nll={evaluations['p5e1']['mean_nll']:.6f} "
        f"p5e2_nll={evaluations['p5e2']['mean_nll']:.6f} "
        f"base_top5={evaluations['base']['token_top5_rate']:.6f} "
        f"p5e1_top5={evaluations['p5e1']['token_top5_rate']:.6f} "
        f"p5e2_top5={evaluations['p5e2']['token_top5_rate']:.6f} "
        f"base_first_rank={evaluations['base']['mean_first_rank']:.2f} "
        f"p5e1_first_rank={evaluations['p5e1']['mean_first_rank']:.2f} "
        f"p5e2_first_rank={evaluations['p5e2']['mean_first_rank']:.2f} "
        f"generation_gate_passed=false "
        f"reasons={','.join(reasons) if reasons else 'none'} "
        f"report={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-p5e3c-diagnose: {exc}", file=sys.stderr)
        raise
