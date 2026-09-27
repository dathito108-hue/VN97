from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn as nn

from .chat_boundary import recover_chat_response_text
from .cognition_adapter import TorchVN97InferenceEngine, VN97InferenceLimits
from .p4_task_evaluation import P4_CATEGORIES
from .p5e3c_target_rank_diagnostic_cli import _score_target
from .p5e4b_fresh_validation_cli import _build_probes
from .p5e4d_fresh_decoder_validation_cli import _load_repaired
from .p5e4e_output_expansion_cli import (
    ADAPTER_RANK,
    P5E4E_ADAPTER_SCHEMA,
    P5E4E_SCHEMA,
    ResidualOutputAdapter,
)
from .training import VN97ChatMessage
from .training_cli import _atomic_write


P5E4F_SCHEMA = "VN97P5E4F"
P5E4F_PROFILE_ID = "vn97-p5e4f-fresh-output-validation-v1"
PER_CATEGORY = 14
QAT_MIN_GENERATION_RATE = 0.10


class VN97P5E4FError(RuntimeError):
    pass


class VN97OutputExpandedModel(nn.Module):
    """Inference-only view: frozen P5E4C core + P5E4E residual output adapter."""

    def __init__(self, base_model, adapter: ResidualOutputAdapter) -> None:
        super().__init__()
        self.base_model = base_model
        self.adapter = adapter
        self.config = base_model.config

    def forward(self, input_ids, states=None):
        hidden, new_states = self.base_model.forward_hidden(
            input_ids,
            states,
        )
        logits = self.base_model.lm_head(hidden) + self.adapter(hidden)
        return logits, new_states

    def forward_sequential_reference(self, input_ids, states=None):
        hidden, new_states = (
            self.base_model.forward_hidden_embeddings_sequential_reference(
                self.base_model.embedding(input_ids),
                states,
            )
        )
        logits = self.base_model.lm_head(hidden) + self.adapter(hidden)
        return logits, new_states


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fresh one-shot generation validation for P5E4E output expansion. "
            "Compares the exact P5E4C repaired baseline against the frozen "
            "P5E4C core plus P5E4E low-rank output residual on a newly derived "
            "manifest that is distinct from P5E4B/P5E4D."
        )
    )
    parser.add_argument("--repaired-dir", required=True)
    parser.add_argument("--expanded-dir", required=True)
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


def _load_expansion(
    root: Path,
    *,
    repaired_meta: dict[str, Any],
    d_model: int,
    vocab_size: int,
) -> tuple[ResidualOutputAdapter, dict[str, Any]]:
    resolved = root.resolve(strict=True)
    report_path = resolved / "p5e4e-report.json"
    adapter_path = resolved / "output-adapter.pt"
    tokenizer_path = resolved / "tokenizer.vn97tk1"
    for path in (report_path, adapter_path, tokenizer_path):
        if not path.is_file():
            raise VN97P5E4FError(f"expanded artifact missing {path.name}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict)
        or report.get("schema") != P5E4E_SCHEMA
        or report.get("status") != "OUTPUT_EXPANSION_SIGNAL"
        or report.get("ready_for_fresh_validation") is not True
        or report.get("ready_for_qat") is not False
    ):
        raise VN97P5E4FError("P5E4E report contract mismatch")
    if (
        report.get("repaired_student_sha256")
        != repaired_meta.get("student_sha256")
    ):
        raise VN97P5E4FError("P5E4E/P5E4C checkpoint identity mismatch")

    payload = torch.load(
        adapter_path,
        map_location="cpu",
        weights_only=False,
    )
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != P5E4E_ADAPTER_SCHEMA
        or payload.get("status") != "OUTPUT_EXPANSION_SIGNAL"
        or payload.get("ready_for_fresh_validation") is not True
        or payload.get("ready_for_qat") is not False
    ):
        raise VN97P5E4FError("P5E4E adapter contract mismatch")
    if int(payload.get("rank", -1)) != ADAPTER_RANK:
        raise VN97P5E4FError("P5E4E adapter rank mismatch")
    if int(payload.get("d_model", -1)) != d_model:
        raise VN97P5E4FError("P5E4E adapter d_model mismatch")
    if int(payload.get("vocab_size", -1)) != vocab_size:
        raise VN97P5E4FError("P5E4E adapter vocabulary mismatch")
    if (
        payload.get("repaired_student_sha256")
        != repaired_meta.get("student_sha256")
    ):
        raise VN97P5E4FError("P5E4E adapter parent mismatch")
    if (
        payload.get("tokenizer_sha256")
        != repaired_meta.get("tokenizer_sha256")
    ):
        raise VN97P5E4FError("P5E4E adapter tokenizer mismatch")
    tokenizer_sha = _sha256(tokenizer_path)
    if tokenizer_sha != repaired_meta.get("tokenizer_sha256"):
        raise VN97P5E4FError("P5E4E copied tokenizer hash mismatch")

    state = payload.get("adapter_state_dict")
    if not isinstance(state, dict):
        raise VN97P5E4FError("P5E4E adapter state missing")
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
        "frozen_core_sha256": payload.get("frozen_core_sha256"),
        "status": payload.get("status"),
        "rank": int(payload["rank"]),
    }


def _fresh_seed(
    repaired_sha: str,
    adapter_sha: str,
) -> str:
    return hashlib.sha256(
        (
            P5E4F_PROFILE_ID
            + "\0"
            + repaired_sha
            + "\0"
            + adapter_sha
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

        totals["tokens"] += tokens
        totals["nll_sum"] += nll
        totals["top1_tokens"] += top1
        totals["top5_tokens"] += top5
        totals["first_rank_sum"] += first_rank
        totals["generation_passed"] += int(generation_ok)
        totals["generation_errors"] += int(bool(error_type))

        row = categories[probe.category]
        row["tasks"] = int(row["tasks"]) + 1
        row["tokens"] = int(row["tokens"]) + tokens
        row["nll_sum"] = float(row["nll_sum"]) + nll
        row["top1_tokens"] = int(row["top1_tokens"]) + top1
        row["top5_tokens"] = int(row["top5_tokens"]) + top5
        row["first_rank_sum"] = int(row["first_rank_sum"]) + first_rank
        row["generation_passed"] = int(
            row["generation_passed"]
        ) + int(generation_ok)
        row["generation_errors"] = int(
            row["generation_errors"]
        ) + int(bool(error_type))

        if index % 14 == 0 or index == len(probes):
            print(
                "VN97 P5E4F PROGRESS "
                f"model={label} tasks={index}/{len(probes)} "
                f"gen_passed={totals['generation_passed']} "
                f"mean_nll="
                f"{float(totals['nll_sum']) / max(int(totals['tokens']), 1):.6f} "
                f"top1="
                f"{int(totals['top1_tokens']) / max(int(totals['tokens']), 1):.6f}",
                flush=True,
            )

    if int(totals["tokens"]) <= 0:
        raise VN97P5E4FError("fresh output validation produced no targets")

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
    baseline: dict[str, Any],
    expanded: dict[str, Any],
) -> tuple[str, list[str], bool]:
    reasons: list[str] = []

    if float(expanded["mean_nll"]) > float(baseline["mean_nll"]) * 1.01:
        reasons.append("fresh_nll_over_1pct")
    if (
        float(expanded["token_top5_rate"]) + 0.005
        < float(baseline["token_top5_rate"])
    ):
        reasons.append("fresh_top5_drop_over_0.5pp")
    if (
        float(expanded["mean_first_rank"])
        > float(baseline["mean_first_rank"]) * 1.05
    ):
        reasons.append("fresh_first_rank_over_5pct")
    if int(expanded["generation_errors"]) > int(
        baseline["generation_errors"]
    ):
        reasons.append("fresh_more_generation_errors")

    for category in ("tool_intent", "authority_behavior"):
        base_cat = baseline["categories"][category]
        expanded_cat = expanded["categories"][category]
        if (
            float(expanded_cat["mean_nll"])
            > float(base_cat["mean_nll"]) * 1.03
        ):
            reasons.append(f"fresh_{category}_nll_over_3pct")

    if reasons:
        return "FRESH_OUTPUT_REJECTED_REGRESSION", reasons, False

    generation_gain = int(expanded["generation_passed"]) > int(
        baseline["generation_passed"]
    )
    qat_ready = (
        generation_gain
        and float(expanded["generation_pass_rate"])
        >= QAT_MIN_GENERATION_RATE
    )
    if generation_gain:
        return "FRESH_OUTPUT_GENERATION_GAIN", [], qat_ready

    latent_gain = (
        float(expanded["mean_nll"])
        <= float(baseline["mean_nll"]) * 0.99
        or float(expanded["token_top1_rate"])
        >= float(baseline["token_top1_rate"]) + 0.01
        or float(expanded["token_top5_rate"])
        >= float(baseline["token_top5_rate"]) + 0.01
        or float(expanded["mean_first_rank"])
        <= float(baseline["mean_first_rank"]) * 0.95
    )
    if latent_gain:
        return "FRESH_OUTPUT_LATENT_GAIN", [], False
    if (
        int(baseline["generation_passed"]) == 0
        and int(expanded["generation_passed"]) == 0
    ):
        return "FRESH_OUTPUT_ZERO_GENERATION", [], False
    return "FRESH_OUTPUT_PRESERVED", [], False


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise VN97P5E4FError("CUDA device requested but CUDA is unavailable")

    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise VN97P5E4FError(
            "output already exists; P5E4F is a one-shot validation"
        )

    base_model, tokenizer, repaired_meta = _load_repaired(
        Path(args.repaired_dir)
    )
    adapter, expanded_meta = _load_expansion(
        Path(args.expanded_dir),
        repaired_meta=repaired_meta,
        d_model=base_model.config.d_model,
        vocab_size=base_model.config.vocab_size,
    )
    expanded_model = VN97OutputExpandedModel(
        base_model,
        adapter,
    )

    seed_hex = _fresh_seed(
        str(repaired_meta["student_sha256"]),
        str(expanded_meta["adapter_sha256"]),
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
        "VN97 P5E4F FRESH "
        f"seed_sha256={seed_hex} "
        f"manifest_sha256={manifest_sha256} "
        f"tasks={len(probes)} "
        "reused_p4_suite=false "
        "reused_p5e4b_manifest=false "
        "reused_p5e4d_manifest=false "
        f"adapter_sha256={expanded_meta['adapter_sha256']}",
        flush=True,
    )

    baseline = _evaluate(
        label="p5e4c-baseline",
        model=base_model,
        tokenizer=tokenizer,
        probes=probes,
        device=args.device,
    )
    expanded = _evaluate(
        label="p5e4e-expanded",
        model=expanded_model,
        tokenizer=tokenizer,
        probes=probes,
        device=args.device,
    )
    status, reasons, qat_ready = _gate(
        baseline=baseline,
        expanded=expanded,
    )

    report = {
        "artifacts": {
            "repaired": repaired_meta,
            "expanded": expanded_meta,
        },
        "evaluations": {
            "baseline": baseline,
            "expanded": expanded,
        },
        "fresh_generation": {
            "manifest_sha256": manifest_sha256,
            "per_category": PER_CATEGORY,
            "reused_p4_suite": False,
            "reused_p5e4b_manifest": False,
            "reused_p5e4d_manifest": False,
            "seed_sha256": seed_hex,
            "task_count": len(probes),
        },
        "one_shot": True,
        "profile_id": P5E4F_PROFILE_ID,
        "qat_min_generation_rate": QAT_MIN_GENERATION_RATE,
        "ready_for_decoding_repair": (
            not qat_ready
            and status
            in {
                "FRESH_OUTPUT_LATENT_GAIN",
                "FRESH_OUTPUT_ZERO_GENERATION",
                "FRESH_OUTPUT_PRESERVED",
                "FRESH_OUTPUT_GENERATION_GAIN",
            }
        ),
        "ready_for_qat": qat_ready,
        "reasons": reasons,
        "schema": P5E4F_SCHEMA,
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
        "VN97P5E4F "
        f"status={status} "
        f"baseline_nll={baseline['mean_nll']:.6f} "
        f"expanded_nll={expanded['mean_nll']:.6f} "
        f"baseline_top1={baseline['token_top1_rate']:.6f} "
        f"expanded_top1={expanded['token_top1_rate']:.6f} "
        f"baseline_top5={baseline['token_top5_rate']:.6f} "
        f"expanded_top5={expanded['token_top5_rate']:.6f} "
        f"baseline_first_rank={baseline['mean_first_rank']:.2f} "
        f"expanded_first_rank={expanded['mean_first_rank']:.2f} "
        f"baseline_generation="
        f"{baseline['generation_passed']}/{baseline['tasks']} "
        f"expanded_generation="
        f"{expanded['generation_passed']}/{expanded['tasks']} "
        f"ready_for_qat={str(qat_ready).lower()} "
        f"ready_for_decoding_repair="
        f"{str(report['ready_for_decoding_repair']).lower()} "
        f"reasons={','.join(reasons) if reasons else 'none'} "
        f"report={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-p5e4f-validate: {exc}", file=sys.stderr)
        raise
