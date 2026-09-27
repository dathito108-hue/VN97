from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

from .chat_boundary import recover_chat_response_text
from .cognition_adapter import TorchVN97InferenceEngine, VN97InferenceLimits
from .p4_task_evaluation import P4_CATEGORIES, load_p4_task_suite, score_p4_output
from .p5e3_heldout_validation_cli import _load_artifact
from .training import VN97ChatMessage
from .training_cli import _atomic_write


P5E3B_SCHEMA = "VN97P5E3B"
P5E3B_PROFILE_ID = "vn97-p5e3b-canonical-chat-heldout-validation-v1"
PROGRESS_INTERVAL = 10


class VN97P5E3BError(RuntimeError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run P5E3 on the canonical segmented chat-completion prefix "
            "used by VN97 training, scoring both raw and boundary-recovered "
            "outputs. This distinguishes a decoder/prefix failure from a "
            "real capability regression."
        )
    )
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--p5e1-dir", required=True)
    parser.add_argument("--p5e2-dir", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


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
    engine = TorchVN97InferenceEngine(
        model,
        tokenizer,
        limits=VN97InferenceLimits(
            max_prompt_tokens=4096,
            repetition_penalty=1.12,
            no_repeat_ngram_size=4,
        ),
    )

    categories = {
        category: {
            "tasks": 0,
            "raw_passed": 0,
            "recovered_passed": 0,
            "errors": 0,
        }
        for category in P4_CATEGORIES
    }
    rows: list[dict[str, Any]] = []
    started = time.monotonic()

    for index, task in enumerate(suite.tasks, start=1):
        raw = ""
        recovered = ""
        error_type = ""
        try:
            raw = engine.generate_chat_completion(
                (
                    VN97ChatMessage(
                        role="user",
                        content=task.prompt,
                    ),
                ),
                max_new_tokens=task.max_new_tokens,
            )
            recovered = recover_chat_response_text(raw)
            raw_ok = bool(score_p4_output(task, raw))
            recovered_ok = bool(score_p4_output(task, recovered))
        except Exception as exc:
            raw_ok = False
            recovered_ok = False
            error_type = type(exc).__name__

        category = categories[task.category]
        category["tasks"] += 1
        category["raw_passed"] += int(raw_ok)
        category["recovered_passed"] += int(recovered_ok)
        category["errors"] += int(bool(error_type))

        raw_bytes = raw.encode("utf-8")
        recovered_bytes = recovered.encode("utf-8")
        rows.append(
            {
                "category": task.category,
                "error_type": error_type,
                "raw_output_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "raw_output_utf8_bytes": len(raw_bytes),
                "raw_passed": raw_ok,
                "recovered_output_sha256": hashlib.sha256(
                    recovered_bytes
                ).hexdigest(),
                "recovered_output_utf8_bytes": len(recovered_bytes),
                "recovered_passed": recovered_ok,
                "task_id": task.task_id,
            }
        )

        if index % PROGRESS_INTERVAL == 0 or index == len(suite.tasks):
            raw_passed = sum(int(row["raw_passed"]) for row in rows)
            recovered_passed = sum(
                int(row["recovered_passed"]) for row in rows
            )
            print(
                "VN97 P5E3B PROGRESS "
                f"model={label} "
                f"tasks={index}/{len(suite.tasks)} "
                f"raw={raw_passed} "
                f"recovered={recovered_passed} "
                f"elapsed_s={time.monotonic() - started:.1f}",
                flush=True,
            )

    normalized_categories: dict[str, dict[str, Any]] = {}
    for category, row in categories.items():
        tasks = int(row["tasks"])
        normalized_categories[category] = {
            "errors": int(row["errors"]),
            "raw_pass_rate": (
                int(row["raw_passed"]) / tasks if tasks else 0.0
            ),
            "raw_passed": int(row["raw_passed"]),
            "recovered_pass_rate": (
                int(row["recovered_passed"]) / tasks if tasks else 0.0
            ),
            "recovered_passed": int(row["recovered_passed"]),
            "tasks": tasks,
        }

    raw_passed = sum(int(row["raw_passed"]) for row in rows)
    recovered_passed = sum(int(row["recovered_passed"]) for row in rows)
    errors = sum(int(bool(row["error_type"])) for row in rows)

    model.cpu()
    torch.cuda.empty_cache()

    return {
        "categories": normalized_categories,
        "errors": errors,
        "raw_pass_rate": raw_passed / len(rows),
        "raw_passed": raw_passed,
        "recovered_pass_rate": recovered_passed / len(rows),
        "recovered_passed": recovered_passed,
        "results": rows,
        "tasks": len(rows),
    }


def _gate(
    base: dict[str, Any],
    p5e1: dict[str, Any],
    p5e2: dict[str, Any],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    guarded = ("tool_intent", "authority_behavior")

    if int(p5e2["recovered_passed"]) == 0:
        return "INCONCLUSIVE_ZERO_SIGNAL", ["p5e2_recovered_zero"]

    if int(p5e2["recovered_passed"]) < int(base["recovered_passed"]):
        reasons.append("overall_recovered_below_base")
    if int(p5e2["errors"]) > int(base["errors"]):
        reasons.append("more_runtime_errors_than_base")

    for category in guarded:
        candidate = int(
            p5e2["categories"][category]["recovered_passed"]
        )
        baseline = int(
            base["categories"][category]["recovered_passed"]
        )
        if candidate < baseline:
            reasons.append(f"{category}_recovered_below_base")

    if reasons:
        return "REJECTED_HELD_OUT_REGRESSION", reasons

    strongest_previous = max(
        int(base["recovered_passed"]),
        int(p5e1["recovered_passed"]),
    )
    if int(p5e2["recovered_passed"]) > strongest_previous:
        return "HELD_OUT_GAIN", []
    return "HELD_OUT_NO_REGRESSION", []


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise VN97P5E3BError("CUDA device requested but CUDA is unavailable")

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
        print(f"VN97 P5E3B LOAD model={label}", flush=True)
        model, tokenizer, meta = _load_artifact(
            artifact_dirs[label],
            label=label,
        )
        current_hash = str(meta["tokenizer_sha256"])
        if tokenizer_hash is None:
            tokenizer_hash = current_hash
        elif current_hash != tokenizer_hash:
            raise VN97P5E3BError(
                "P5E3B artifacts do not share one tokenizer"
            )

        metadata[label] = meta
        evaluations[label] = _evaluate(
            label=label,
            model=model,
            tokenizer=tokenizer,
            suite=suite,
            device=args.device,
        )
        print(
            "VN97 P5E3B MODEL "
            f"model={label} "
            f"raw={evaluations[label]['raw_passed']}/"
            f"{evaluations[label]['tasks']} "
            f"recovered={evaluations[label]['recovered_passed']}/"
            f"{evaluations[label]['tasks']} "
            f"errors={evaluations[label]['errors']}",
            flush=True,
        )
        del model, tokenizer
        torch.cuda.empty_cache()

    status, reasons = _gate(
        evaluations["base"],
        evaluations["p5e1"],
        evaluations["p5e2"],
    )

    report = {
        "artifacts": metadata,
        "decoder": {
            "boundary_recovery": True,
            "canonical_segmented_chat_prefix": True,
            "legacy_whole_string_prefix": False,
        },
        "evaluations": evaluations,
        "gate": {
            "guarded_categories": ["tool_intent", "authority_behavior"],
            "reasons": reasons,
        },
        "profile_id": P5E3B_PROFILE_ID,
        "schema": P5E3B_SCHEMA,
        "status": status,
        "suite_sha256": suite.suite_sha256,
        "task_count": len(suite.tasks),
        "tokenizer_sha256": tokenizer_hash,
    }
    data = (
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise VN97P5E3BError("output path must not already exist")
    _atomic_write(output, data)

    print(
        "VN97P5E3B "
        f"status={status} "
        f"base_raw={evaluations['base']['raw_passed']}/"
        f"{evaluations['base']['tasks']} "
        f"base_recovered={evaluations['base']['recovered_passed']}/"
        f"{evaluations['base']['tasks']} "
        f"p5e1_raw={evaluations['p5e1']['raw_passed']}/"
        f"{evaluations['p5e1']['tasks']} "
        f"p5e1_recovered={evaluations['p5e1']['recovered_passed']}/"
        f"{evaluations['p5e1']['tasks']} "
        f"p5e2_raw={evaluations['p5e2']['raw_passed']}/"
        f"{evaluations['p5e2']['tasks']} "
        f"p5e2_recovered={evaluations['p5e2']['recovered_passed']}/"
        f"{evaluations['p5e2']['tasks']} "
        f"reasons={','.join(reasons) if reasons else 'none'} "
        f"report={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-p5e3b-validate: {exc}", file=sys.stderr)
        raise
