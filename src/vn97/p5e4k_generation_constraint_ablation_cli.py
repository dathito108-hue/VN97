from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

from .chat_boundary import recover_chat_response_text
from .cognition_adapter import TorchVN97InferenceEngine, VN97InferenceLimits
from .p5e4b_fresh_validation_cli import _build_probes
from .p5e4d_fresh_decoder_validation_cli import _load_repaired
from .p5e4f_fresh_output_validation_cli import (
    VN97OutputExpandedModel,
    _load_expansion,
)
from .p5e4h_fresh_post_prefix_validation_cli import _load_prefix_repaired
from .p5e4j_autoregressive_parity_diagnostic_cli import P5E4J_SCHEMA
from .training import VN97ChatMessage
from .training_cli import _atomic_write


P5E4K_SCHEMA = "VN97P5E4K"
P5E4K_PROFILE_ID = "vn97-p5e4k-generation-constraint-ablation-v1"
PER_CATEGORY = 16
MIN_PROMOTION_RATE = 0.10
MIN_ABSOLUTE_GAIN = 4

POLICIES = (
    ("production_current", 1.12, 4),
    ("no_ngram_block", 1.12, 0),
    ("no_repetition_penalty", 1.0, 4),
    ("pure_greedy", 1.0, 0),
)


class VN97P5E4KError(RuntimeError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "P5E4K generation-constraint ablation. Evaluate the accepted P5E4G "
            "candidate on a fresh manifest under four decoding policies to "
            "determine whether repetition/no-repeat constraints are the direct "
            "cause of the exact-generation failure diagnosed by P5E4J."
        )
    )
    parser.add_argument("--repaired-dir", required=True)
    parser.add_argument("--expanded-dir", required=True)
    parser.add_argument("--prefix-repaired-dir", required=True)
    parser.add_argument("--p5e4j-report", required=True)
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


def _verify_p5e4j(
    path: Path,
    *,
    repaired_meta: dict[str, Any],
    prefix_meta: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    resolved = path.resolve(strict=True)
    raw = resolved.read_bytes()
    report = json.loads(raw.decode("utf-8"))
    if not isinstance(report, dict) or report.get("schema") != P5E4J_SCHEMA:
        raise VN97P5E4KError("invalid P5E4J report schema")
    if report.get("one_shot") is not True:
        raise VN97P5E4KError("P5E4J evidence is not one-shot")
    if report.get("ready_for_qat") is not False:
        raise VN97P5E4KError("P5E4K is only valid before QAT")
    if report.get("status") != "DECODING_CONSTRAINT_INTERFERENCE":
        raise VN97P5E4KError(
            "P5E4J did not diagnose decoding constraint interference"
        )
    if report.get("recommended_next_stage") != "generation_constraint_repair":
        raise VN97P5E4KError("P5E4J next-stage recommendation mismatch")
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
        raise VN97P5E4KError("P5E4J artifact identities missing")
    if repaired.get("student_sha256") != repaired_meta.get("student_sha256"):
        raise VN97P5E4KError("P5E4J/P5E4C checkpoint identity mismatch")
    if prefix.get("adapter_sha256") != prefix_meta.get("adapter_sha256"):
        raise VN97P5E4KError("P5E4J/P5E4G adapter identity mismatch")
    return report, hashlib.sha256(raw).hexdigest()


def _char_prefix_ratio(generated: str, target: str) -> float:
    if not target:
        return 1.0 if not generated else 0.0
    length = 0
    limit = min(len(generated), len(target))
    while length < limit and generated[length] == target[length]:
        length += 1
    return length / len(target)


@torch.inference_mode()
def _evaluate_policy(
    *,
    model,
    tokenizer,
    probes,
    device: str,
    name: str,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> dict[str, Any]:
    model.to(device)
    model.eval()
    engine = TorchVN97InferenceEngine(
        model,
        tokenizer,
        limits=VN97InferenceLimits(
            max_prompt_tokens=4096,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        ),
    )

    exact = 0
    errors = 0
    nonempty = 0
    prefix_sum = 0.0
    output_length_sum = 0
    target_length_sum = 0
    category_exact: dict[str, int] = {}
    category_tasks: dict[str, int] = {}

    for index, probe in enumerate(probes, start=1):
        category_tasks[probe.category] = (
            category_tasks.get(probe.category, 0) + 1
        )
        raw = ""
        recovered = ""
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
        except Exception as exc:
            error_type = type(exc).__name__

        ok = not error_type and recovered == probe.target
        exact += int(ok)
        errors += int(bool(error_type))
        nonempty += int(bool(recovered))
        prefix_sum += _char_prefix_ratio(recovered, probe.target)
        output_length_sum += len(recovered)
        target_length_sum += len(probe.target)
        if ok:
            category_exact[probe.category] = (
                category_exact.get(probe.category, 0) + 1
            )

        if index % 16 == 0 or index == len(probes):
            print(
                "VN97 P5E4K PROGRESS "
                f"policy={name} tasks={index}/{len(probes)} "
                f"exact={exact} "
                f"prefix_ratio={prefix_sum / index:.6f} "
                f"errors={errors}",
                flush=True,
            )

    tasks = len(probes)
    return {
        "name": name,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "tasks": tasks,
        "exact_generation": exact,
        "exact_generation_rate": exact / tasks,
        "generation_errors": errors,
        "nonempty_generation": nonempty,
        "nonempty_generation_rate": nonempty / tasks,
        "mean_char_prefix_ratio": prefix_sum / tasks,
        "mean_output_chars": output_length_sum / tasks,
        "mean_target_chars": target_length_sum / tasks,
        "category_exact": {
            category: category_exact.get(category, 0)
            for category in sorted(category_tasks)
        },
        "category_tasks": {
            category: category_tasks[category]
            for category in sorted(category_tasks)
        },
    }


def _select_best(results: list[dict[str, Any]]) -> dict[str, Any]:
    current = results[0]
    return max(
        results,
        key=lambda item: (
            int(item["exact_generation"]),
            float(item["mean_char_prefix_ratio"]),
            -int(item["generation_errors"]),
            1 if item["name"] == current["name"] else 0,
        ),
    )


def _decide(
    results: list[dict[str, Any]],
) -> tuple[str, bool, str]:
    current = results[0]
    best = _select_best(results)
    gain = int(best["exact_generation"]) - int(
        current["exact_generation"]
    )

    if (
        best["name"] != current["name"]
        and gain >= MIN_ABSOLUTE_GAIN
        and float(best["exact_generation_rate"]) >= MIN_PROMOTION_RATE
        and int(best["generation_errors"]) <= int(
            current["generation_errors"]
        )
    ):
        return (
            "DECODING_POLICY_GAIN",
            True,
            str(best["name"]),
        )

    if best["name"] != current["name"] and (
        float(best["mean_char_prefix_ratio"])
        >= float(current["mean_char_prefix_ratio"]) + 0.05
    ):
        return (
            "DECODING_POLICY_PREFIX_GAIN_ONLY",
            False,
            str(best["name"]),
        )

    return (
        "DECODING_POLICY_NO_MATERIAL_GAIN",
        False,
        str(current["name"]),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise VN97P5E4KError("CUDA device requested but CUDA is unavailable")

    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise VN97P5E4KError(
            "output already exists; P5E4K is a one-shot ablation"
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
    p5e4j_report, p5e4j_sha = _verify_p5e4j(
        Path(args.p5e4j_report),
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
            P5E4K_PROFILE_ID
            + "\0"
            + str(repaired_meta["student_sha256"])
            + "\0"
            + str(prefix_meta["adapter_sha256"])
            + "\0"
            + p5e4j_sha
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

    diagnostic = p5e4j_report.get("diagnostic", {})
    print(
        "VN97 P5E4K START "
        f"p5e4j_status={p5e4j_report['status']} "
        f"constraint_flips={diagnostic.get('constraint_flip_prod')} "
        f"target_banned={diagnostic.get('target_banned_prod')} "
        f"runtime_agreement={diagnostic.get('constrained_top1_agreement')} "
        f"seed_sha256={seed_hex} "
        f"manifest_sha256={manifest_sha256} "
        f"tasks={len(probes)} "
        "ready_for_qat=false",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    for name, repetition_penalty, no_repeat_ngram_size in POLICIES:
        results.append(
            _evaluate_policy(
                model=model,
                tokenizer=tokenizer,
                probes=probes,
                device=args.device,
                name=name,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        )

    status, ready_for_policy_promotion, selected_policy = _decide(
        results
    )
    selected = next(
        item for item in results
        if item["name"] == selected_policy
    )
    current = results[0]

    report = {
        "artifacts": {
            "repaired_core": repaired_meta,
            "parent_output": parent_meta,
            "prefix_repaired_output": prefix_meta,
        },
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
            "reused_p5e4j_manifest": False,
        },
        "one_shot": True,
        "p5e4j_report_sha256": p5e4j_sha,
        "p5e4j_status": p5e4j_report["status"],
        "policies": results,
        "profile_id": P5E4K_PROFILE_ID,
        "ready_for_policy_promotion": ready_for_policy_promotion,
        "ready_for_qat": False,
        "selected_policy": selected_policy,
        "schema": P5E4K_SCHEMA,
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
        "VN97P5E4K "
        f"status={status} "
        f"current_exact={current['exact_generation']}/{current['tasks']} "
        f"current_prefix={current['mean_char_prefix_ratio']:.6f} "
        f"selected_policy={selected_policy} "
        f"selected_exact={selected['exact_generation']}/{selected['tasks']} "
        f"selected_prefix={selected['mean_char_prefix_ratio']:.6f} "
        f"selected_repetition_penalty={selected['repetition_penalty']} "
        f"selected_no_repeat_ngram_size={selected['no_repeat_ngram_size']} "
        f"ready_for_policy_promotion="
        f"{str(ready_for_policy_promotion).lower()} "
        "ready_for_qat=false "
        f"report={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-p5e4k-ablate: {exc}", file=sys.stderr)
        raise
