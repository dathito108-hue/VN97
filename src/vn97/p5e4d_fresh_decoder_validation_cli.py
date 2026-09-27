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
from .config import VN97Config
from .model import VN97LanguageCore
from .p4_task_evaluation import P4_CATEGORIES
from .p5e3c_target_rank_diagnostic_cli import _score_target
from .p5e4b_fresh_validation_cli import _build_probes, _load_candidate
from .p5e4c_decoder_repair_cli import (
    P5E4C_FLOAT_SCHEMA,
    _state_hash,
)
from .quantization import set_float_shadow_mode
from .tokenizer import VN97Tokenizer, VN97TokenizerPackage
from .training import VN97ChatMessage
from .training_cli import _atomic_write


P5E4D_SCHEMA = "VN97P5E4D"
P5E4D_PROFILE_ID = "vn97-p5e4d-fresh-decoder-validation-v1"
PER_CATEGORY = 12
QAT_MIN_GENERATION_RATE = 0.10


class VN97P5E4DError(RuntimeError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fresh one-shot validation for P5E4C final-norm decoder repair. "
            "Compares the frozen P5E4A parent and repaired P5E4C candidate on "
            "a newly generated probe manifest that is not reused from P5E4B."
        )
    )
    parser.add_argument("--parent-dir", required=True)
    parser.add_argument("--repaired-dir", required=True)
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


def _load_repaired(
    root: Path,
) -> tuple[VN97LanguageCore, VN97Tokenizer, dict[str, Any]]:
    resolved = root.resolve(strict=True)
    model_path = resolved / "student-float.pt"
    tokenizer_path = resolved / "tokenizer.vn97tk1"
    report_path = resolved / "p5e4c-report.json"
    for path in (model_path, tokenizer_path, report_path):
        if not path.is_file():
            raise VN97P5E4DError(f"repaired artifact missing {path.name}")

    payload = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != P5E4C_FLOAT_SCHEMA
        or payload.get("float_shadow_required") is not True
        or payload.get("status") != "DECODER_REPAIR_SIGNAL"
        or payload.get("fresh_holdout_required") is not True
        or payload.get("ready_for_qat") is not False
    ):
        raise VN97P5E4DError("P5E4C repaired artifact contract mismatch")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict)
        or report.get("status") != "DECODER_REPAIR_SIGNAL"
        or report.get("ready_for_fresh_validation") is not True
        or report.get("ready_for_qat") is not False
    ):
        raise VN97P5E4DError("P5E4C report contract mismatch")

    config_raw = payload.get("config")
    state_dict = payload.get("state_dict")
    if not isinstance(config_raw, dict) or not isinstance(state_dict, dict):
        raise VN97P5E4DError("P5E4C payload is incomplete")

    config = VN97Config(**config_raw)
    if (
        config.d_model != 1536
        or config.n_layers != 32
        or config.d_state != 16
    ):
        raise VN97P5E4DError("P5E4C is not canonical 309M VN97")

    tokenizer_bytes = tokenizer_path.read_bytes()
    tokenizer_sha = hashlib.sha256(tokenizer_bytes).hexdigest()
    if payload.get("tokenizer_sha256") != tokenizer_sha:
        raise VN97P5E4DError("P5E4C tokenizer hash mismatch")

    package = VN97TokenizerPackage.from_bytes(tokenizer_bytes)
    if package.vocab_size != config.vocab_size:
        raise VN97P5E4DError("P5E4C tokenizer/model vocabulary mismatch")

    model = VN97LanguageCore(config)
    set_float_shadow_mode(model, True)
    model.load_state_dict(state_dict)

    return (
        model,
        VN97Tokenizer(package),
        {
            "root": str(resolved),
            "student_sha256": _sha256(model_path),
            "tokenizer_sha256": tokenizer_sha,
            "status": payload.get("status"),
            "parent_candidate_sha256": payload.get(
                "parent_candidate_sha256"
            ),
            "immutable_state_sha256": payload.get(
                "immutable_state_sha256"
            ),
            "final_norm_before_sha256": report.get(
                "final_norm_before_sha256"
            ),
            "final_norm_after_sha256": report.get(
                "final_norm_after_sha256"
            ),
        },
    )


def _fresh_seed(
    parent_sha: str,
    repaired_sha: str,
) -> str:
    return hashlib.sha256(
        (
            P5E4D_PROFILE_ID
            + "\0"
            + parent_sha
            + "\0"
            + repaired_sha
        ).encode("utf-8")
    ).hexdigest()


@torch.inference_mode()
def _evaluate(
    *,
    label: str,
    model: VN97LanguageCore,
    tokenizer: VN97Tokenizer,
    probes,
    device: str,
) -> dict[str, Any]:
    model.to(device)
    model.eval()
    set_float_shadow_mode(model, True)

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
        recovered = ""
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
        top5 = int(latent["top5_tokens"])
        first_rank = int(latent["first_rank"])

        totals["tokens"] += tokens
        totals["nll_sum"] += nll
        totals["top5_tokens"] += top5
        totals["first_rank_sum"] += first_rank
        totals["generation_passed"] += int(generation_ok)
        totals["generation_errors"] += int(bool(error_type))

        row = categories[probe.category]
        row["tasks"] = int(row["tasks"]) + 1
        row["tokens"] = int(row["tokens"]) + tokens
        row["nll_sum"] = float(row["nll_sum"]) + nll
        row["top5_tokens"] = int(row["top5_tokens"]) + top5
        row["first_rank_sum"] = int(row["first_rank_sum"]) + first_rank
        row["generation_passed"] = int(
            row["generation_passed"]
        ) + int(generation_ok)
        row["generation_errors"] = int(
            row["generation_errors"]
        ) + int(bool(error_type))

        if index % 12 == 0 or index == len(probes):
            print(
                "VN97 P5E4D PROGRESS "
                f"model={label} "
                f"tasks={index}/{len(probes)} "
                f"gen_passed={totals['generation_passed']} "
                f"mean_nll="
                f"{float(totals['nll_sum']) / max(int(totals['tokens']), 1):.6f}",
                flush=True,
            )

    if int(totals["tokens"]) <= 0:
        raise VN97P5E4DError("fresh decoder validation produced no targets")

    normalized: dict[str, dict[str, float | int]] = {}
    for category in P4_CATEGORIES:
        row = categories[category]
        tasks = int(row["tasks"])
        tokens = int(row["tokens"])
        normalized[category] = {
            "tasks": tasks,
            "tokens": tokens,
            "mean_nll": float(row["nll_sum"]) / max(tokens, 1),
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

    if float(repaired["mean_nll"]) > float(parent["mean_nll"]) * 1.01:
        reasons.append("fresh_nll_over_1pct")
    if (
        float(repaired["token_top5_rate"]) + 0.005
        < float(parent["token_top5_rate"])
    ):
        reasons.append("fresh_top5_drop_over_0.5pp")
    if (
        float(repaired["mean_first_rank"])
        > float(parent["mean_first_rank"]) * 1.05
    ):
        reasons.append("fresh_first_rank_over_5pct")
    if int(repaired["generation_errors"]) > int(parent["generation_errors"]):
        reasons.append("fresh_more_generation_errors")

    for category in ("tool_intent", "authority_behavior"):
        parent_cat = parent["categories"][category]
        repaired_cat = repaired["categories"][category]
        if (
            float(repaired_cat["mean_nll"])
            > float(parent_cat["mean_nll"]) * 1.03
        ):
            reasons.append(f"fresh_{category}_nll_over_3pct")

    if reasons:
        return "FRESH_DECODER_REJECTED_REGRESSION", reasons, False

    generation_gain = int(repaired["generation_passed"]) > int(
        parent["generation_passed"]
    )
    qat_ready = (
        generation_gain
        and float(repaired["generation_pass_rate"])
        >= QAT_MIN_GENERATION_RATE
    )

    if generation_gain:
        return "FRESH_DECODER_GENERATION_GAIN", [], qat_ready

    latent_gain = (
        float(repaired["mean_nll"])
        <= float(parent["mean_nll"]) * 0.99
        or float(repaired["token_top5_rate"])
        >= float(parent["token_top5_rate"]) + 0.01
        or float(repaired["mean_first_rank"])
        <= float(parent["mean_first_rank"]) * 0.95
    )
    if latent_gain:
        return "FRESH_DECODER_LATENT_GAIN", [], False

    if (
        int(parent["generation_passed"]) == 0
        and int(repaired["generation_passed"]) == 0
    ):
        return "FRESH_DECODER_PRESERVED_ZERO_GENERATION", [], False

    return "FRESH_DECODER_PRESERVED", [], False


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise VN97P5E4DError("CUDA device requested but CUDA is unavailable")

    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise VN97P5E4DError(
            "output already exists; P5E4D is a one-shot validation"
        )

    parent_model, parent_tokenizer, parent_meta = _load_candidate(
        Path(args.parent_dir)
    )
    repaired_model, repaired_tokenizer, repaired_meta = _load_repaired(
        Path(args.repaired_dir)
    )

    if (
        parent_meta["tokenizer_sha256"]
        != repaired_meta["tokenizer_sha256"]
    ):
        raise VN97P5E4DError("parent/repaired tokenizer mismatch")
    if parent_model.config != repaired_model.config:
        raise VN97P5E4DError("parent/repaired config mismatch")
    if (
        repaired_meta["parent_candidate_sha256"]
        != parent_meta["student_sha256"]
    ):
        raise VN97P5E4DError("P5E4C parent checkpoint identity mismatch")

    parent_immutable = _state_hash(
        parent_model,
        exclude_final_norm=True,
    )
    repaired_immutable = _state_hash(
        repaired_model,
        exclude_final_norm=True,
    )
    if parent_immutable != repaired_immutable:
        raise VN97P5E4DError(
            "P5E4C changed frozen SSM/embedding state"
        )
    if (
        repaired_meta["immutable_state_sha256"]
        and repaired_meta["immutable_state_sha256"]
        != repaired_immutable
    ):
        raise VN97P5E4DError(
            "P5E4C immutable-state evidence mismatch"
        )

    seed_hex = _fresh_seed(
        str(parent_meta["student_sha256"]),
        str(repaired_meta["student_sha256"]),
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
        "VN97 P5E4D FRESH "
        f"seed_sha256={seed_hex} "
        f"manifest_sha256={manifest_sha256} "
        f"tasks={len(probes)} "
        f"reused_p4_suite=false "
        f"reused_p5e4b_manifest=false "
        f"immutable_state_match=true",
        flush=True,
    )

    parent = _evaluate(
        label="p5e4a-parent",
        model=parent_model,
        tokenizer=parent_tokenizer,
        probes=probes,
        device=args.device,
    )
    repaired = _evaluate(
        label="p5e4c-repaired",
        model=repaired_model,
        tokenizer=repaired_tokenizer,
        probes=probes,
        device=args.device,
    )
    status, reasons, qat_ready = _gate(
        parent=parent,
        repaired=repaired,
    )

    report = {
        "artifacts": {
            "parent": parent_meta,
            "repaired": repaired_meta,
        },
        "evaluations": {
            "parent": parent,
            "repaired": repaired,
        },
        "fresh_generation": {
            "manifest_sha256": manifest_sha256,
            "per_category": PER_CATEGORY,
            "reused_p4_suite": False,
            "reused_p5e4b_manifest": False,
            "seed_sha256": seed_hex,
            "task_count": len(probes),
        },
        "immutable_state_match": True,
        "one_shot": True,
        "profile_id": P5E4D_PROFILE_ID,
        "qat_min_generation_rate": QAT_MIN_GENERATION_RATE,
        "ready_for_output_expansion": (
            status
            in {
                "FRESH_DECODER_LATENT_GAIN",
                "FRESH_DECODER_PRESERVED_ZERO_GENERATION",
                "FRESH_DECODER_PRESERVED",
            }
            and not qat_ready
        ),
        "ready_for_qat": qat_ready,
        "reasons": reasons,
        "schema": P5E4D_SCHEMA,
        "status": status,
    }

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
        "VN97P5E4D "
        f"status={status} "
        f"parent_nll={parent['mean_nll']:.6f} "
        f"repaired_nll={repaired['mean_nll']:.6f} "
        f"parent_top5={parent['token_top5_rate']:.6f} "
        f"repaired_top5={repaired['token_top5_rate']:.6f} "
        f"parent_first_rank={parent['mean_first_rank']:.2f} "
        f"repaired_first_rank={repaired['mean_first_rank']:.2f} "
        f"parent_generation={parent['generation_passed']}/{parent['tasks']} "
        f"repaired_generation={repaired['generation_passed']}/{repaired['tasks']} "
        f"ready_for_qat={str(qat_ready).lower()} "
        f"ready_for_output_expansion="
        f"{str(report['ready_for_output_expansion']).lower()} "
        f"reasons={','.join(reasons) if reasons else 'none'} "
        f"report={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-p5e4d-validate: {exc}", file=sys.stderr)
        raise
