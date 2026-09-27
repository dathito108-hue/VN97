from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

from .cognition_adapter import TorchVN97InferenceEngine, VN97InferenceLimits
from .config import VN97Config
from .model import VN97LanguageCore
from .p4_task_evaluation import (
    P4_CATEGORIES,
    load_p4_task_suite,
    render_p4_chat_prompt,
    score_p4_output,
)
from .p5d3_relational_distillation import P5D3B_FLOAT_ARTIFACT_SCHEMA
from .p5e_direct_ssm_transplant_cli import P5E1_FLOAT_SCHEMA
from .p5e2_calibration_cli import P5E2_FLOAT_SCHEMA
from .quantization import set_float_shadow_mode
from .tokenizer import VN97Tokenizer, VN97TokenizerPackage
from .training_cli import _atomic_write


P5E3_SCHEMA = "VN97P5E3"
P5E3_PROFILE_ID = "vn97-p5e3-held-out-capability-validation-v1"
PROGRESS_INTERVAL = 10


class VN97P5E3Error(RuntimeError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the canonical P5D3B base, P5E1 direct SSM transplant, "
            "and P5E2 calibrated transplant on the held-out P4 task suite."
        )
    )
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--p5e1-dir", required=True)
    parser.add_argument("--p5e2-dir", required=True)
    parser.add_argument("--suite", required=True)
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


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VN97P5E3Error(f"{path.name} must contain a JSON object")
    return value


def _artifact_contract(
    label: str,
) -> tuple[str, str | None, tuple[str, ...]]:
    if label == "base":
        return (
            P5D3B_FLOAT_ARTIFACT_SCHEMA,
            None,
            ("student-float.pt", "tokenizer.vn97tk1"),
        )
    if label == "p5e1":
        return (
            P5E1_FLOAT_SCHEMA,
            "DIRECT_SSM_TRANSPLANT_READY_FOR_CALIBRATION",
            ("student-float.pt", "tokenizer.vn97tk1", "p5e1-report.json"),
        )
    if label == "p5e2":
        return (
            P5E2_FLOAT_SCHEMA,
            None,
            ("student-float.pt", "tokenizer.vn97tk1", "p5e2-report.json"),
        )
    raise AssertionError("unreachable artifact label")


def _load_artifact(
    root: Path,
    *,
    label: str,
) -> tuple[VN97LanguageCore, VN97Tokenizer, dict[str, Any]]:
    resolved = root.resolve(strict=True)
    expected_schema, required_status, required_files = _artifact_contract(label)
    for filename in required_files:
        if not (resolved / filename).is_file():
            raise VN97P5E3Error(f"{label} artifact missing {filename}")

    student_path = resolved / "student-float.pt"
    tokenizer_path = resolved / "tokenizer.vn97tk1"
    payload = torch.load(
        student_path,
        map_location="cpu",
        weights_only=False,
    )
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != expected_schema
        or payload.get("float_shadow_required") is not True
    ):
        raise VN97P5E3Error(f"{label} float artifact schema mismatch")

    config_raw = payload.get("config")
    state_dict = payload.get("state_dict")
    if not isinstance(config_raw, dict) or not isinstance(state_dict, dict):
        raise VN97P5E3Error(f"{label} float artifact is incomplete")
    config = VN97Config(**config_raw)
    if (
        config.d_model != 1536
        or config.n_layers != 32
        or config.d_state != 16
    ):
        raise VN97P5E3Error(f"{label} is not canonical 309M VN97")

    tokenizer_bytes = tokenizer_path.read_bytes()
    tokenizer_sha = hashlib.sha256(tokenizer_bytes).hexdigest()
    embedded_tokenizer_sha = payload.get("tokenizer_sha256")
    if (
        embedded_tokenizer_sha is not None
        and embedded_tokenizer_sha != tokenizer_sha
    ):
        raise VN97P5E3Error(f"{label} tokenizer hash mismatch")
    package = VN97TokenizerPackage.from_bytes(tokenizer_bytes)
    if package.vocab_size != config.vocab_size:
        raise VN97P5E3Error(f"{label} tokenizer/model vocabulary mismatch")

    report: dict[str, Any] = {}
    if label == "p5e1":
        report = _load_json(resolved / "p5e1-report.json")
    elif label == "p5e2":
        report = _load_json(resolved / "p5e2-report.json")

    if required_status is not None and report.get("status") != required_status:
        raise VN97P5E3Error(
            f"{label} report status must be {required_status}"
        )
    if label == "p5e2" and report.get("status") not in {
        "CALIBRATION_SIGNAL",
        "CALIBRATION_STABLE",
    }:
        raise VN97P5E3Error(
            "P5E2 artifact was not accepted by the calibration gate"
        )

    model = VN97LanguageCore(config)
    set_float_shadow_mode(model, True)
    model.load_state_dict(state_dict)

    metadata = {
        "root": str(resolved),
        "student_sha256": _sha256(student_path),
        "tokenizer_sha256": tokenizer_sha,
        "report_status": report.get("status"),
    }
    return model, VN97Tokenizer(package), metadata


def _evaluate(
    *,
    label: str,
    model: VN97LanguageCore,
    tokenizer: VN97Tokenizer,
    suite,
    device: str,
) -> dict[str, Any]:
    model.to(device)
    set_float_shadow_mode(model, True)
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

    rows: list[dict[str, Any]] = []
    categories = {
        category: {"passed": 0, "tasks": 0, "errors": 0}
        for category in P4_CATEGORIES
    }
    started = time.monotonic()

    for index, task in enumerate(suite.tasks, start=1):
        output = ""
        error_type = ""
        try:
            output = engine.generate_text(
                render_p4_chat_prompt(task.prompt),
                max_new_tokens=task.max_new_tokens,
            )
            passed = bool(score_p4_output(task, output))
        except Exception as exc:
            passed = False
            error_type = type(exc).__name__

        row = categories[task.category]
        row["tasks"] += 1
        if passed:
            row["passed"] += 1
        if error_type:
            row["errors"] += 1

        output_bytes = output.encode("utf-8")
        rows.append(
            {
                "task_id": task.task_id,
                "category": task.category,
                "passed": passed,
                "error_type": error_type,
                "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
                "output_utf8_bytes": len(output_bytes),
            }
        )

        if index % PROGRESS_INTERVAL == 0 or index == len(suite.tasks):
            passed_count = sum(1 for item in rows if item["passed"])
            elapsed = time.monotonic() - started
            print(
                "VN97 P5E3 PROGRESS "
                f"model={label} "
                f"tasks={index}/{len(suite.tasks)} "
                f"passed={passed_count} "
                f"elapsed_s={elapsed:.1f}",
                flush=True,
            )

    normalized_categories = {
        category: {
            "passed": int(value["passed"]),
            "tasks": int(value["tasks"]),
            "errors": int(value["errors"]),
            "pass_rate": (
                float(value["passed"]) / float(value["tasks"])
                if int(value["tasks"]) > 0
                else 0.0
            ),
        }
        for category, value in categories.items()
    }
    passed_count = sum(1 for item in rows if item["passed"])
    error_count = sum(1 for item in rows if item["error_type"])

    model.cpu()
    torch.cuda.empty_cache()

    return {
        "categories": normalized_categories,
        "errors": error_count,
        "pass_rate": passed_count / len(rows),
        "passed": passed_count,
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

    if int(p5e2["passed"]) < int(base["passed"]):
        reasons.append("overall_below_base")
    if int(p5e2["errors"]) > int(base["errors"]):
        reasons.append("more_runtime_errors_than_base")

    for category in guarded:
        candidate_passed = int(p5e2["categories"][category]["passed"])
        base_passed = int(base["categories"][category]["passed"])
        if candidate_passed < base_passed:
            reasons.append(f"{category}_below_base")

    if reasons:
        return "REJECTED_HELD_OUT_REGRESSION", reasons

    strongest_previous = max(int(base["passed"]), int(p5e1["passed"]))
    if int(p5e2["passed"]) > strongest_previous:
        return "HELD_OUT_GAIN", reasons
    return "HELD_OUT_NO_REGRESSION", reasons


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise VN97P5E3Error("CUDA device requested but CUDA is unavailable")

    suite = load_p4_task_suite(Path(args.suite).resolve(strict=True))
    artifacts = {
        "base": Path(args.base_dir),
        "p5e1": Path(args.p5e1_dir),
        "p5e2": Path(args.p5e2_dir),
    }

    evaluations: dict[str, dict[str, Any]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    tokenizer_hash: str | None = None

    for label in ("base", "p5e1", "p5e2"):
        print(f"VN97 P5E3 LOAD model={label}", flush=True)
        model, tokenizer, artifact_meta = _load_artifact(
            artifacts[label],
            label=label,
        )
        if tokenizer_hash is None:
            tokenizer_hash = str(artifact_meta["tokenizer_sha256"])
        elif artifact_meta["tokenizer_sha256"] != tokenizer_hash:
            raise VN97P5E3Error("P5E3 artifacts do not share one tokenizer")

        metadata[label] = artifact_meta
        evaluations[label] = _evaluate(
            label=label,
            model=model,
            tokenizer=tokenizer,
            suite=suite,
            device=args.device,
        )
        print(
            "VN97 P5E3 MODEL "
            f"model={label} "
            f"passed={evaluations[label]['passed']}/{evaluations[label]['tasks']} "
            f"pass_rate={evaluations[label]['pass_rate']:.6f} "
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
        "evaluations": evaluations,
        "gate": {
            "guarded_categories": ["tool_intent", "authority_behavior"],
            "reasons": reasons,
        },
        "profile_id": P5E3_PROFILE_ID,
        "schema": P5E3_SCHEMA,
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
        raise VN97P5E3Error("output path must not already exist")
    _atomic_write(output, data)

    print(
        "VN97P5E3 "
        f"status={status} "
        f"base={evaluations['base']['passed']}/{evaluations['base']['tasks']} "
        f"p5e1={evaluations['p5e1']['passed']}/{evaluations['p5e1']['tasks']} "
        f"p5e2={evaluations['p5e2']['passed']}/{evaluations['p5e2']['tasks']} "
        f"reasons={','.join(reasons) if reasons else 'none'} "
        f"report={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-p5e3-validate: {exc}", file=sys.stderr)
        raise
