from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from ..cognition_adapter import TorchVN97InferenceEngine
from ..training import VN97ChatMessage, VN97TrainingConfig
from ..training_cli import (
    _atomic_write,
    _load_records,
    _read_bounded_regular_file,
)
from .bridge import VN97R2InferenceView, assert_tokenizer_compatible
from .checkpoint import load_r2_checkpoint
from .config import r2_cpu_pilot_config, r2_smoke_config
from .data_bridge import build_completion_windows, load_vn97tk1
from .dense_training import (
    R2DenseTrainingConfig,
    evaluate_dense_loss,
    train_dense,
)
from .evaluation import EvaluationDomain
from .model import VN97R2Model
from .pilot_contract import (
    assert_r2_pilot_scale,
    build_pilot_corpus_evidence,
    estimate_pilot_training_resources,
)
from .pilot_evaluation import (
    R2PilotGateDecision,
    R2PilotProbe,
    evaluate_pilot_gate,
    evaluate_pilot_probes,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CPU/GPU-agnostic dense-first VN97-R2 pilot over canonical "
            "VN97TK1 + chat JSONL. R2-C pilot mode enforces 50-150M scale, "
            "train/validation disjointness and memory preflight. No "
            "quantization or ternary mode is used."
        )
    )
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--validation-jsonl", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--profile",
        choices=("smoke", "pilot"),
        default="smoke",
    )
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--max-windows", type=int, default=20_000)
    parser.add_argument("--max-examples", type=int, default=100_000)
    parser.add_argument(
        "--max-input-bytes",
        type=int,
        default=64 * 1024 * 1024,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=9705)

    parser.add_argument(
        "--probe-jsonl",
        default=None,
        help=(
            "Optional R2-C held-out probe suite. Required for a complete "
            "pilot acceptance decision."
        ),
    )
    parser.add_argument(
        "--probe-max-input-bytes",
        type=int,
        default=8 * 1024 * 1024,
    )
    parser.add_argument(
        "--probe-max-examples",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--probe-max-new-tokens",
        type=int,
        default=96,
    )
    parser.add_argument(
        "--allow-low-memory",
        action="store_true",
        help=(
            "Override the conservative R2-C RAM preflight. The run may OOM; "
            "this flag does not weaken model/evaluation gates."
        ),
    )
    parser.add_argument(
        "--require-pilot-gate",
        action="store_true",
        help=(
            "Return a non-zero exit code when the 50-150M pilot does not "
            "improve held-out loss or fails the complete probe gate."
        ),
    )
    return parser


def _combined_identity(
    *,
    train_sha: str,
    validation_sha: str,
    tokenizer_sha: str,
    profile: str,
) -> str:
    payload = json.dumps(
        {
            "train": train_sha,
            "validation": validation_sha,
            "tokenizer": tokenizer_sha,
            "profile": profile,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(
        b"VN97R2PILOTDATA\0" + payload
    ).hexdigest()


def _load_pilot_probes(
    path: Path,
    *,
    max_input_bytes: int,
    max_examples: int,
) -> tuple[tuple[R2PilotProbe, ...], str]:
    if max_input_bytes <= 0 or max_examples <= 0:
        raise ValueError("probe input bounds must be positive")
    data = _read_bounded_regular_file(
        path,
        max_bytes=max_input_bytes,
    )
    digest = hashlib.sha256(
        b"VN97R2PROBES1\0"
        + len(data).to_bytes(8, "little")
        + data
    ).hexdigest()
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("probe JSONL must be UTF-8") from exc

    probes: list[R2PilotProbe] = []
    for line_number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid probe JSONL at {path}:{line_number}"
            ) from exc
        if not isinstance(raw, dict):
            raise ValueError(
                f"probe record must be an object at {path}:{line_number}"
            )

        allowed = {
            "domain",
            "messages",
            "expected",
            "requires_external_write",
        }
        required = {"domain", "messages", "expected"}
        if set(raw) - allowed or not required.issubset(raw):
            raise ValueError(
                f"invalid probe keys at {path}:{line_number}"
            )
        if not isinstance(raw["messages"], list):
            raise ValueError(
                f"probe messages must be a list at {path}:{line_number}"
            )
        messages: list[VN97ChatMessage] = []
        for message in raw["messages"]:
            if (
                not isinstance(message, dict)
                or set(message) != {"role", "content"}
            ):
                raise ValueError(
                    f"invalid probe message at {path}:{line_number}"
                )
            messages.append(
                VN97ChatMessage(
                    role=message["role"],
                    content=message["content"],
                )
            )

        try:
            domain = EvaluationDomain(str(raw["domain"]))
        except ValueError as exc:
            raise ValueError(
                f"invalid probe domain at {path}:{line_number}"
            ) from exc
        requires_external_write = raw.get(
            "requires_external_write",
            False,
        )
        if type(requires_external_write) is not bool:
            raise ValueError(
                "probe requires_external_write must be boolean"
            )
        probes.append(
            R2PilotProbe(
                domain=domain,
                messages=tuple(messages),
                expected=raw["expected"],
                requires_external_write=requires_external_write,
            )
        )
        if len(probes) > max_examples:
            raise ValueError("probe count exceeds --probe-max-examples")

    if not probes:
        raise ValueError("probe suite contains no usable records")
    return tuple(probes), digest


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = Path(args.output_dir)
    if output.exists() and (
        output.is_symlink()
        or not output.is_dir()
        or any(output.iterdir())
    ):
        raise ValueError("output-dir must be new or empty")
    output.mkdir(parents=True, exist_ok=True)

    tokenizer_path = Path(args.tokenizer).resolve(strict=True)
    tokenizer = load_vn97tk1(tokenizer_path)
    tokenizer_sha = hashlib.sha256(
        tokenizer_path.read_bytes()
    ).hexdigest()

    training_records, train_sha = _load_records(
        [Path(args.train_jsonl)],
        mode="chat",
        max_input_bytes=args.max_input_bytes,
        max_examples=args.max_examples,
    )
    validation_records, validation_sha = _load_records(
        [Path(args.validation_jsonl)],
        mode="chat",
        max_input_bytes=args.max_input_bytes,
        max_examples=args.max_examples,
    )
    corpus_evidence = build_pilot_corpus_evidence(
        training_records,
        validation_records,
        reject_overlap=True,
    )

    if args.profile == "pilot":
        config = r2_cpu_pilot_config(tokenizer.vocab_size)
        assert_r2_pilot_scale(config)
    else:
        config = r2_smoke_config(tokenizer.vocab_size)
    model = VN97R2Model(config)
    assert_tokenizer_compatible(model, tokenizer)

    resources = estimate_pilot_training_resources(
        config,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
    )
    if (
        args.profile == "pilot"
        and resources.fits_available_ram is False
        and not args.allow_low_memory
    ):
        raise RuntimeError(
            "R2-C RAM preflight rejected this run: "
            f"recommended={resources.recommended_ram_bytes} "
            f"available={resources.available_ram_bytes}. "
            "Reduce sequence/batch size or pass --allow-low-memory "
            "only if you deliberately accept OOM risk."
        )

    window_config = VN97TrainingConfig(
        sequence_length=args.sequence_length,
        stride=args.stride,
        batch_size=args.batch_size,
        epochs=1,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        shuffle=True,
        max_windows=args.max_windows,
    )
    training_windows = build_completion_windows(
        tokenizer,
        training_records,
        window_config,
    )
    validation_windows = build_completion_windows(
        tokenizer,
        validation_records,
        window_config,
    )

    initial_validation = evaluate_dense_loss(
        model,
        validation_windows,
        batch_size=args.batch_size,
        device=args.device,
    )

    dataset_identity = _combined_identity(
        train_sha=train_sha,
        validation_sha=validation_sha,
        tokenizer_sha=tokenizer_sha,
        profile=args.profile,
    )
    dense_config = R2DenseTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        checkpoint_every_steps=args.checkpoint_every,
    )
    checkpoint_path = output / "model.r2.pt"
    result = train_dense(
        model,
        training_windows,
        validation_windows,
        dense_config,
        work_dir=args.work_dir,
        best_checkpoint_path=checkpoint_path,
        dataset_identity=dataset_identity,
        device=args.device,
    )

    best_model, checkpoint_evidence = load_r2_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )
    final_validation = evaluate_dense_loss(
        best_model,
        validation_windows,
        batch_size=args.batch_size,
        device="cpu",
    )

    initial_loss = float(initial_validation["mean_loss"])
    final_loss = float(final_validation["mean_loss"])
    relative_loss_improvement = (
        (initial_loss - final_loss) / initial_loss
        if initial_loss > 0.0
        else 0.0
    )

    first_messages = validation_records[0]
    first_user_text = next(
        (
            message.content
            for message in first_messages
            if message.role == "user"
        ),
        "VN97-R2",
    )
    engine = TorchVN97InferenceEngine(
        VN97R2InferenceView(
            best_model,
            profile="deep",
        ),
        tokenizer,
    )
    embedding = engine.embed_text(
        first_user_text,
        vector_dim=best_model.config.d_model,
    )
    if len(embedding) != best_model.config.d_model:
        raise RuntimeError("R2 cognition bridge embedding size mismatch")

    probe_sha: str | None = None
    probe_metrics = None
    probe_results = None
    probe_gate = R2PilotGateDecision(
        passed=False,
        reasons=("probe_suite_missing",),
    )
    if args.probe_jsonl is not None:
        probes, probe_sha = _load_pilot_probes(
            Path(args.probe_jsonl),
            max_input_bytes=args.probe_max_input_bytes,
            max_examples=args.probe_max_examples,
        )
        probe_metrics, probe_results = evaluate_pilot_probes(
            best_model,
            tokenizer,
            probes,
            max_new_tokens=args.probe_max_new_tokens,
            profile="deep",
            device="cpu",
        )
        probe_gate = evaluate_pilot_gate(probe_metrics)

    pilot_accepted = (
        args.profile == "pilot"
        and relative_loss_improvement > 0.0
        and probe_gate.passed
    )

    shutil.copyfile(
        tokenizer_path,
        output / "tokenizer.vn97tk1",
    )
    report = {
        "schema": "VN97R2DENSEPILOT2",
        "status": "PASS",
        "profile": args.profile,
        "architecture_id": best_model.config.architecture_id,
        "config_fingerprint": best_model.config.fingerprint(),
        "parameter_count": best_model.parameter_count(),
        "pilot_scale_50m_150m": (
            args.profile == "pilot"
            and 50_000_000 <= best_model.parameter_count() <= 150_000_000
        ),
        "dataset_identity": dataset_identity,
        "train_sha256": train_sha,
        "validation_sha256": validation_sha,
        "tokenizer_sha256": tokenizer_sha,
        "corpus": corpus_evidence.as_dict(),
        "resources": resources.as_dict(),
        "training_windows": len(training_windows),
        "validation_windows": len(validation_windows),
        "initial_validation": initial_validation,
        "training": {
            "steps": result.steps,
            "target_tokens": result.target_tokens,
            "mean_loss": result.mean_loss,
            "final_loss": result.final_loss,
            "best_epoch": result.best_epoch,
            "best_validation_loss": result.best_validation_loss,
            "best_checkpoint_sha256": (
                result.best_checkpoint_sha256
            ),
        },
        "final_validation": final_validation,
        "relative_validation_loss_improvement": (
            relative_loss_improvement
        ),
        "checkpoint": checkpoint_evidence,
        "cognition_bridge_embedding_dim": len(embedding),
        "probe_suite_sha256": probe_sha,
        "probe_metrics": (
            None if probe_metrics is None else probe_metrics.as_dict()
        ),
        "probe_gate": probe_gate.as_dict(),
        "probe_results": (
            None
            if probe_results is None
            else [item.as_dict() for item in probe_results]
        ),
        "pilot_accepted": pilot_accepted,
        "quantization_used": False,
    }
    _atomic_write(
        output / "r2-dense-pilot-report.json",
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8") + b"\n",
    )
    print(
        "VN97R2DENSEPILOT "
        f"status=PASS "
        f"profile={args.profile} "
        f"parameters={best_model.parameter_count()} "
        f"steps={result.steps} "
        f"best_epoch={result.best_epoch} "
        f"best_val_loss={result.best_validation_loss:.6f} "
        f"loss_improvement={relative_loss_improvement:.6f} "
        f"pilot_accepted={str(pilot_accepted).lower()} "
        f"quantization_used=false "
        f"output={output}",
        flush=True,
    )
    if args.require_pilot_gate and not pilot_accepted:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
