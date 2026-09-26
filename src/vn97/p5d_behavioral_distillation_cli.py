from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import torch
import torch.nn.functional as F

from .model import VN97LanguageCore
from .p3_language_campaign_cli import _validate_corpus
from .p4_compositional_repair_cli import _canonical_measure_records
from .p4_generalization_cli import _probe_records
from .p4_generalization_curriculum import default_validation as p4_validation
from .p5_scaled_foundation_cli import (
    _select_windows,
    _tensor_batch,
)
from .p5d_behavioral_distillation import (
    CANDIDATES,
    CHECKPOINT_INTERVAL_STEPS,
    LOGICAL_BATCH_SIZE,
    MAX_GRAD_NORM,
    MAX_TRAIN_WINDOWS,
    P3_REPLAY_RECORDS,
    P3_VALIDATION_MAX_WINDOWS,
    P5D2_FLOAT_ARTIFACT_SCHEMA,
    P5D2_PROFILE_ID,
    PROGRESS_INTERVAL_STEPS,
    SEQUENCE_LENGTH,
    TEACHER_HOLDOUT_GENERAL_LANGUAGE,
    TEACHER_HOLDOUT_PER_P4_CATEGORY,
    WEIGHT_DECAY,
    profile_sha256,
)
from .p5d_distillation import target_config
from .quantization import set_float_shadow_mode
from .tokenizer import VN97Tokenizer, VN97TokenizerPackage
from .training import (
    IGNORE_INDEX,
    VN97ChatMessage,
    VN97TrainingConfig,
    build_training_windows,
    encode_chat_completion_messages,
)
from .training_cli import (
    _atomic_write,
    _load_records,
)


class VN97P5D2Error(RuntimeError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train one 309M VN97-RT float-shadow behavioral-distillation "
            "candidate from the sealed P5D1 teacher corpus."
        )
    )
    parser.add_argument("--teacher-corpus-dir", required=True)
    parser.add_argument("--vn97-tokenizer", required=True)
    parser.add_argument("--p3-corpus-dir", required=True)
    parser.add_argument("--candidate-index", type=int, required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _verify_teacher_corpus(root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    resolved = root.resolve(strict=True)
    required = {
        "SHA256SUMS",
        "p5d1-report.json",
        "teacher-corpus.jsonl",
    }
    names = {item.name for item in resolved.iterdir()}
    if names != required:
        raise VN97P5D2Error("P5D1 artifact file set mismatch")

    sums: dict[str, str] = {}
    for line in (resolved / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        if len(line) < 67 or line[64:66] != "  ":
            raise VN97P5D2Error("malformed P5D1 SHA256SUMS")
        sums[line[66:]] = line[:64]

    if set(sums) != {"p5d1-report.json", "teacher-corpus.jsonl"}:
        raise VN97P5D2Error("P5D1 SHA256SUMS file set mismatch")

    for name, expected in sums.items():
        if _sha256(resolved / name) != expected:
            raise VN97P5D2Error(f"P5D1 SHA256 mismatch: {name}")

    report = json.loads((resolved / "p5d1-report.json").read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict)
        or report.get("schema") != "VN97P5D1"
        or report.get("completed_records") != report.get("total_records")
    ):
        raise VN97P5D2Error("P5D1 report is not complete")

    records: list[dict[str, object]] = []
    for line in (resolved / "teacher-corpus.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise VN97P5D2Error("P5D1 corpus row must be an object")
        records.append(value)

    if len(records) != int(report["total_records"]):
        raise VN97P5D2Error("P5D1 corpus/report record count mismatch")
    return records, report


def _split_teacher_records(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_category: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        category = str(row["category"])
        by_category.setdefault(category, []).append(row)

    train: list[dict[str, object]] = []
    holdout: list[dict[str, object]] = []
    for category, category_rows in sorted(by_category.items()):
        category_rows.sort(key=lambda row: str(row["record_id"]))
        holdout_count = (
            TEACHER_HOLDOUT_GENERAL_LANGUAGE
            if category == "general_language"
            else TEACHER_HOLDOUT_PER_P4_CATEGORY
        )
        if len(category_rows) <= holdout_count:
            raise VN97P5D2Error(f"not enough P5D1 rows for {category}")
        holdout.extend(category_rows[:holdout_count])
        train.extend(category_rows[holdout_count:])

    train.sort(key=lambda row: str(row["record_id"]))
    holdout.sort(key=lambda row: str(row["record_id"]))
    return train, holdout


def _teacher_messages(row: dict[str, object]) -> tuple[VN97ChatMessage, ...]:
    raw_messages = row["messages"]
    if not isinstance(raw_messages, list):
        raise VN97P5D2Error("teacher messages must be a list")
    messages = [
        VN97ChatMessage(
            role=str(item["role"]),
            content=str(item["content"]),
        )
        for item in raw_messages
    ]
    messages.append(
        VN97ChatMessage(
            role="assistant",
            content=str(row["teacher_response"]),
        )
    )
    return tuple(messages)


def _windows_from_messages(
    *,
    tokenizer: VN97Tokenizer,
    messages_list: list[tuple[VN97ChatMessage, ...]],
) -> tuple[object, ...]:
    config = VN97TrainingConfig(
        sequence_length=SEQUENCE_LENGTH,
        batch_size=1,
        epochs=1,
        learning_rate=1e-4,
        seed=0,
        shuffle=False,
        max_windows=100_000,
    )
    examples = [
        encode_chat_completion_messages(tokenizer, messages)
        for messages in messages_list
    ]
    return build_training_windows(
        examples,
        config,
        pad_token_id=tokenizer.pad_id,
    )


@torch.inference_mode()
def _evaluate_fp32(
    model: VN97LanguageCore,
    windows: tuple[object, ...],
    *,
    device: str,
) -> dict[str, float | int]:
    if not windows:
        raise VN97P5D2Error("evaluation windows are empty")

    model.eval()
    loss_sum = 0.0
    target_tokens = 0
    correct = 0

    for window in windows:
        inputs, labels = _tensor_batch(window, device=device)
        logits, _ = model.forward_sequential_reference(inputs)
        mask = labels != IGNORE_INDEX
        if not bool(mask.any()):
            continue
        selected_logits = logits[mask].float()
        selected_labels = labels[mask]
        loss = F.cross_entropy(
            selected_logits,
            selected_labels,
            reduction="sum",
        )
        count = int(selected_labels.numel())
        loss_sum += float(loss.item())
        target_tokens += count
        correct += int(
            (selected_logits.argmax(dim=-1) == selected_labels).sum().item()
        )
        del inputs, labels, logits, selected_logits, selected_labels, loss

    if target_tokens <= 0:
        raise VN97P5D2Error("evaluation produced no target tokens")

    return {
        "mean_loss": loss_sum / target_tokens,
        "target_tokens": target_tokens,
        "top1_accuracy": correct / target_tokens,
    }


def _save_resume(
    path: Path,
    *,
    identity: str,
    model: VN97LanguageCore,
    optimizer: torch.optim.Optimizer,
    next_step: int,
    loss_sum: float,
    target_tokens: int,
) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "identity": identity,
            "loss_sum": loss_sum,
            "model": model.state_dict(),
            "next_step": next_step,
            "optimizer": optimizer.state_dict(),
            "target_tokens": target_tokens,
        },
        temp,
    )
    temp.replace(path)


def _load_resume(
    path: Path,
    *,
    identity: str,
    model: VN97LanguageCore,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, float, int]:
    if not path.exists():
        return 0, 0.0, 0

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    if payload.get("identity") != identity:
        raise VN97P5D2Error("P5D2 resume identity mismatch")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    return (
        int(payload["next_step"]),
        float(payload["loss_sum"]),
        int(payload["target_tokens"]),
    )


def _status(
    *,
    teacher_before: dict[str, float | int],
    teacher_after: dict[str, float | int],
    p3_after: dict[str, float | int],
) -> str:
    before_loss = float(teacher_before["mean_loss"])
    after_loss = float(teacher_after["mean_loss"])
    after_p3 = float(p3_after["mean_loss"])
    if not math.isfinite(after_loss) or not math.isfinite(after_p3):
        return "REJECTED_NONFINITE"

    relative_gain = (before_loss - after_loss) / max(before_loss, 1e-9)
    top1_gain = (
        float(teacher_after["top1_accuracy"])
        - float(teacher_before["top1_accuracy"])
    )
    if relative_gain >= 0.10 and top1_gain > 0.0:
        return "BEHAVIORAL_SIGNAL"
    if after_loss < before_loss and top1_gain >= 0.0:
        return "WEAK_SIGNAL"
    return "NO_SIGNAL"


def _write_float_artifact(
    *,
    output: Path,
    model: VN97LanguageCore,
    tokenizer_bytes: bytes,
    candidate_index: int,
    candidate,
    report_sha256: str,
) -> str:
    artifact_path = output / "student-float.pt"
    temp = artifact_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "candidate": candidate.canonical_object(),
            "candidate_index": candidate_index,
            "config": asdict(model.config),
            "float_shadow_required": True,
            "p5d2_profile_id": P5D2_PROFILE_ID,
            "report_sha256": report_sha256,
            "schema": P5D2_FLOAT_ARTIFACT_SCHEMA,
            "state_dict": model.state_dict(),
            "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
        },
        temp,
    )
    temp.replace(artifact_path)
    return _sha256(artifact_path)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if not (0 <= args.candidate_index < len(CANDIDATES)):
        raise VN97P5D2Error("candidate-index is out of range")
    if not torch.cuda.is_available():
        raise VN97P5D2Error("P5D2 requires CUDA")

    candidate = CANDIDATES[args.candidate_index]
    teacher_rows, p5d1_report = _verify_teacher_corpus(
        Path(args.teacher_corpus_dir)
    )
    teacher_train, teacher_holdout = _split_teacher_records(teacher_rows)

    tokenizer_bytes = Path(args.vn97_tokenizer).read_bytes()
    package = VN97TokenizerPackage.from_bytes(tokenizer_bytes)
    tokenizer = VN97Tokenizer(package)

    p3_root = Path(args.p3_corpus_dir).resolve(strict=True)
    _validate_corpus(p3_root)
    p3_train_raw, p3_train_sha = _load_records(
        [p3_root / "training.jsonl"],
        mode="chat",
        max_input_bytes=64 * 1024 * 1024,
        max_examples=100_000,
    )
    p3_validation_raw, p3_validation_sha = _load_records(
        [p3_root / "validation.jsonl"],
        mode="chat",
        max_input_bytes=64 * 1024 * 1024,
        max_examples=100_000,
    )

    p3_train = sorted(
        (tuple(messages) for messages in p3_train_raw),
        key=lambda messages: hashlib.sha256(
            json.dumps(
                [
                    {"role": item.role, "content": item.content}
                    for item in messages
                ],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).digest(),
    )[:P3_REPLAY_RECORDS]

    teacher_train_windows = _windows_from_messages(
        tokenizer=tokenizer,
        messages_list=[_teacher_messages(row) for row in teacher_train],
    )
    teacher_holdout_windows = _windows_from_messages(
        tokenizer=tokenizer,
        messages_list=[_teacher_messages(row) for row in teacher_holdout],
    )
    p3_train_windows = _windows_from_messages(
        tokenizer=tokenizer,
        messages_list=list(p3_train),
    )
    p3_validation_windows_all = _windows_from_messages(
        tokenizer=tokenizer,
        messages_list=[tuple(messages) for messages in p3_validation_raw],
    )

    half = MAX_TRAIN_WINDOWS // 2
    teacher_take = min(half, len(teacher_train_windows))
    p3_take = min(MAX_TRAIN_WINDOWS - teacher_take, len(p3_train_windows))
    train_windows = [
        *_select_windows(teacher_train_windows, teacher_take),
        *_select_windows(p3_train_windows, p3_take),
    ]
    train_windows.sort(
        key=lambda window: hashlib.sha256(
            bytes(
                int(value) & 0xFF
                for value in window.input_ids[:64]
            )
            + len(window.input_ids).to_bytes(4, "little")
        ).digest()
    )

    p3_validation_take = min(
        P3_VALIDATION_MAX_WINDOWS,
        len(p3_validation_windows_all),
    )
    p3_validation_windows = _select_windows(
        p3_validation_windows_all,
        p3_validation_take,
    )

    torch.manual_seed(candidate.seed)
    model = VN97LanguageCore(
        target_config(vocab_size=package.vocab_size)
    )
    ternary_layers = set_float_shadow_mode(model, True)
    model.to(args.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=candidate.learning_rate,
        weight_decay=WEIGHT_DECAY,
    )

    work = Path(args.work_dir)
    output = Path(args.output_dir)
    if work.is_symlink() or output.is_symlink():
        raise VN97P5D2Error("work/output directories must not be symlinks")
    work.mkdir(parents=True, exist_ok=True)
    if output.exists() and any(output.iterdir()):
        raise VN97P5D2Error("output-dir must be new or empty")

    identity_payload = json.dumps(
        {
            "candidate": candidate.canonical_object(),
            "candidate_index": args.candidate_index,
            "p3_train_sha256": p3_train_sha,
            "p3_validation_sha256": p3_validation_sha,
            "p5d1_corpus_sha256": p5d1_report["corpus_sha256"],
            "profile_sha256": profile_sha256(),
            "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
            "train_windows": len(train_windows),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    identity = hashlib.sha256(b"VN97P5D2RUN\0" + identity_payload).hexdigest()

    teacher_before = _evaluate_fp32(
        model,
        teacher_holdout_windows,
        device=args.device,
    )
    p3_before = _evaluate_fp32(
        model,
        p3_validation_windows,
        device=args.device,
    )

    total_steps = math.ceil(len(train_windows) / LOGICAL_BATCH_SIZE)
    resume_path = work / "state.p5d2.pt"
    start_step, cumulative_loss, cumulative_targets = _load_resume(
        resume_path,
        identity=identity,
        model=model,
        optimizer=optimizer,
    )

    order = list(range(len(train_windows)))
    random.Random(candidate.seed).shuffle(order)

    print(
        "VN97 P5D2 START "
        f"candidate={args.candidate_index} "
        f"candidate_id={candidate.candidate_id} "
        f"parameters={sum(int(p.numel()) for p in model.parameters())} "
        f"ternary_layers_float_shadow={ternary_layers} "
        f"train_windows={len(train_windows)} "
        f"steps={total_steps} "
        f"resume_step={start_step} "
        f"lr={candidate.learning_rate} "
        "precision=fp32 "
        f"device={args.device}",
        flush=True,
    )
    print(
        "VN97 P5D2 BASELINE "
        f"candidate={args.candidate_index} "
        f"teacher_loss={float(teacher_before['mean_loss']):.6f} "
        f"teacher_top1={float(teacher_before['top1_accuracy']):.6f} "
        f"p3_loss={float(p3_before['mean_loss']):.6f} "
        f"p3_top1={float(p3_before['top1_accuracy']):.6f}",
        flush=True,
    )

    model.train()
    started = time.monotonic()
    for step in range(start_step, total_steps):
        begin = step * LOGICAL_BATCH_SIZE
        indices = order[begin : min(begin + LOGICAL_BATCH_SIZE, len(order))]
        batch = [train_windows[index] for index in indices]
        logical_targets = sum(window.target_tokens for window in batch)
        if logical_targets <= 0:
            raise VN97P5D2Error("logical batch has no target tokens")

        optimizer.zero_grad(set_to_none=True)
        logical_loss_sum = 0.0

        for window in batch:
            inputs, labels = _tensor_batch(window, device=args.device)
            logits, _ = model.forward_sequential_reference(inputs)
            token_loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            loss = token_loss_sum / logical_targets
            if not bool(torch.isfinite(loss)):
                raise VN97P5D2Error("training loss became non-finite")
            loss.backward()
            logical_loss_sum += float(token_loss_sum.detach().item())
            del inputs, labels, logits, token_loss_sum, loss

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            MAX_GRAD_NORM,
        )
        if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
            raise VN97P5D2Error("gradient norm became non-finite")
        optimizer.step()

        cumulative_loss += logical_loss_sum
        cumulative_targets += logical_targets
        completed = step + 1

        if (
            completed % PROGRESS_INTERVAL_STEPS == 0
            or completed == total_steps
        ):
            elapsed = time.monotonic() - started
            local_steps = max(1, completed - start_step)
            eta = (total_steps - completed) * (elapsed / local_steps)
            print(
                "VN97 P5D2 PROGRESS "
                f"candidate={args.candidate_index} "
                f"step={completed}/{total_steps} "
                f"percent={100.0 * completed / total_steps:.2f} "
                f"elapsed_s={elapsed:.1f} "
                f"eta_s={eta:.1f} "
                f"batch_loss={logical_loss_sum / logical_targets:.6f} "
                f"mean_loss={cumulative_loss / cumulative_targets:.6f}",
                flush=True,
            )

        if (
            completed % CHECKPOINT_INTERVAL_STEPS == 0
            and completed < total_steps
        ):
            _save_resume(
                resume_path,
                identity=identity,
                model=model,
                optimizer=optimizer,
                next_step=completed,
                loss_sum=cumulative_loss,
                target_tokens=cumulative_targets,
            )
            print(
                "VN97 P5D2 CHECKPOINT "
                f"candidate={args.candidate_index} "
                f"step={completed}/{total_steps} "
                f"path={resume_path}",
                flush=True,
            )

    del optimizer
    torch.cuda.empty_cache()
    model.eval()

    teacher_after = _evaluate_fp32(
        model,
        teacher_holdout_windows,
        device=args.device,
    )
    p3_after = _evaluate_fp32(
        model,
        p3_validation_windows,
        device=args.device,
    )
    probe_after = _canonical_measure_records(
        model=model,
        tokenizer=tokenizer,
        records=_probe_records(p4_validation(), per_category=2),
        device=args.device,
    )

    status = _status(
        teacher_before=teacher_before,
        teacher_after=teacher_after,
        p3_after=p3_after,
    )

    report = {
        "candidate": candidate.canonical_object(),
        "candidate_id": candidate.candidate_id,
        "candidate_index": args.candidate_index,
        "p3": {
            "after": p3_after,
            "before": p3_before,
        },
        "p4_probe_after": probe_after,
        "p5d1": {
            "corpus_sha256": p5d1_report["corpus_sha256"],
            "teacher": p5d1_report["teacher"],
        },
        "profile_id": P5D2_PROFILE_ID,
        "profile_sha256": profile_sha256(),
        "schema": "VN97P5D2CAND1",
        "status": status,
        "teacher_holdout": {
            "after": teacher_after,
            "before": teacher_before,
            "records": len(teacher_holdout),
        },
        "training": {
            "final_mean_loss": cumulative_loss / cumulative_targets,
            "precision": "fp32-float-shadow",
            "steps": total_steps,
            "target_tokens": cumulative_targets,
            "train_windows": len(train_windows),
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    report_bytes = (
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    report_sha = hashlib.sha256(report_bytes).hexdigest()
    _atomic_write(output / "p5d2-report.json", report_bytes)

    model.cpu()
    torch.cuda.empty_cache()
    float_sha = _write_float_artifact(
        output=output,
        model=model,
        tokenizer_bytes=tokenizer_bytes,
        candidate_index=args.candidate_index,
        candidate=candidate,
        report_sha256=report_sha,
    )
    _atomic_write(output / "tokenizer.vn97tk1", tokenizer_bytes)

    files = {
        "p5d2-report.json": output / "p5d2-report.json",
        "student-float.pt": output / "student-float.pt",
        "tokenizer.vn97tk1": output / "tokenizer.vn97tk1",
    }
    sums = "".join(
        f"{_sha256(path)}  {name}\n"
        for name, path in sorted(files.items())
    ).encode("ascii")
    _atomic_write(output / "SHA256SUMS", sums)

    resume_path.unlink(missing_ok=True)

    print(
        "VN97P5D2 "
        f"status={status} "
        f"candidate={args.candidate_index} "
        f"candidate_id={candidate.candidate_id} "
        f"teacher_before_loss={float(teacher_before['mean_loss']):.6f} "
        f"teacher_after_loss={float(teacher_after['mean_loss']):.6f} "
        f"teacher_after_top1={float(teacher_after['top1_accuracy']):.6f} "
        f"p3_after_top1={float(p3_after['top1_accuracy']):.6f} "
        f"probe_after={int(probe_after['passed'])}/{int(probe_after['task_count'])} "
        f"student_sha256={float_sha}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-p5d2-distill: {exc}", file=sys.stderr)
        raise
