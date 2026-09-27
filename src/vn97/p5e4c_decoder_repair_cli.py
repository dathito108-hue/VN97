from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import torch
import torch.nn.functional as F

from .p3_tensor_cache import evaluate_vn97_from_tensors
from .p4_generalization_cli import _select_replay, _windows_to_tensors
from .p4_generalization_curriculum import P4D_CATEGORIES, default_validation
from .p4_targeted_repair_curriculum import targeted_training
from .p5e4b_fresh_validation_cli import _load_candidate
from .quantization import set_float_shadow_mode
from .training import (
    VN97TrainingConfig,
    build_training_windows,
    encode_chat_completion_messages,
)
from .training_cli import _atomic_write, _load_records


P5E4C_SCHEMA = "VN97P5E4C"
P5E4C_FLOAT_SCHEMA = "VN97P5E4CFLOAT1"
P5E4C_PROFILE_ID = "vn97-p5e4c-final-norm-decoder-repair-v1"

TRAIN_PER_CATEGORY = 80
P3_REPLAY_RECORDS = 320
SEQUENCE_LENGTH = 128
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
MAX_GRAD_NORM = 1.0
SEED = 9754


class VN97P5E4CError(RuntimeError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Repair VN97 output alignment by training only final RMSNorm while "
            "freezing the transplanted 32-layer SSM and tied lexical interface."
        )
    )
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--p3-corpus-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _state_hash(model, *, exclude_final_norm: bool) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if exclude_final_norm and name == "final_norm.weight":
            continue
        raw = tensor.detach().cpu().contiguous().numpy().tobytes()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(raw)
    return digest.hexdigest()


def _select_calibration_records():
    records = targeted_training()
    counts = defaultdict(int)
    selected = []
    for item in records:
        if counts[item.category] >= TRAIN_PER_CATEGORY:
            continue
        selected.append(item)
        counts[item.category] += 1
    expected = {
        category: TRAIN_PER_CATEGORY
        for category in P4D_CATEGORIES
    }
    if dict(counts) != expected:
        raise VN97P5E4CError(
            f"could not build balanced calibration set: {dict(counts)}"
        )
    return tuple(selected)


def _pre_norm_hidden(model, inputs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        x = model.embedding(inputs)
        for layer in model.layers:
            x, _ = layer(x, None)
    return x.detach()


def _train_final_norm(
    *,
    model,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    device: str,
    work_dir: Path,
) -> dict[str, float | int]:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise VN97P5E4CError("P5E4C requires CUDA")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.final_norm.weight.requires_grad_(True)

    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    if trainable_count != model.config.d_model:
        raise VN97P5E4CError(
            "P5E4C must train exactly final_norm.weight"
        )

    model.to(resolved)
    set_float_shadow_mode(model, True)
    model.train()

    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )
    rng = random.Random(SEED)
    indices = list(range(int(inputs.shape[0])))
    rng.shuffle(indices)

    total_steps = math.ceil(len(indices) / BATCH_SIZE)
    loss_sum = 0.0
    target_tokens = 0
    final_loss = math.nan

    work_dir.mkdir(parents=True, exist_ok=True)
    resume_path = work_dir / "state.p5e4c.pt"

    for step, batch_start in enumerate(
        range(0, len(indices), BATCH_SIZE),
        start=1,
    ):
        batch_indices = indices[batch_start : batch_start + BATCH_SIZE]
        cpu_inputs = inputs[batch_indices]
        cpu_labels = labels[batch_indices]
        batch_inputs = cpu_inputs.to(resolved)
        batch_labels = cpu_labels.to(resolved)

        optimizer.zero_grad(set_to_none=True)
        pre_norm = _pre_norm_hidden(model, batch_inputs)
        hidden = model.final_norm(pre_norm)
        logits = model.lm_head(hidden)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            batch_labels.reshape(-1),
            ignore_index=-100,
        )
        if not bool(torch.isfinite(loss)):
            raise VN97P5E4CError("decoder repair loss became non-finite")

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable,
            MAX_GRAD_NORM,
        )
        if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
            raise VN97P5E4CError(
                "decoder repair gradient norm became non-finite"
            )
        optimizer.step()

        final_loss = float(loss.detach().cpu())
        loss_sum += final_loss
        target_tokens += int((batch_labels != -100).sum().item())

        if step % 25 == 0 or step == total_steps:
            print(
                "VN97 P5E4C PROGRESS "
                f"step={step}/{total_steps} "
                f"percent={100.0 * step / total_steps:.2f} "
                f"loss={final_loss:.6f} "
                f"mean_loss={loss_sum / step:.6f} "
                f"grad_norm={float(grad_norm):.6f}",
                flush=True,
            )
        if step % 50 == 0 or step == total_steps:
            temp = resume_path.with_suffix(".tmp")
            torch.save(
                {
                    "schema": "VN97P5E4CRESUME1",
                    "step": step,
                    "total_steps": total_steps,
                    "final_norm_weight": (
                        model.final_norm.weight.detach().cpu()
                    ),
                },
                temp,
            )
            temp.replace(resume_path)

    model.eval()
    return {
        "steps": total_steps,
        "target_tokens": target_tokens,
        "mean_loss": loss_sum / total_steps,
        "final_loss": final_loss,
        "trainable_parameters": trainable_count,
    }


def _evaluation_dict(result) -> dict[str, float | int]:
    return {
        "windows": result.windows,
        "target_tokens": result.target_tokens,
        "mean_loss": result.mean_loss,
        "top1_accuracy": result.top1_accuracy,
    }


def _gate(
    *,
    dev_before,
    dev_after,
    p3_before,
    p3_after,
) -> tuple[str, list[str]]:
    reasons: list[str] = []

    dev_gain = (
        dev_after.mean_loss <= dev_before.mean_loss * 0.99
        or dev_after.top1_accuracy >= dev_before.top1_accuracy + 0.005
    )
    if not dev_gain:
        reasons.append("decoder_dev_no_material_gain")

    if p3_after.mean_loss > p3_before.mean_loss * 1.01:
        reasons.append("p3_loss_over_1pct")
    if p3_after.top1_accuracy + 0.005 < p3_before.top1_accuracy:
        reasons.append("p3_top1_drop_over_0.5pp")

    if reasons:
        return "DECODER_REPAIR_REJECTED", reasons
    return "DECODER_REPAIR_SIGNAL", []


def _write_torch_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.unlink(missing_ok=True)
    torch.save(payload, temp)
    temp.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise VN97P5E4CError("P5E4C requires CUDA")

    output = Path(args.output_dir)
    if output.exists() and (
        output.is_symlink()
        or not output.is_dir()
        or any(output.iterdir())
    ):
        raise VN97P5E4CError("output-dir must be new or empty")

    model, tokenizer, candidate_meta = _load_candidate(
        Path(args.candidate_dir)
    )
    set_float_shadow_mode(model, True)

    immutable_before = _state_hash(
        model,
        exclude_final_norm=True,
    )
    final_norm_before = hashlib.sha256(
        model.final_norm.weight.detach().cpu().numpy().tobytes()
    ).hexdigest()

    p3_root = Path(args.p3_corpus_dir).resolve(strict=True)
    p3_training_records, p3_training_sha = _load_records(
        [p3_root / "training.jsonl"],
        mode="chat",
        max_input_bytes=64 * 1024 * 1024,
        max_examples=100_000,
    )
    p3_validation_records, p3_validation_sha = _load_records(
        [p3_root / "validation.jsonl"],
        mode="chat",
        max_input_bytes=64 * 1024 * 1024,
        max_examples=100_000,
    )

    calibration_records = _select_calibration_records()
    replay_records = _select_replay(
        p3_training_records,
        count=P3_REPLAY_RECORDS,
    )

    train_messages = [
        item.messages
        for item in calibration_records
    ]
    train_messages.extend(replay_records)

    config = VN97TrainingConfig(
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        epochs=1,
        learning_rate=LEARNING_RATE,
        weight_decay=0.0,
        max_grad_norm=MAX_GRAD_NORM,
        seed=SEED,
        shuffle=True,
        max_windows=20_000,
    )
    eval_config = VN97TrainingConfig(
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        epochs=1,
        seed=0,
        shuffle=False,
        max_windows=20_000,
    )

    train_examples = [
        encode_chat_completion_messages(tokenizer, messages)
        for messages in train_messages
    ]
    dev_examples = [
        encode_chat_completion_messages(tokenizer, item.messages)
        for item in default_validation()
    ]
    p3_examples = [
        encode_chat_completion_messages(tokenizer, messages)
        for messages in p3_validation_records
    ]

    train_windows = build_training_windows(
        train_examples,
        config,
        pad_token_id=tokenizer.pad_id,
    )
    dev_windows = build_training_windows(
        dev_examples,
        eval_config,
        pad_token_id=tokenizer.pad_id,
    )
    p3_windows = build_training_windows(
        p3_examples,
        eval_config,
        pad_token_id=tokenizer.pad_id,
    )

    train_inputs, train_labels = _windows_to_tensors(train_windows)
    dev_inputs, dev_labels = _windows_to_tensors(dev_windows)
    p3_inputs, p3_labels = _windows_to_tensors(p3_windows)

    dev_before = evaluate_vn97_from_tensors(
        model,
        dev_inputs,
        dev_labels,
        batch_size=BATCH_SIZE,
        device=args.device,
        micro_batch_size=1,
    )
    p3_before = evaluate_vn97_from_tensors(
        model,
        p3_inputs,
        p3_labels,
        batch_size=BATCH_SIZE,
        device=args.device,
        micro_batch_size=1,
    )

    print(
        "VN97 P5E4C BEFORE "
        f"dev_loss={dev_before.mean_loss:.6f} "
        f"dev_top1={dev_before.top1_accuracy:.6f} "
        f"p3_loss={p3_before.mean_loss:.6f} "
        f"p3_top1={p3_before.top1_accuracy:.6f}",
        flush=True,
    )

    training = _train_final_norm(
        model=model,
        inputs=train_inputs,
        labels=train_labels,
        device=args.device,
        work_dir=Path(args.work_dir),
    )

    immutable_after = _state_hash(
        model,
        exclude_final_norm=True,
    )
    if immutable_after != immutable_before:
        raise VN97P5E4CError(
            "frozen SSM/embedding invariant changed during decoder repair"
        )

    dev_after = evaluate_vn97_from_tensors(
        model,
        dev_inputs,
        dev_labels,
        batch_size=BATCH_SIZE,
        device=args.device,
        micro_batch_size=1,
    )
    p3_after = evaluate_vn97_from_tensors(
        model,
        p3_inputs,
        p3_labels,
        batch_size=BATCH_SIZE,
        device=args.device,
        micro_batch_size=1,
    )

    status, reasons = _gate(
        dev_before=dev_before,
        dev_after=dev_after,
        p3_before=p3_before,
        p3_after=p3_after,
    )

    output.mkdir(parents=True, exist_ok=True)
    tokenizer_bytes = (
        Path(args.candidate_dir) / "tokenizer.vn97tk1"
    ).read_bytes()
    final_norm_after = hashlib.sha256(
        model.final_norm.weight.detach().cpu().numpy().tobytes()
    ).hexdigest()

    report = {
        "candidate_parent": candidate_meta,
        "calibration": {
            "p3_replay_records": P3_REPLAY_RECORDS,
            "seed": SEED,
            "sequence_length": SEQUENCE_LENGTH,
            "targeted_records_per_category": TRAIN_PER_CATEGORY,
            "trainable_component": "final_norm.weight",
            "training": training,
        },
        "evaluations": {
            "dev_before": _evaluation_dict(dev_before),
            "dev_after": _evaluation_dict(dev_after),
            "p3_before": _evaluation_dict(p3_before),
            "p3_after": _evaluation_dict(p3_after),
        },
        "fresh_holdout_required": True,
        "immutable_state_sha256": immutable_after,
        "p3_training_sha256": p3_training_sha,
        "p3_validation_sha256": p3_validation_sha,
        "profile_id": P5E4C_PROFILE_ID,
        "ready_for_fresh_validation": status == "DECODER_REPAIR_SIGNAL",
        "ready_for_qat": False,
        "reasons": reasons,
        "schema": P5E4C_SCHEMA,
        "status": status,
        "final_norm_before_sha256": final_norm_before,
        "final_norm_after_sha256": final_norm_after,
    }

    _atomic_write(
        output / "p5e4c-report.json",
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8") + b"\n",
    )

    if status == "DECODER_REPAIR_SIGNAL":
        payload = {
            "config": asdict(model.config),
            "float_shadow_required": True,
            "fresh_holdout_required": True,
            "immutable_state_sha256": immutable_after,
            "parent_candidate_sha256": candidate_meta["student_sha256"],
            "profile_id": P5E4C_PROFILE_ID,
            "ready_for_qat": False,
            "schema": P5E4C_FLOAT_SCHEMA,
            "state_dict": model.state_dict(),
            "status": status,
            "tokenizer_sha256": _sha256_bytes(tokenizer_bytes),
        }
        _write_torch_atomic(
            output / "student-float.pt",
            payload,
        )
        _atomic_write(
            output / "tokenizer.vn97tk1",
            tokenizer_bytes,
        )
        sums = {
            "p5e4c-report.json": _sha256_file(
                output / "p5e4c-report.json"
            ),
            "student-float.pt": _sha256_file(
                output / "student-float.pt"
            ),
            "tokenizer.vn97tk1": _sha256_file(
                output / "tokenizer.vn97tk1"
            ),
        }
        _atomic_write(
            output / "SHA256SUMS",
            "".join(
                f"{digest}  {name}\n"
                for name, digest in sorted(sums.items())
            ).encode("ascii"),
        )

    print(
        "VN97P5E4C "
        f"status={status} "
        f"dev_loss_before={dev_before.mean_loss:.6f} "
        f"dev_loss_after={dev_after.mean_loss:.6f} "
        f"dev_top1_before={dev_before.top1_accuracy:.6f} "
        f"dev_top1_after={dev_after.top1_accuracy:.6f} "
        f"p3_loss_before={p3_before.mean_loss:.6f} "
        f"p3_loss_after={p3_after.mean_loss:.6f} "
        f"p3_top1_before={p3_before.top1_accuracy:.6f} "
        f"p3_top1_after={p3_after.top1_accuracy:.6f} "
        f"immutable_state_match={str(immutable_before == immutable_after).lower()} "
        f"ready_for_fresh_validation="
        f"{str(status == 'DECODER_REPAIR_SIGNAL').lower()} "
        f"ready_for_qat=false "
        f"reasons={','.join(reasons) if reasons else 'none'}",
        flush=True,
    )
    print(f"P5E4C final: {output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-p5e4c-repair: {exc}", file=sys.stderr)
        raise
