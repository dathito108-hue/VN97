from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

from .p4_generalization_cli import _select_replay, _windows_to_tensors
from .p4_generalization_curriculum import P4D_CATEGORIES, default_validation
from .p4_targeted_repair_curriculum import targeted_training
from .p5e4c_decoder_repair_cli import (
    P3_REPLAY_RECORDS,
    SEQUENCE_LENGTH,
    _sha256_file,
    _state_hash,
)
from .p5e4d_fresh_decoder_validation_cli import _load_repaired
from .p5e4e_output_expansion_cli import (
    ADAPTER_RANK,
    ResidualOutputAdapter,
)
from .p5e4f_fresh_output_validation_cli import (
    P5E4F_SCHEMA,
    _load_expansion,
)
from .quantization import set_float_shadow_mode
from .training import (
    VN97TrainingConfig,
    build_training_windows,
    encode_chat_completion_messages,
)
from .training_cli import _atomic_write, _load_records


P5E4G_SCHEMA = "VN97P5E4G"
P5E4G_ADAPTER_SCHEMA = "VN97P5E4GOUTPUT1"
P5E4G_PROFILE_ID = "vn97-p5e4g-prefix-decoding-repair-v1"

TARGETED_PER_CATEGORY = 160
BATCH_SIZE = 2
LEARNING_RATE = 8e-5
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
LOGIT_DELTA_L2 = 5e-6
FIRST_TOKEN_WEIGHT = 8.0
EARLY_4_WEIGHT = 4.0
EARLY_16_WEIGHT = 2.0
FINAL_TOKEN_WEIGHT = 2.0
PROGRESS_INTERVAL = 25
CHECKPOINT_INTERVAL = 50
SEED = 9756


class VN97P5E4GError(RuntimeError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "P5E4G prefix decoding repair. Continue training only the existing "
            "P5E4E low-rank output adapter with strong early-response token "
            "weighting while the complete VN97 recurrent core remains frozen."
        )
    )
    parser.add_argument("--repaired-dir", required=True)
    parser.add_argument("--expanded-dir", required=True)
    parser.add_argument("--p5e4f-report", required=True)
    parser.add_argument("--p3-corpus-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verify_p5e4f_gate(
    path: Path,
    *,
    repaired_meta: dict[str, Any],
    expanded_meta: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    resolved = path.resolve(strict=True)
    raw = resolved.read_bytes()
    report = json.loads(raw.decode("utf-8"))
    if not isinstance(report, dict) or report.get("schema") != P5E4F_SCHEMA:
        raise VN97P5E4GError("invalid P5E4F report schema")
    if report.get("one_shot") is not True:
        raise VN97P5E4GError("P5E4F evidence is not one-shot")
    if report.get("ready_for_qat") is not False:
        raise VN97P5E4GError("P5E4G cannot run after QAT authorization")
    if report.get("ready_for_decoding_repair") is not True:
        raise VN97P5E4GError("P5E4F did not authorize decoding repair")
    if report.get("status") not in {
        "FRESH_OUTPUT_LATENT_GAIN",
        "FRESH_OUTPUT_ZERO_GENERATION",
        "FRESH_OUTPUT_PRESERVED",
        "FRESH_OUTPUT_GENERATION_GAIN",
    }:
        raise VN97P5E4GError("P5E4F status is not decoding-repair eligible")

    artifacts = report.get("artifacts")
    repaired = artifacts.get("repaired") if isinstance(artifacts, dict) else None
    expanded = artifacts.get("expanded") if isinstance(artifacts, dict) else None
    if not isinstance(repaired, dict) or not isinstance(expanded, dict):
        raise VN97P5E4GError("P5E4F artifact identities missing")
    if repaired.get("student_sha256") != repaired_meta.get("student_sha256"):
        raise VN97P5E4GError("P5E4F/P5E4C checkpoint identity mismatch")
    if expanded.get("adapter_sha256") != expanded_meta.get("adapter_sha256"):
        raise VN97P5E4GError("P5E4F/P5E4E adapter identity mismatch")
    return report, _sha256_bytes(raw)


def _select_targeted_records():
    records = targeted_training()
    counts = defaultdict(int)
    selected = []
    for item in records:
        if counts[item.category] >= TARGETED_PER_CATEGORY:
            continue
        selected.append(item)
        counts[item.category] += 1
    expected = {
        category: TARGETED_PER_CATEGORY
        for category in P4D_CATEGORIES
    }
    if dict(counts) != expected:
        raise VN97P5E4GError(
            f"could not build balanced decoding repair set: {dict(counts)}"
        )
    return tuple(selected)


def _hidden_and_base_logits(model, inputs: torch.Tensor):
    with torch.no_grad():
        hidden, _ = model.forward_hidden_embeddings_sequential_reference(
            model.embedding(inputs),
            None,
        )
        logits = model.lm_head(hidden)
    return hidden.detach(), logits.detach()


def _prefix_weights(labels: torch.Tensor) -> torch.Tensor:
    weights = torch.zeros_like(labels, dtype=torch.float32)
    for row_index in range(int(labels.shape[0])):
        positions = torch.nonzero(
            labels[row_index] != -100,
            as_tuple=False,
        ).flatten()
        if positions.numel() == 0:
            continue
        weights[row_index, positions] = 1.0
        weights[row_index, positions[:16]] = EARLY_16_WEIGHT
        weights[row_index, positions[:4]] = EARLY_4_WEIGHT
        weights[row_index, positions[:1]] = FIRST_TOKEN_WEIGHT
        last = int(positions[-1].item())
        weights[row_index, last] = max(
            float(weights[row_index, last].item()),
            FINAL_TOKEN_WEIGHT,
        )
    return weights


def _weighted_target_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    mask = labels != -100
    if not bool(mask.any()):
        raise VN97P5E4GError("targeted batch contains no supervised tokens")
    flat_logits = logits[mask]
    flat_labels = labels[mask]
    flat_weights = _prefix_weights(labels)[mask].to(
        device=logits.device,
        dtype=torch.float32,
    )
    per_token = F.cross_entropy(
        flat_logits.float(),
        flat_labels,
        reduction="none",
    )
    return (per_token * flat_weights).sum() / flat_weights.sum()


@torch.inference_mode()
def _evaluate(
    *,
    model,
    adapter: ResidualOutputAdapter,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    device: str,
) -> dict[str, float | int]:
    resolved = torch.device(device)
    model.eval()
    adapter.eval()

    loss_sum = 0.0
    target_tokens = 0
    correct = 0
    first_count = 0
    first_correct = 0
    first_rank_sum = 0

    windows = int(inputs.shape[0])
    for start in range(0, windows, BATCH_SIZE):
        x = inputs[start : start + BATCH_SIZE].to(
            device=resolved,
            dtype=torch.long,
        )
        y = labels[start : start + BATCH_SIZE].to(
            device=resolved,
            dtype=torch.long,
        )
        hidden, base = _hidden_and_base_logits(model, x)
        logits = base + adapter(hidden)

        flat_y = y.reshape(-1)
        mask = flat_y != -100
        count = int(mask.sum().item())
        if count > 0:
            selected_logits = logits.reshape(-1, logits.shape[-1])[mask]
            selected_y = flat_y[mask]
            loss_sum += float(
                F.cross_entropy(
                    selected_logits.float(),
                    selected_y,
                    reduction="sum",
                ).item()
            )
            target_tokens += count
            correct += int(
                (selected_logits.argmax(dim=-1) == selected_y).sum().item()
            )

        for row_index in range(int(y.shape[0])):
            positions = torch.nonzero(
                y[row_index] != -100,
                as_tuple=False,
            ).flatten()
            if positions.numel() == 0:
                continue
            pos = int(positions[0].item())
            label = y[row_index, pos]
            scores = logits[row_index, pos].float()
            label_score = scores[label]
            rank = int((scores > label_score).sum().item()) + 1
            first_count += 1
            first_rank_sum += rank
            first_correct += int(rank == 1)

    if target_tokens <= 0 or first_count <= 0:
        raise VN97P5E4GError("evaluation produced no supervised signal")

    return {
        "windows": windows,
        "target_tokens": target_tokens,
        "mean_loss": loss_sum / target_tokens,
        "top1_accuracy": correct / target_tokens,
        "first_target_count": first_count,
        "first_top1_accuracy": first_correct / first_count,
        "mean_first_rank": first_rank_sum / first_count,
    }


def _move_optimizer_state(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _train(
    *,
    model,
    adapter: ResidualOutputAdapter,
    targeted_inputs: torch.Tensor,
    targeted_labels: torch.Tensor,
    replay_inputs: torch.Tensor,
    replay_labels: torch.Tensor,
    device: str,
    work_dir: Path,
    identity: str,
) -> dict[str, float | int]:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise VN97P5E4GError("P5E4G requires CUDA")

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapter.to(resolved)
    adapter.train()

    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    rng = random.Random(SEED)
    targeted_order = list(range(int(targeted_inputs.shape[0])))
    replay_order = list(range(int(replay_inputs.shape[0])))
    rng.shuffle(targeted_order)
    rng.shuffle(replay_order)

    schedule: list[tuple[str, list[int]]] = []
    for start in range(0, len(targeted_order), BATCH_SIZE):
        schedule.append(
            ("targeted", targeted_order[start : start + BATCH_SIZE])
        )
    for start in range(0, len(replay_order), BATCH_SIZE):
        schedule.append(
            ("replay", replay_order[start : start + BATCH_SIZE])
        )
    rng.shuffle(schedule)

    total_steps = len(schedule)
    if total_steps <= 0:
        raise VN97P5E4GError("decoding repair schedule is empty")

    work_dir.mkdir(parents=True, exist_ok=True)
    resume_path = work_dir / "state.p5e4g.pt"
    start_step = 0
    loss_sum = 0.0
    targeted_steps = 0
    replay_steps = 0

    if resume_path.is_file():
        resume = torch.load(
            resume_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(resume, dict) or resume.get("identity") != identity:
            raise VN97P5E4GError("P5E4G resume identity mismatch")
        adapter.load_state_dict(resume["adapter_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        _move_optimizer_state(optimizer, resolved)
        start_step = int(resume.get("next_step", 0))
        loss_sum = float(resume.get("loss_sum", 0.0))
        targeted_steps = int(resume.get("targeted_steps", 0))
        replay_steps = int(resume.get("replay_steps", 0))
        if not 0 <= start_step <= total_steps:
            raise VN97P5E4GError("P5E4G resume step out of range")
        print(
            "VN97 P5E4G RESUME "
            f"step={start_step}/{total_steps} path={resume_path}",
            flush=True,
        )

    started = time.monotonic()
    final_loss = math.nan

    for step_index in range(start_step, total_steps):
        kind, batch_indices = schedule[step_index]
        if kind == "targeted":
            x = targeted_inputs[batch_indices]
            y = targeted_labels[batch_indices]
        else:
            x = replay_inputs[batch_indices]
            y = replay_labels[batch_indices]
        x = x.to(device=resolved, dtype=torch.long)
        y = y.to(device=resolved, dtype=torch.long)

        hidden, base = _hidden_and_base_logits(model, x)
        optimizer.zero_grad(set_to_none=True)
        delta = adapter(hidden)
        logits = base + delta

        if kind == "targeted":
            task_loss = _weighted_target_loss(logits, y)
            targeted_steps += 1
        else:
            task_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                y.reshape(-1),
                ignore_index=-100,
            )
            replay_steps += 1

        regularizer = delta.float().pow(2).mean()
        loss = task_loss + LOGIT_DELTA_L2 * regularizer
        if not bool(torch.isfinite(loss)):
            raise VN97P5E4GError("decoding repair loss became non-finite")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            adapter.parameters(),
            MAX_GRAD_NORM,
        )
        if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
            raise VN97P5E4GError("decoding repair gradient became non-finite")
        optimizer.step()

        completed = step_index + 1
        final_loss = float(task_loss.detach().cpu())
        loss_sum += final_loss

        if completed % PROGRESS_INTERVAL == 0 or completed == total_steps:
            elapsed = time.monotonic() - started
            local_steps = max(1, completed - start_step)
            eta = (total_steps - completed) * elapsed / local_steps
            print(
                "VN97 P5E4G PROGRESS "
                f"step={completed}/{total_steps} "
                f"percent={100.0 * completed / total_steps:.2f} "
                f"kind={kind} elapsed_s={elapsed:.1f} eta_s={eta:.1f} "
                f"loss={final_loss:.6f} "
                f"mean_loss={loss_sum / completed:.6f} "
                f"grad_norm={float(grad_norm):.6f} "
                f"targeted_steps={targeted_steps} "
                f"replay_steps={replay_steps}",
                flush=True,
            )

        if (
            completed % CHECKPOINT_INTERVAL == 0
            and completed < total_steps
        ):
            temp = resume_path.with_suffix(".tmp")
            torch.save(
                {
                    "schema": "VN97P5E4GRESUME1",
                    "identity": identity,
                    "next_step": completed,
                    "adapter_state_dict": {
                        name: tensor.detach().cpu()
                        for name, tensor in adapter.state_dict().items()
                    },
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss_sum": loss_sum,
                    "targeted_steps": targeted_steps,
                    "replay_steps": replay_steps,
                },
                temp,
            )
            temp.replace(resume_path)
            print(
                "VN97 P5E4G CHECKPOINT "
                f"step={completed}/{total_steps} path={resume_path}",
                flush=True,
            )

    return {
        "steps": total_steps,
        "targeted_steps": targeted_steps,
        "replay_steps": replay_steps,
        "mean_step_loss": loss_sum / total_steps,
        "final_loss": final_loss,
        "trainable_parameters": adapter.parameter_count(),
    }


def _gate(
    *,
    dev_before: dict[str, Any],
    dev_after: dict[str, Any],
    p3_before: dict[str, Any],
    p3_after: dict[str, Any],
) -> tuple[str, list[str]]:
    reasons: list[str] = []

    early_gain = (
        float(dev_after["first_top1_accuracy"])
        >= float(dev_before["first_top1_accuracy"]) + 0.02
        or float(dev_after["mean_first_rank"])
        <= float(dev_before["mean_first_rank"]) * 0.80
        or float(dev_after["top1_accuracy"])
        >= float(dev_before["top1_accuracy"]) + 0.01
    )
    if not early_gain:
        reasons.append("no_material_prefix_gain")

    if float(dev_after["mean_loss"]) > float(dev_before["mean_loss"]) * 1.02:
        reasons.append("dev_loss_over_2pct")
    if float(p3_after["mean_loss"]) > float(p3_before["mean_loss"]) * 1.015:
        reasons.append("p3_loss_over_1.5pct")
    if (
        float(p3_after["top1_accuracy"]) + 0.0075
        < float(p3_before["top1_accuracy"])
    ):
        reasons.append("p3_top1_drop_over_0.75pp")

    if reasons:
        return "PREFIX_DECODING_REPAIR_REJECTED", reasons
    return "PREFIX_DECODING_REPAIR_SIGNAL", []


def _write_torch_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.unlink(missing_ok=True)
    torch.save(payload, temp)
    temp.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise VN97P5E4GError("P5E4G requires CUDA")

    output = Path(args.output_dir)
    if output.exists() and (
        output.is_symlink()
        or not output.is_dir()
        or any(output.iterdir())
    ):
        raise VN97P5E4GError("output-dir must be new or empty")

    model, tokenizer, repaired_meta = _load_repaired(
        Path(args.repaired_dir)
    )
    set_float_shadow_mode(model, True)
    adapter, expanded_meta = _load_expansion(
        Path(args.expanded_dir),
        repaired_meta=repaired_meta,
        d_model=model.config.d_model,
        vocab_size=model.config.vocab_size,
    )
    p5e4f_report, p5e4f_sha = _verify_p5e4f_gate(
        Path(args.p5e4f_report),
        repaired_meta=repaired_meta,
        expanded_meta=expanded_meta,
    )

    core_before = _state_hash(model, exclude_final_norm=False)
    parent_adapter_sha = expanded_meta["adapter_sha256"]
    model.to(args.device)
    model.eval()
    adapter.to(args.device)

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

    targeted_records = _select_targeted_records()
    replay_records = _select_replay(
        p3_training_records,
        count=P3_REPLAY_RECORDS,
    )

    train_config = VN97TrainingConfig(
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        epochs=1,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        seed=SEED,
        shuffle=True,
        max_windows=30_000,
    )
    eval_config = VN97TrainingConfig(
        sequence_length=SEQUENCE_LENGTH,
        batch_size=1,
        epochs=1,
        seed=0,
        shuffle=False,
        max_windows=30_000,
    )

    targeted_examples = [
        encode_chat_completion_messages(tokenizer, item.messages)
        for item in targeted_records
    ]
    replay_examples = [
        encode_chat_completion_messages(tokenizer, messages)
        for messages in replay_records
    ]
    dev_examples = [
        encode_chat_completion_messages(tokenizer, item.messages)
        for item in default_validation()
    ]
    p3_examples = [
        encode_chat_completion_messages(tokenizer, messages)
        for messages in p3_validation_records
    ]

    targeted_windows = build_training_windows(
        targeted_examples,
        train_config,
        pad_token_id=tokenizer.pad_id,
    )
    replay_windows = build_training_windows(
        replay_examples,
        train_config,
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

    targeted_inputs, targeted_labels = _windows_to_tensors(
        targeted_windows
    )
    replay_inputs, replay_labels = _windows_to_tensors(
        replay_windows
    )
    dev_inputs, dev_labels = _windows_to_tensors(dev_windows)
    p3_inputs, p3_labels = _windows_to_tensors(p3_windows)

    dev_before = _evaluate(
        model=model,
        adapter=adapter,
        inputs=dev_inputs,
        labels=dev_labels,
        device=args.device,
    )
    p3_before = _evaluate(
        model=model,
        adapter=adapter,
        inputs=p3_inputs,
        labels=p3_labels,
        device=args.device,
    )

    print(
        "VN97 P5E4G START "
        f"p5e4f_status={p5e4f_report['status']} "
        f"parent_adapter_sha256={parent_adapter_sha} "
        f"targeted_records={len(targeted_records)} "
        f"replay_records={len(replay_records)} "
        f"dev_loss={float(dev_before['mean_loss']):.6f} "
        f"dev_top1={float(dev_before['top1_accuracy']):.6f} "
        f"dev_first_top1={float(dev_before['first_top1_accuracy']):.6f} "
        f"dev_first_rank={float(dev_before['mean_first_rank']):.2f} "
        f"p3_loss={float(p3_before['mean_loss']):.6f} "
        f"p3_top1={float(p3_before['top1_accuracy']):.6f}",
        flush=True,
    )

    identity_payload = json.dumps(
        {
            "profile_id": P5E4G_PROFILE_ID,
            "parent_adapter_sha256": parent_adapter_sha,
            "p5e4f_report_sha256": p5e4f_sha,
            "p3_training_sha256": p3_training_sha,
            "p3_validation_sha256": p3_validation_sha,
            "core_sha256": core_before,
            "targeted_per_category": TARGETED_PER_CATEGORY,
            "weights": [
                FIRST_TOKEN_WEIGHT,
                EARLY_4_WEIGHT,
                EARLY_16_WEIGHT,
                FINAL_TOKEN_WEIGHT,
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    identity = hashlib.sha256(
        b"VN97P5E4GRUN\0" + identity_payload
    ).hexdigest()

    training = _train(
        model=model,
        adapter=adapter,
        targeted_inputs=targeted_inputs,
        targeted_labels=targeted_labels,
        replay_inputs=replay_inputs,
        replay_labels=replay_labels,
        device=args.device,
        work_dir=Path(args.work_dir),
        identity=identity,
    )

    core_after = _state_hash(model, exclude_final_norm=False)
    if core_after != core_before:
        raise VN97P5E4GError(
            "frozen VN97 core changed during decoding repair"
        )

    dev_after = _evaluate(
        model=model,
        adapter=adapter,
        inputs=dev_inputs,
        labels=dev_labels,
        device=args.device,
    )
    p3_after = _evaluate(
        model=model,
        adapter=adapter,
        inputs=p3_inputs,
        labels=p3_labels,
        device=args.device,
    )
    status, reasons = _gate(
        dev_before=dev_before,
        dev_after=dev_after,
        p3_before=p3_before,
        p3_after=p3_after,
    )

    output.mkdir(parents=True, exist_ok=True)
    tokenizer_bytes = (
        Path(args.repaired_dir) / "tokenizer.vn97tk1"
    ).read_bytes()
    tokenizer_sha = _sha256_bytes(tokenizer_bytes)
    if tokenizer_sha != repaired_meta["tokenizer_sha256"]:
        raise VN97P5E4GError("repaired tokenizer hash changed")

    report = {
        "adapter": {
            "rank": ADAPTER_RANK,
            "trainable_parameters": adapter.parameter_count(),
            "parent_adapter_sha256": parent_adapter_sha,
            "training": training,
        },
        "curriculum": {
            "targeted_records_per_category": TARGETED_PER_CATEGORY,
            "p3_replay_records": P3_REPLAY_RECORDS,
            "first_token_weight": FIRST_TOKEN_WEIGHT,
            "early_4_weight": EARLY_4_WEIGHT,
            "early_16_weight": EARLY_16_WEIGHT,
            "final_token_weight": FINAL_TOKEN_WEIGHT,
        },
        "evaluations": {
            "dev_before": dev_before,
            "dev_after": dev_after,
            "p3_before": p3_before,
            "p3_after": p3_after,
        },
        "frozen_core_sha256": core_after,
        "p3_training_sha256": p3_training_sha,
        "p3_validation_sha256": p3_validation_sha,
        "p5e4f_report_sha256": p5e4f_sha,
        "p5e4f_status": p5e4f_report["status"],
        "profile_id": P5E4G_PROFILE_ID,
        "ready_for_fresh_validation": (
            status == "PREFIX_DECODING_REPAIR_SIGNAL"
        ),
        "ready_for_qat": False,
        "reasons": reasons,
        "repaired_student_sha256": repaired_meta["student_sha256"],
        "schema": P5E4G_SCHEMA,
        "status": status,
        "tokenizer_sha256": tokenizer_sha,
    }

    _atomic_write(
        output / "p5e4g-report.json",
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8") + b"\n",
    )

    if status == "PREFIX_DECODING_REPAIR_SIGNAL":
        payload = {
            "schema": P5E4G_ADAPTER_SCHEMA,
            "profile_id": P5E4G_PROFILE_ID,
            "status": status,
            "rank": ADAPTER_RANK,
            "d_model": model.config.d_model,
            "vocab_size": model.config.vocab_size,
            "repaired_student_sha256": repaired_meta["student_sha256"],
            "parent_adapter_sha256": parent_adapter_sha,
            "frozen_core_sha256": core_after,
            "p5e4f_report_sha256": p5e4f_sha,
            "tokenizer_sha256": tokenizer_sha,
            "ready_for_fresh_validation": True,
            "ready_for_qat": False,
            "adapter_state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in adapter.state_dict().items()
            },
        }
        _write_torch_atomic(
            output / "output-adapter.pt",
            payload,
        )
        _atomic_write(
            output / "tokenizer.vn97tk1",
            tokenizer_bytes,
        )
        sums = {
            "output-adapter.pt": _sha256_file(
                output / "output-adapter.pt"
            ),
            "p5e4g-report.json": _sha256_file(
                output / "p5e4g-report.json"
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
        "VN97P5E4G "
        f"status={status} "
        f"dev_loss_before={float(dev_before['mean_loss']):.6f} "
        f"dev_loss_after={float(dev_after['mean_loss']):.6f} "
        f"dev_top1_before={float(dev_before['top1_accuracy']):.6f} "
        f"dev_top1_after={float(dev_after['top1_accuracy']):.6f} "
        f"dev_first_top1_before={float(dev_before['first_top1_accuracy']):.6f} "
        f"dev_first_top1_after={float(dev_after['first_top1_accuracy']):.6f} "
        f"dev_first_rank_before={float(dev_before['mean_first_rank']):.2f} "
        f"dev_first_rank_after={float(dev_after['mean_first_rank']):.2f} "
        f"p3_loss_before={float(p3_before['mean_loss']):.6f} "
        f"p3_loss_after={float(p3_after['mean_loss']):.6f} "
        f"p3_top1_before={float(p3_before['top1_accuracy']):.6f} "
        f"p3_top1_after={float(p3_after['top1_accuracy']):.6f} "
        f"frozen_core_match={str(core_before == core_after).lower()} "
        f"ready_for_fresh_validation="
        f"{str(status == 'PREFIX_DECODING_REPAIR_SIGNAL').lower()} "
        "ready_for_qat=false "
        f"reasons={','.join(reasons) if reasons else 'none'}",
        flush=True,
    )
    print(f"P5E4G final: {output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-p5e4g-repair: {exc}", file=sys.stderr)
        raise
