from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .p4_generalization_cli import _select_replay, _windows_to_tensors
from .p4_generalization_curriculum import default_validation
from .p5e4c_decoder_repair_cli import (
    P3_REPLAY_RECORDS,
    SEQUENCE_LENGTH,
    _select_calibration_records,
    _sha256_file,
    _state_hash,
)
from .p5e4d_fresh_decoder_validation_cli import P5E4D_SCHEMA, _load_repaired
from .quantization import set_float_shadow_mode
from .training import (
    VN97TrainingConfig,
    build_training_windows,
    encode_chat_completion_messages,
)
from .training_cli import _atomic_write, _load_records


P5E4E_SCHEMA = "VN97P5E4E"
P5E4E_ADAPTER_SCHEMA = "VN97P5E4EOUTPUT1"
P5E4E_PROFILE_ID = "vn97-p5e4e-output-residual-expansion-v1"

ADAPTER_RANK = 32
BATCH_SIZE = 2
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
LOGIT_DELTA_L2 = 1e-5
PROGRESS_INTERVAL = 25
CHECKPOINT_INTERVAL = 50
SEED = 9755


class VN97P5E4EError(RuntimeError):
    pass


class ResidualOutputAdapter(nn.Module):
    """Low-rank output-only residual over the frozen VN97 lexical head."""

    def __init__(self, d_model: int, vocab_size: int, rank: int) -> None:
        super().__init__()
        if d_model <= 0 or vocab_size <= 1 or not 0 < rank < d_model:
            raise VN97P5E4EError("invalid residual output adapter shape")
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.rank = rank
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, vocab_size, bias=False)
        nn.init.normal_(
            self.down.weight,
            mean=0.0,
            std=1.0 / math.sqrt(d_model),
        )
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(hidden))

    def parameter_count(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "P5E4E output expansion: train only a low-rank residual output "
            "adapter over P5E4C while freezing SSM, input embedding, tied "
            "lexical head, and final RMSNorm."
        )
    )
    parser.add_argument("--repaired-dir", required=True)
    parser.add_argument("--p5e4d-report", required=True)
    parser.add_argument("--p3-corpus-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verify_p5e4d_gate(
    path: Path,
    *,
    repaired_meta: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    resolved = path.resolve(strict=True)
    raw = resolved.read_bytes()
    report = json.loads(raw.decode("utf-8"))
    if not isinstance(report, dict) or report.get("schema") != P5E4D_SCHEMA:
        raise VN97P5E4EError("invalid P5E4D report schema")
    if report.get("one_shot") is not True:
        raise VN97P5E4EError("P5E4D evidence is not one-shot")
    if report.get("immutable_state_match") is not True:
        raise VN97P5E4EError("P5E4D immutable-state evidence failed")
    if report.get("ready_for_qat") is not False:
        raise VN97P5E4EError("P5E4E cannot run after QAT authorization")
    if report.get("ready_for_output_expansion") is not True:
        raise VN97P5E4EError("P5E4D did not authorize output expansion")
    if report.get("status") not in {
        "FRESH_DECODER_LATENT_GAIN",
        "FRESH_DECODER_PRESERVED_ZERO_GENERATION",
        "FRESH_DECODER_PRESERVED",
    }:
        raise VN97P5E4EError("P5E4D status is not output-expansion eligible")
    artifacts = report.get("artifacts")
    repaired = artifacts.get("repaired") if isinstance(artifacts, dict) else None
    if not isinstance(repaired, dict):
        raise VN97P5E4EError("P5E4D repaired identity missing")
    if repaired.get("student_sha256") != repaired_meta.get("student_sha256"):
        raise VN97P5E4EError("P5E4D/P5E4C checkpoint identity mismatch")
    return report, _sha256_bytes(raw)


def _hidden_and_base_logits(model, inputs: torch.Tensor):
    with torch.no_grad():
        embeds = model.embedding(inputs)
        hidden, _ = model.forward_hidden_embeddings_sequential_reference(
            embeds,
            None,
        )
        logits = model.lm_head(hidden)
    return hidden.detach(), logits.detach()


@torch.inference_mode()
def _evaluate(
    *,
    model,
    adapter: ResidualOutputAdapter,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    device: str,
) -> dict[str, float | int]:
    if int(inputs.shape[0]) != int(labels.shape[0]):
        raise VN97P5E4EError("evaluation input/label count mismatch")
    resolved = torch.device(device)
    model.eval()
    adapter.eval()
    loss_sum = 0.0
    target_tokens = 0
    correct = 0
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
        if count == 0:
            continue
        selected_logits = logits.reshape(-1, logits.shape[-1])[mask]
        selected_y = flat_y[mask]
        loss_sum += float(
            F.cross_entropy(
                selected_logits,
                selected_y,
                reduction="sum",
            ).item()
        )
        target_tokens += count
        correct += int(
            (selected_logits.argmax(dim=-1) == selected_y).sum().item()
        )
    if target_tokens <= 0:
        raise VN97P5E4EError("evaluation produced no target tokens")
    return {
        "windows": windows,
        "target_tokens": target_tokens,
        "mean_loss": loss_sum / target_tokens,
        "top1_accuracy": correct / target_tokens,
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
    inputs: torch.Tensor,
    labels: torch.Tensor,
    device: str,
    work_dir: Path,
    identity: str,
) -> dict[str, float | int]:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise VN97P5E4EError("P5E4E requires CUDA")
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

    order = list(range(int(inputs.shape[0])))
    random.Random(SEED).shuffle(order)
    batches = [
        order[start : start + BATCH_SIZE]
        for start in range(0, len(order), BATCH_SIZE)
    ]
    total_steps = len(batches)
    if total_steps <= 0:
        raise VN97P5E4EError("P5E4E training set is empty")

    work_dir.mkdir(parents=True, exist_ok=True)
    resume_path = work_dir / "state.p5e4e.pt"
    start_step = 0
    loss_sum = 0.0
    target_tokens = 0
    if resume_path.is_file():
        resume = torch.load(
            resume_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(resume, dict) or resume.get("identity") != identity:
            raise VN97P5E4EError("P5E4E resume identity mismatch")
        adapter.load_state_dict(resume["adapter_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        _move_optimizer_state(optimizer, resolved)
        start_step = int(resume.get("next_step", 0))
        loss_sum = float(resume.get("loss_sum", 0.0))
        target_tokens = int(resume.get("target_tokens", 0))
        if not 0 <= start_step <= total_steps:
            raise VN97P5E4EError("P5E4E resume step out of range")
        print(
            "VN97 P5E4E RESUME "
            f"step={start_step}/{total_steps} path={resume_path}",
            flush=True,
        )

    started = time.monotonic()
    final_loss = math.nan
    for step_index in range(start_step, total_steps):
        batch_indices = batches[step_index]
        x = inputs[batch_indices].to(
            device=resolved,
            dtype=torch.long,
        )
        y = labels[batch_indices].to(
            device=resolved,
            dtype=torch.long,
        )
        hidden, base = _hidden_and_base_logits(model, x)
        optimizer.zero_grad(set_to_none=True)
        delta = adapter(hidden)
        logits = base + delta
        task_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
            ignore_index=-100,
        )
        regularizer = delta.float().pow(2).mean()
        loss = task_loss + LOGIT_DELTA_L2 * regularizer
        if not bool(torch.isfinite(loss)):
            raise VN97P5E4EError("output expansion loss became non-finite")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            adapter.parameters(),
            MAX_GRAD_NORM,
        )
        if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
            raise VN97P5E4EError("output expansion gradient became non-finite")
        optimizer.step()

        completed = step_index + 1
        final_loss = float(task_loss.detach().cpu())
        loss_sum += final_loss
        target_tokens += int((y != -100).sum().item())

        if completed % PROGRESS_INTERVAL == 0 or completed == total_steps:
            elapsed = time.monotonic() - started
            local_steps = max(1, completed - start_step)
            eta = (total_steps - completed) * elapsed / local_steps
            print(
                "VN97 P5E4E PROGRESS "
                f"step={completed}/{total_steps} "
                f"percent={100.0 * completed / total_steps:.2f} "
                f"elapsed_s={elapsed:.1f} eta_s={eta:.1f} "
                f"loss={final_loss:.6f} "
                f"mean_loss={loss_sum / completed:.6f} "
                f"grad_norm={float(grad_norm):.6f}",
                flush=True,
            )

        if (
            completed % CHECKPOINT_INTERVAL == 0
            and completed < total_steps
        ):
            temp = resume_path.with_suffix(".tmp")
            torch.save(
                {
                    "schema": "VN97P5E4ERESUME1",
                    "identity": identity,
                    "next_step": completed,
                    "adapter_state_dict": {
                        name: tensor.detach().cpu()
                        for name, tensor in adapter.state_dict().items()
                    },
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss_sum": loss_sum,
                    "target_tokens": target_tokens,
                },
                temp,
            )
            temp.replace(resume_path)
            print(
                "VN97 P5E4E CHECKPOINT "
                f"step={completed}/{total_steps} path={resume_path}",
                flush=True,
            )

    return {
        "steps": total_steps,
        "target_tokens": target_tokens,
        "mean_step_loss": loss_sum / total_steps,
        "final_loss": final_loss,
        "trainable_parameters": adapter.parameter_count(),
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
        float(dev_after["mean_loss"])
        <= float(dev_before["mean_loss"]) * 0.985
        or float(dev_after["top1_accuracy"])
        >= float(dev_before["top1_accuracy"]) + 0.0075
    )
    if not dev_gain:
        reasons.append("output_dev_no_material_gain")
    if float(p3_after["mean_loss"]) > float(p3_before["mean_loss"]) * 1.02:
        reasons.append("p3_loss_over_2pct")
    if (
        float(p3_after["top1_accuracy"]) + 0.01
        < float(p3_before["top1_accuracy"])
    ):
        reasons.append("p3_top1_drop_over_1pp")
    if reasons:
        return "OUTPUT_EXPANSION_REJECTED", reasons
    return "OUTPUT_EXPANSION_SIGNAL", []


def _write_torch_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.unlink(missing_ok=True)
    torch.save(payload, temp)
    temp.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise VN97P5E4EError("P5E4E requires CUDA")

    output = Path(args.output_dir)
    if output.exists() and (
        output.is_symlink()
        or not output.is_dir()
        or any(output.iterdir())
    ):
        raise VN97P5E4EError("output-dir must be new or empty")

    model, tokenizer, repaired_meta = _load_repaired(
        Path(args.repaired_dir)
    )
    set_float_shadow_mode(model, True)
    p5e4d_report, p5e4d_sha = _verify_p5e4d_gate(
        Path(args.p5e4d_report),
        repaired_meta=repaired_meta,
    )
    if model.config.embedding_rank is None:
        raise VN97P5E4EError(
            "P5E4E expects the canonical factorized lexical interface"
        )

    core_before = _state_hash(model, exclude_final_norm=False)
    model.to(args.device)
    model.eval()

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
    train_messages = [item.messages for item in calibration_records]
    train_messages.extend(replay_records)

    train_config = VN97TrainingConfig(
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        epochs=1,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        seed=SEED,
        shuffle=True,
        max_windows=20_000,
    )
    eval_config = VN97TrainingConfig(
        sequence_length=SEQUENCE_LENGTH,
        batch_size=1,
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
    train_inputs, train_labels = _windows_to_tensors(train_windows)
    dev_inputs, dev_labels = _windows_to_tensors(dev_windows)
    p3_inputs, p3_labels = _windows_to_tensors(p3_windows)

    adapter = ResidualOutputAdapter(
        model.config.d_model,
        model.config.vocab_size,
        ADAPTER_RANK,
    ).to(args.device)

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
        "VN97 P5E4E START "
        f"rank={ADAPTER_RANK} "
        f"trainable_parameters={adapter.parameter_count()} "
        f"frozen_core_sha256={core_before} "
        f"p5e4d_status={p5e4d_report['status']} "
        f"dev_loss={float(dev_before['mean_loss']):.6f} "
        f"dev_top1={float(dev_before['top1_accuracy']):.6f} "
        f"p3_loss={float(p3_before['mean_loss']):.6f} "
        f"p3_top1={float(p3_before['top1_accuracy']):.6f}",
        flush=True,
    )

    identity_payload = json.dumps(
        {
            "profile_id": P5E4E_PROFILE_ID,
            "rank": ADAPTER_RANK,
            "repaired_student_sha256": repaired_meta["student_sha256"],
            "p5e4d_report_sha256": p5e4d_sha,
            "p3_training_sha256": p3_training_sha,
            "p3_validation_sha256": p3_validation_sha,
            "core_sha256": core_before,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    identity = hashlib.sha256(
        b"VN97P5E4ERUN\0" + identity_payload
    ).hexdigest()

    training = _train(
        model=model,
        adapter=adapter,
        inputs=train_inputs,
        labels=train_labels,
        device=args.device,
        work_dir=Path(args.work_dir),
        identity=identity,
    )

    core_after = _state_hash(model, exclude_final_norm=False)
    if core_after != core_before:
        raise VN97P5E4EError(
            "frozen VN97 core changed during output expansion"
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
        raise VN97P5E4EError("repaired tokenizer hash changed")

    report = {
        "adapter": {
            "rank": ADAPTER_RANK,
            "trainable_parameters": adapter.parameter_count(),
            "base_head_frozen": True,
            "input_embedding_frozen": True,
            "recurrent_core_frozen": True,
            "final_norm_frozen": True,
            "training": training,
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
        "p5e4d_report_sha256": p5e4d_sha,
        "p5e4d_status": p5e4d_report["status"],
        "profile_id": P5E4E_PROFILE_ID,
        "ready_for_fresh_validation": (
            status == "OUTPUT_EXPANSION_SIGNAL"
        ),
        "ready_for_qat": False,
        "reasons": reasons,
        "repaired_student_sha256": repaired_meta["student_sha256"],
        "schema": P5E4E_SCHEMA,
        "status": status,
        "tokenizer_sha256": tokenizer_sha,
    }
    _atomic_write(
        output / "p5e4e-report.json",
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8") + b"\n",
    )

    if status == "OUTPUT_EXPANSION_SIGNAL":
        payload = {
            "schema": P5E4E_ADAPTER_SCHEMA,
            "profile_id": P5E4E_PROFILE_ID,
            "status": status,
            "rank": ADAPTER_RANK,
            "d_model": model.config.d_model,
            "vocab_size": model.config.vocab_size,
            "repaired_student_sha256": repaired_meta["student_sha256"],
            "frozen_core_sha256": core_after,
            "p5e4d_report_sha256": p5e4d_sha,
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
            "p5e4e-report.json": _sha256_file(
                output / "p5e4e-report.json"
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
        "VN97P5E4E "
        f"status={status} "
        f"dev_loss_before={float(dev_before['mean_loss']):.6f} "
        f"dev_loss_after={float(dev_after['mean_loss']):.6f} "
        f"dev_top1_before={float(dev_before['top1_accuracy']):.6f} "
        f"dev_top1_after={float(dev_after['top1_accuracy']):.6f} "
        f"p3_loss_before={float(p3_before['mean_loss']):.6f} "
        f"p3_loss_after={float(p3_after['mean_loss']):.6f} "
        f"p3_top1_before={float(p3_before['top1_accuracy']):.6f} "
        f"p3_top1_after={float(p3_after['top1_accuracy']):.6f} "
        f"frozen_core_match={str(core_before == core_after).lower()} "
        f"ready_for_fresh_validation="
        f"{str(status == 'OUTPUT_EXPANSION_SIGNAL').lower()} "
        "ready_for_qat=false "
        f"reasons={','.join(reasons) if reasons else 'none'}",
        flush=True,
    )
    print(f"P5E4E final: {output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-p5e4e-expand: {exc}", file=sys.stderr)
        raise
