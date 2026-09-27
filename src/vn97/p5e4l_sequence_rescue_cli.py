from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .chat_boundary import recover_chat_response_text
from .cognition_adapter import TorchVN97InferenceEngine, VN97InferenceLimits
from .p4_generalization_cli import _select_replay, _windows_to_tensors
from .p5e3c_target_rank_diagnostic_cli import _score_target
from .p5e4b_fresh_validation_cli import _build_probes
from .p5e4c_decoder_repair_cli import (
    P3_REPLAY_RECORDS,
    SEQUENCE_LENGTH,
    _sha256_file,
    _state_hash,
)
from .p5e4d_fresh_decoder_validation_cli import _load_repaired
from .p5e4e_output_expansion_cli import ResidualOutputAdapter
from .p5e4f_fresh_output_validation_cli import (
    VN97OutputExpandedModel,
    _load_expansion,
)
from .p5e4h_fresh_post_prefix_validation_cli import _load_prefix_repaired
from .p5e4k_generation_constraint_ablation_cli import P5E4K_SCHEMA
from .quantization import set_float_shadow_mode
from .training import (
    VN97ChatMessage,
    VN97TrainingConfig,
    build_training_windows,
    encode_chat_completion_messages,
    encode_chat_completion_prompt,
)
from .training_cli import _atomic_write, _load_records


P5E4L_SCHEMA = "VN97P5E4L"
P5E4L_ADAPTER_SCHEMA = "VN97P5E4LOUTPUT1"
P5E4L_PROFILE_ID = "vn97-p5e4l-high-capacity-sequence-rescue-v1"

PARENT_RANK = 32
CORRECTION_RANK = 64
MERGED_RANK = PARENT_RANK + CORRECTION_RANK

TRAIN_SHARDS = 4
TRAIN_PER_CATEGORY_PER_SHARD = 64
DEV_PER_CATEGORY = 16
HOLDOUT_PER_CATEGORY = 24
MAX_EPOCHS = 4
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
LOGIT_DELTA_L2 = 2e-6
EARLY_8_WEIGHT = 2.0
EARLY_32_WEIGHT = 1.25
PROGRESS_INTERVAL = 50
SEED = 9758
MIN_QAT_EXACT_RATE = 0.10
MIN_EXACT_GAIN = 8


class VN97P5E4LError(RuntimeError):
    pass


class StackedOutputAdapter(nn.Module):
    def __init__(
        self,
        parent: ResidualOutputAdapter,
        correction: ResidualOutputAdapter,
    ) -> None:
        super().__init__()
        self.parent = parent
        self.correction = correction
        for parameter in self.parent.parameters():
            parameter.requires_grad_(False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            parent_delta = self.parent(hidden)
        return parent_delta + self.correction(hidden)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "P5E4L one-run rescue: keep the accepted P5E4G adapter frozen, "
            "add a rank-64 correction, train on a large disjoint synthetic "
            "sequence curriculum plus P3 replay, select by fresh dev exact "
            "generation, then run a final untouched holdout gate."
        )
    )
    parser.add_argument("--repaired-dir", required=True)
    parser.add_argument("--expanded-dir", required=True)
    parser.add_argument("--prefix-repaired-dir", required=True)
    parser.add_argument("--p5e4k-report", required=True)
    parser.add_argument("--p3-corpus-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    return parser


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verify_p5e4k(
    path: Path,
    *,
    repaired_meta: dict[str, Any],
    prefix_meta: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    resolved = path.resolve(strict=True)
    raw = resolved.read_bytes()
    report = json.loads(raw.decode("utf-8"))
    if not isinstance(report, dict) or report.get("schema") != P5E4K_SCHEMA:
        raise VN97P5E4LError("invalid P5E4K report schema")
    if report.get("one_shot") is not True:
        raise VN97P5E4LError("P5E4K evidence is not one-shot")
    if report.get("ready_for_qat") is not False:
        raise VN97P5E4LError("P5E4L cannot run after QAT authorization")
    if report.get("ready_for_policy_promotion") is not False:
        raise VN97P5E4LError(
            "P5E4L expects policy ablation to have no promotable decoder policy"
        )
    if report.get("status") not in {
        "DECODING_POLICY_NO_MATERIAL_GAIN",
        "DECODING_POLICY_PREFIX_GAIN_ONLY",
    }:
        raise VN97P5E4LError("P5E4K status is not rescue eligible")

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
        raise VN97P5E4LError("P5E4K artifact identities missing")
    if repaired.get("student_sha256") != repaired_meta.get("student_sha256"):
        raise VN97P5E4LError("P5E4K/P5E4C checkpoint identity mismatch")
    if prefix.get("adapter_sha256") != prefix_meta.get("adapter_sha256"):
        raise VN97P5E4LError("P5E4K/P5E4G adapter identity mismatch")
    return report, hashlib.sha256(raw).hexdigest()


def _seed(root: str, label: str, index: int = 0) -> str:
    return hashlib.sha256(
        (
            P5E4L_PROFILE_ID
            + "\0"
            + root
            + "\0"
            + label
            + "\0"
            + str(index)
        ).encode("utf-8")
    ).hexdigest()


def _build_train_probes(root_seed: str):
    probes = []
    seen: set[str] = set()
    for shard in range(TRAIN_SHARDS):
        shard_seed = _seed(root_seed, "train", shard)
        for probe in _build_probes(
            seed_hex=shard_seed,
            per_category=TRAIN_PER_CATEGORY_PER_SHARD,
        ):
            identity = hashlib.sha256(
                (
                    probe.category
                    + "\0"
                    + probe.prompt
                    + "\0"
                    + probe.target
                ).encode("utf-8")
            ).hexdigest()
            if identity in seen:
                continue
            seen.add(identity)
            probes.append(probe)
    return tuple(probes)


def _messages_for_probe(probe) -> tuple[VN97ChatMessage, VN97ChatMessage]:
    return (
        VN97ChatMessage(role="user", content=probe.prompt),
        VN97ChatMessage(role="assistant", content=probe.target),
    )


def _weighted_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    mask = labels != -100
    if not bool(mask.any()):
        raise VN97P5E4LError("batch contains no supervised tokens")

    weights = torch.zeros_like(labels, dtype=torch.float32)
    for row in range(int(labels.shape[0])):
        positions = torch.nonzero(
            labels[row] != -100,
            as_tuple=False,
        ).flatten()
        if positions.numel() == 0:
            continue
        weights[row, positions] = 1.0
        weights[row, positions[:32]] = EARLY_32_WEIGHT
        weights[row, positions[:8]] = EARLY_8_WEIGHT

    flat_logits = logits[mask].float()
    flat_labels = labels[mask]
    flat_weights = weights[mask].to(logits.device)
    per_token = F.cross_entropy(
        flat_logits,
        flat_labels,
        reduction="none",
    )
    return (per_token * flat_weights).sum() / flat_weights.sum()


def _hidden_and_base(model, inputs: torch.Tensor):
    with torch.no_grad():
        hidden, _ = model.forward_hidden_embeddings_sequential_reference(
            model.embedding(inputs),
            None,
        )
        base = model.lm_head(hidden)
    return hidden.detach(), base.detach()


@torch.inference_mode()
def _p3_eval(
    *,
    model,
    adapter,
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
    for start in range(0, int(inputs.shape[0]), BATCH_SIZE):
        x = inputs[start : start + BATCH_SIZE].to(
            device=resolved,
            dtype=torch.long,
        )
        y = labels[start : start + BATCH_SIZE].to(
            device=resolved,
            dtype=torch.long,
        )
        hidden, base = _hidden_and_base(model, x)
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
                selected_logits.float(),
                selected_y,
                reduction="sum",
            ).item()
        )
        target_tokens += count
        correct += int(
            (selected_logits.argmax(dim=-1) == selected_y).sum().item()
        )
    if target_tokens <= 0:
        raise VN97P5E4LError("P3 evaluation produced no target tokens")
    return {
        "target_tokens": target_tokens,
        "mean_loss": loss_sum / target_tokens,
        "top1_accuracy": correct / target_tokens,
    }


def _char_prefix_ratio(generated: str, target: str) -> float:
    if not target:
        return 1.0 if not generated else 0.0
    count = 0
    for left, right in zip(generated, target):
        if left != right:
            break
        count += 1
    return count / len(target)


@torch.inference_mode()
def _generation_eval(
    *,
    model,
    tokenizer,
    probes,
    device: str,
    label: str,
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

    exact = 0
    errors = 0
    prefix_sum = 0.0
    nonempty = 0
    first_top1 = 0
    first_rank_sum = 0
    token_top1 = 0
    token_count = 0
    nll_sum = 0.0

    for index, probe in enumerate(probes, start=1):
        latent = _score_target(
            model=model,
            tokenizer=tokenizer,
            prompt=probe.prompt,
            target_text=probe.target,
            device=device,
        )
        first_rank = int(latent["first_rank"])
        first_top1 += int(first_rank == 1)
        first_rank_sum += first_rank
        token_top1 += int(latent["top1_tokens"])
        token_count += int(latent["token_count"])
        nll_sum += float(latent["nll_sum"])

        recovered = ""
        error = ""
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
            error = type(exc).__name__

        ok = not error and recovered == probe.target
        exact += int(ok)
        errors += int(bool(error))
        nonempty += int(bool(recovered))
        prefix_sum += _char_prefix_ratio(recovered, probe.target)

        if index % 24 == 0 or index == len(probes):
            print(
                "VN97 P5E4L EVAL "
                f"label={label} tasks={index}/{len(probes)} "
                f"exact={exact} "
                f"prefix={prefix_sum / index:.6f} "
                f"first_top1={first_top1 / index:.6f}",
                flush=True,
            )

    tasks = len(probes)
    return {
        "tasks": tasks,
        "exact_generation": exact,
        "exact_generation_rate": exact / tasks,
        "generation_errors": errors,
        "nonempty_generation_rate": nonempty / tasks,
        "mean_char_prefix_ratio": prefix_sum / tasks,
        "first_top1_rate": first_top1 / tasks,
        "mean_first_rank": first_rank_sum / tasks,
        "token_top1_rate": token_top1 / max(token_count, 1),
        "mean_nll": nll_sum / max(token_count, 1),
        "target_tokens": token_count,
    }


def _merge_adapter(
    parent: ResidualOutputAdapter,
    correction: ResidualOutputAdapter,
) -> ResidualOutputAdapter:
    if parent.rank != PARENT_RANK or correction.rank != CORRECTION_RANK:
        raise VN97P5E4LError("adapter rank invariant failed")
    merged = ResidualOutputAdapter(
        d_model=parent.d_model,
        vocab_size=parent.vocab_size,
        rank=MERGED_RANK,
    )
    with torch.no_grad():
        merged.down.weight[:PARENT_RANK].copy_(
            parent.down.weight.detach().cpu()
        )
        merged.down.weight[PARENT_RANK:].copy_(
            correction.down.weight.detach().cpu()
        )
        merged.up.weight[:, :PARENT_RANK].copy_(
            parent.up.weight.detach().cpu()
        )
        merged.up.weight[:, PARENT_RANK:].copy_(
            correction.up.weight.detach().cpu()
        )
    return merged


def _train_epoch(
    *,
    model,
    stacked: StackedOutputAdapter,
    optimizer,
    targeted_inputs,
    targeted_labels,
    replay_inputs,
    replay_labels,
    device: str,
    epoch: int,
) -> dict[str, Any]:
    resolved = torch.device(device)
    stacked.train()
    rng = random.Random(SEED + epoch)

    schedule: list[tuple[str, list[int]]] = []
    targeted_order = list(range(int(targeted_inputs.shape[0])))
    replay_order = list(range(int(replay_inputs.shape[0])))
    rng.shuffle(targeted_order)
    rng.shuffle(replay_order)

    for start in range(0, len(targeted_order), BATCH_SIZE):
        schedule.append(
            ("targeted", targeted_order[start : start + BATCH_SIZE])
        )
    for start in range(0, len(replay_order), BATCH_SIZE):
        schedule.append(
            ("replay", replay_order[start : start + BATCH_SIZE])
        )
    rng.shuffle(schedule)

    loss_sum = 0.0
    targeted_steps = 0
    replay_steps = 0
    started = time.monotonic()

    for step, (kind, indices) in enumerate(schedule, start=1):
        if kind == "targeted":
            x = targeted_inputs[indices]
            y = targeted_labels[indices]
        else:
            x = replay_inputs[indices]
            y = replay_labels[indices]

        x = x.to(device=resolved, dtype=torch.long)
        y = y.to(device=resolved, dtype=torch.long)
        hidden, base = _hidden_and_base(model, x)

        optimizer.zero_grad(set_to_none=True)
        delta = stacked(hidden)
        logits = base + delta

        if kind == "targeted":
            task_loss = _weighted_ce(logits, y)
            targeted_steps += 1
        else:
            task_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                y.reshape(-1),
                ignore_index=-100,
            )
            replay_steps += 1

        regularizer = stacked.correction(hidden).float().pow(2).mean()
        loss = task_loss + LOGIT_DELTA_L2 * regularizer
        if not bool(torch.isfinite(loss)):
            raise VN97P5E4LError("sequence rescue loss became non-finite")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            stacked.correction.parameters(),
            MAX_GRAD_NORM,
        )
        if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
            raise VN97P5E4LError("sequence rescue gradient became non-finite")
        optimizer.step()

        loss_sum += float(task_loss.detach().cpu())
        if step % PROGRESS_INTERVAL == 0 or step == len(schedule):
            elapsed = time.monotonic() - started
            eta = (len(schedule) - step) * elapsed / max(step, 1)
            print(
                "VN97 P5E4L TRAIN "
                f"epoch={epoch} step={step}/{len(schedule)} "
                f"percent={100.0 * step / len(schedule):.2f} "
                f"kind={kind} loss={float(task_loss.detach().cpu()):.6f} "
                f"mean_loss={loss_sum / step:.6f} "
                f"grad_norm={float(grad_norm):.6f} "
                f"elapsed_s={elapsed:.1f} eta_s={eta:.1f}",
                flush=True,
            )

    return {
        "steps": len(schedule),
        "targeted_steps": targeted_steps,
        "replay_steps": replay_steps,
        "mean_step_loss": loss_sum / len(schedule),
        "elapsed_s": time.monotonic() - started,
    }


def _write_torch_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.unlink(missing_ok=True)
    torch.save(payload, temp)
    temp.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise VN97P5E4LError("P5E4L requires CUDA")
    if args.max_epochs < 1 or args.max_epochs > MAX_EPOCHS:
        raise VN97P5E4LError(
            f"max-epochs must be in [1, {MAX_EPOCHS}]"
        )

    output = Path(args.output_dir)
    if output.exists() and (
        output.is_symlink()
        or not output.is_dir()
        or any(output.iterdir())
    ):
        raise VN97P5E4LError("output-dir must be new or empty")

    model, tokenizer, repaired_meta = _load_repaired(
        Path(args.repaired_dir)
    )
    set_float_shadow_mode(model, True)

    p5e4e_adapter, p5e4e_meta = _load_expansion(
        Path(args.expanded_dir),
        repaired_meta=repaired_meta,
        d_model=model.config.d_model,
        vocab_size=model.config.vocab_size,
    )
    del p5e4e_adapter

    parent, prefix_meta = _load_prefix_repaired(
        Path(args.prefix_repaired_dir),
        repaired_meta=repaired_meta,
        parent_meta=p5e4e_meta,
        d_model=model.config.d_model,
        vocab_size=model.config.vocab_size,
    )
    if parent.rank != PARENT_RANK:
        raise VN97P5E4LError("P5E4G parent rank is not canonical rank-32")

    p5e4k_report, p5e4k_sha = _verify_p5e4k(
        Path(args.p5e4k_report),
        repaired_meta=repaired_meta,
        prefix_meta=prefix_meta,
    )

    core_before = _state_hash(model, exclude_final_norm=False)
    model.to(args.device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    parent = parent.to(args.device)
    parent.eval()
    for parameter in parent.parameters():
        parameter.requires_grad_(False)

    correction = ResidualOutputAdapter(
        d_model=model.config.d_model,
        vocab_size=model.config.vocab_size,
        rank=CORRECTION_RANK,
    ).to(args.device)
    stacked = StackedOutputAdapter(parent, correction).to(args.device)

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
    replay_records = _select_replay(
        p3_training_records,
        count=P3_REPLAY_RECORDS,
    )

    root_seed = hashlib.sha256(
        (
            str(repaired_meta["student_sha256"])
            + "\0"
            + str(prefix_meta["adapter_sha256"])
            + "\0"
            + p5e4k_sha
        ).encode("utf-8")
    ).hexdigest()

    train_probes = _build_train_probes(root_seed)
    dev_probes = tuple(
        _build_probes(
            seed_hex=_seed(root_seed, "dev"),
            per_category=DEV_PER_CATEGORY,
        )
    )
    holdout_probes = tuple(
        _build_probes(
            seed_hex=_seed(root_seed, "holdout"),
            per_category=HOLDOUT_PER_CATEGORY,
        )
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
        max_windows=40_000,
    )
    eval_config = VN97TrainingConfig(
        sequence_length=SEQUENCE_LENGTH,
        batch_size=1,
        epochs=1,
        seed=0,
        shuffle=False,
        max_windows=40_000,
    )

    targeted_examples = [
        encode_chat_completion_messages(
            tokenizer,
            _messages_for_probe(probe),
        )
        for probe in train_probes
    ]
    replay_examples = [
        encode_chat_completion_messages(tokenizer, messages)
        for messages in replay_records
    ]
    p3_examples = [
        encode_chat_completion_messages(tokenizer, messages)
        for messages in p3_validation_records
    ]

    targeted_inputs, targeted_labels = _windows_to_tensors(
        build_training_windows(
            targeted_examples,
            train_config,
            pad_token_id=tokenizer.pad_id,
        )
    )
    replay_inputs, replay_labels = _windows_to_tensors(
        build_training_windows(
            replay_examples,
            train_config,
            pad_token_id=tokenizer.pad_id,
        )
    )
    p3_inputs, p3_labels = _windows_to_tensors(
        build_training_windows(
            p3_examples,
            eval_config,
            pad_token_id=tokenizer.pad_id,
        )
    )

    parent_model = VN97OutputExpandedModel(
        model,
        parent,
    )
    parent_dev = _generation_eval(
        model=parent_model,
        tokenizer=tokenizer,
        probes=dev_probes,
        device=args.device,
        label="parent-dev",
    )
    parent_p3 = _p3_eval(
        model=model,
        adapter=parent,
        inputs=p3_inputs,
        labels=p3_labels,
        device=args.device,
    )

    print(
        "VN97 P5E4L START "
        f"p5e4k_status={p5e4k_report['status']} "
        f"parent_adapter_sha256={prefix_meta['adapter_sha256']} "
        f"train_probes={len(train_probes)} "
        f"dev_probes={len(dev_probes)} "
        f"holdout_probes={len(holdout_probes)} "
        f"targeted_windows={int(targeted_inputs.shape[0])} "
        f"replay_windows={int(replay_inputs.shape[0])} "
        f"parent_dev_exact={parent_dev['exact_generation']}/{parent_dev['tasks']} "
        f"parent_dev_first_top1={parent_dev['first_top1_rate']:.6f} "
        f"parent_p3_loss={parent_p3['mean_loss']:.6f}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        correction.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_key = (-1, -1.0, -1.0)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs: list[dict[str, Any]] = []

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.max_epochs + 1):
        train_result = _train_epoch(
            model=model,
            stacked=stacked,
            optimizer=optimizer,
            targeted_inputs=targeted_inputs,
            targeted_labels=targeted_labels,
            replay_inputs=replay_inputs,
            replay_labels=replay_labels,
            device=args.device,
            epoch=epoch,
        )
        merged = _merge_adapter(parent, correction).to(args.device)
        candidate_model = VN97OutputExpandedModel(model, merged)
        dev = _generation_eval(
            model=candidate_model,
            tokenizer=tokenizer,
            probes=dev_probes,
            device=args.device,
            label=f"epoch-{epoch}-dev",
        )
        key = (
            int(dev["exact_generation"]),
            float(dev["first_top1_rate"]),
            float(dev["mean_char_prefix_ratio"]),
        )
        epochs.append(
            {
                "epoch": epoch,
                "training": train_result,
                "dev": dev,
            }
        )

        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu()
                for name, tensor in correction.state_dict().items()
            }
            _write_torch_atomic(
                work / "best-correction.pt",
                {
                    "schema": "VN97P5E4LBEST1",
                    "epoch": best_epoch,
                    "key": best_key,
                    "correction_state_dict": best_state,
                },
            )

        print(
            "VN97 P5E4L EPOCH "
            f"epoch={epoch}/{args.max_epochs} "
            f"dev_exact={dev['exact_generation']}/{dev['tasks']} "
            f"dev_first_top1={dev['first_top1_rate']:.6f} "
            f"dev_prefix={dev['mean_char_prefix_ratio']:.6f} "
            f"best_epoch={best_epoch}",
            flush=True,
        )

        if (
            float(dev["exact_generation_rate"]) >= MIN_QAT_EXACT_RATE
            and int(dev["exact_generation"])
            >= int(parent_dev["exact_generation"]) + MIN_EXACT_GAIN
        ):
            print(
                "VN97 P5E4L EARLY_STOP "
                f"epoch={epoch} reason=dev_exact_gate_reached",
                flush=True,
            )
            break

    if best_state is None:
        raise VN97P5E4LError("no rescue checkpoint was selected")

    correction.load_state_dict(best_state)
    merged = _merge_adapter(parent, correction).to(args.device)
    candidate_model = VN97OutputExpandedModel(model, merged)

    parent_holdout = _generation_eval(
        model=parent_model,
        tokenizer=tokenizer,
        probes=holdout_probes,
        device=args.device,
        label="parent-holdout",
    )
    candidate_holdout = _generation_eval(
        model=candidate_model,
        tokenizer=tokenizer,
        probes=holdout_probes,
        device=args.device,
        label="candidate-holdout",
    )
    candidate_p3 = _p3_eval(
        model=model,
        adapter=merged,
        inputs=p3_inputs,
        labels=p3_labels,
        device=args.device,
    )

    core_after = _state_hash(model, exclude_final_norm=False)
    if core_after != core_before:
        raise VN97P5E4LError("frozen VN97 core changed during sequence rescue")

    reasons: list[str] = []
    exact_gain = (
        int(candidate_holdout["exact_generation"])
        - int(parent_holdout["exact_generation"])
    )
    if float(candidate_holdout["exact_generation_rate"]) < MIN_QAT_EXACT_RATE:
        reasons.append("fresh_exact_rate_below_10pct")
    if exact_gain < MIN_EXACT_GAIN:
        reasons.append("fresh_exact_gain_below_8")
    if float(candidate_p3["mean_loss"]) > float(parent_p3["mean_loss"]) * 1.03:
        reasons.append("p3_loss_over_3pct")
    if (
        float(candidate_p3["top1_accuracy"]) + 0.01
        < float(parent_p3["top1_accuracy"])
    ):
        reasons.append("p3_top1_drop_over_1pp")

    status = (
        "SEQUENCE_RESCUE_SIGNAL"
        if not reasons
        else "SEQUENCE_RESCUE_INSUFFICIENT"
    )
    ready_for_qat = not reasons

    output.mkdir(parents=True, exist_ok=True)
    tokenizer_bytes = (
        Path(args.repaired_dir) / "tokenizer.vn97tk1"
    ).read_bytes()
    tokenizer_sha = _sha256_bytes(tokenizer_bytes)
    if tokenizer_sha != repaired_meta["tokenizer_sha256"]:
        raise VN97P5E4LError("repaired tokenizer hash changed")

    report = {
        "adapter": {
            "parent_rank": PARENT_RANK,
            "correction_rank": CORRECTION_RANK,
            "merged_rank": MERGED_RANK,
            "parent_adapter_sha256": prefix_meta["adapter_sha256"],
            "best_epoch": best_epoch,
        },
        "curriculum": {
            "train_shards": TRAIN_SHARDS,
            "train_per_category_per_shard": TRAIN_PER_CATEGORY_PER_SHARD,
            "train_probes": len(train_probes),
            "dev_probes": len(dev_probes),
            "holdout_probes": len(holdout_probes),
            "p3_replay_records": P3_REPLAY_RECORDS,
        },
        "epochs": epochs,
        "evaluations": {
            "parent_dev": parent_dev,
            "parent_holdout": parent_holdout,
            "candidate_holdout": candidate_holdout,
            "parent_p3": parent_p3,
            "candidate_p3": candidate_p3,
        },
        "frozen_core_sha256": core_after,
        "p3_training_sha256": p3_training_sha,
        "p3_validation_sha256": p3_validation_sha,
        "p5e4k_report_sha256": p5e4k_sha,
        "p5e4k_status": p5e4k_report["status"],
        "profile_id": P5E4L_PROFILE_ID,
        "ready_for_qat": ready_for_qat,
        "reasons": reasons,
        "repaired_student_sha256": repaired_meta["student_sha256"],
        "schema": P5E4L_SCHEMA,
        "status": status,
        "tokenizer_sha256": tokenizer_sha,
    }

    _atomic_write(
        output / "p5e4l-report.json",
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8") + b"\n",
    )

    payload = {
        "schema": P5E4L_ADAPTER_SCHEMA,
        "profile_id": P5E4L_PROFILE_ID,
        "status": status,
        "rank": MERGED_RANK,
        "d_model": model.config.d_model,
        "vocab_size": model.config.vocab_size,
        "repaired_student_sha256": repaired_meta["student_sha256"],
        "parent_adapter_sha256": prefix_meta["adapter_sha256"],
        "frozen_core_sha256": core_after,
        "p5e4k_report_sha256": p5e4k_sha,
        "tokenizer_sha256": tokenizer_sha,
        "ready_for_qat": ready_for_qat,
        "adapter_state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in merged.state_dict().items()
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
        "output-adapter.pt": _sha256_file(output / "output-adapter.pt"),
        "p5e4l-report.json": _sha256_file(output / "p5e4l-report.json"),
        "tokenizer.vn97tk1": _sha256_file(output / "tokenizer.vn97tk1"),
    }
    _atomic_write(
        output / "SHA256SUMS",
        "".join(
            f"{digest}  {name}\n"
            for name, digest in sorted(sums.items())
        ).encode("ascii"),
    )

    print(
        "VN97P5E4L "
        f"status={status} "
        f"best_epoch={best_epoch} "
        f"parent_holdout_exact="
        f"{parent_holdout['exact_generation']}/{parent_holdout['tasks']} "
        f"candidate_holdout_exact="
        f"{candidate_holdout['exact_generation']}/{candidate_holdout['tasks']} "
        f"candidate_holdout_rate="
        f"{candidate_holdout['exact_generation_rate']:.6f} "
        f"parent_first_top1={parent_holdout['first_top1_rate']:.6f} "
        f"candidate_first_top1={candidate_holdout['first_top1_rate']:.6f} "
        f"parent_p3_loss={parent_p3['mean_loss']:.6f} "
        f"candidate_p3_loss={candidate_p3['mean_loss']:.6f} "
        f"frozen_core_match={str(core_before == core_after).lower()} "
        f"ready_for_qat={str(ready_for_qat).lower()} "
        f"reasons={','.join(reasons) if reasons else 'none'} "
        f"output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"vn97-p5e4l-rescue: {exc}", file=sys.stderr)
        raise
