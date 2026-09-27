from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

from .p4_generalization_cli import _select_replay, _windows_to_tensors
from .p5e4b_fresh_validation_cli import _build_probes
from .p5e4c_decoder_repair_cli import SEQUENCE_LENGTH, _sha256_file
from .p5e4d_fresh_decoder_validation_cli import _load_repaired
from .p5e4e_output_expansion_cli import ResidualOutputAdapter
from .p5e4f_fresh_output_validation_cli import VN97OutputExpandedModel
from .p5e4l_sequence_rescue_cli import (
    P5E4L_ADAPTER_SCHEMA,
    P5E4L_SCHEMA,
    _generation_eval,
    _p3_eval,
)
from .quantization import set_float_shadow_mode
from .training import (
    VN97ChatMessage,
    VN97TrainingConfig,
    build_training_windows,
    encode_chat_completion_messages,
)
from .training_cli import _atomic_write, _load_records


P5E5A_SCHEMA = "VN97P5E5A"
P5E5A_PACKAGE_SCHEMA = "VN97P5E5ALATECORE1"
P5E5A_PROFILE_ID = "vn97-p5e5a-late-core-sequence-repair-v1"

TRAINABLE_LAYERS = 4
TRAIN_START_LAYER = 28
ADAPTER_RANK = 96

TRAIN_SHARDS = 4
TRAIN_PER_CATEGORY_PER_SHARD = 24
DEV_PER_CATEGORY = 12
HOLDOUT_PER_CATEGORY = 24
P3_REPLAY_RECORDS = 128

MAX_EPOCHS = 2
BATCH_SIZE = 1
CORE_LR = 1.5e-5
HEAD_LR = 7e-5
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
EARLY_8_WEIGHT = 2.0
EARLY_32_WEIGHT = 1.25
PROGRESS_INTERVAL = 50
CHECKPOINT_INTERVAL = 100
SEED = 9759

MIN_QAT_EXACT_RATE = 0.10
MIN_EXACT_GAIN = 8


class VN97P5E5AError(RuntimeError):
    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "P5E5A late-core sequence repair. P5E4L established an output-only "
            "ceiling at zero exact generations, so this stage unfreezes only "
            "the last four VN97 SSM blocks plus final RMSNorm and the rank-96 "
            "output adapter. The first 28 layers, embedding, and tied lexical "
            "head remain immutable. One run includes train/dev selection and "
            "a final untouched holdout/QAT gate."
        )
    )
    parser.add_argument("--repaired-dir", required=True)
    parser.add_argument("--p5e4l-dir", required=True)
    parser.add_argument("--p3-corpus-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    return parser


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_p5e4l_seed(
    root: Path,
    *,
    repaired_meta: dict[str, Any],
    d_model: int,
    vocab_size: int,
) -> tuple[ResidualOutputAdapter, dict[str, Any]]:
    resolved = root.resolve(strict=True)
    report_path = resolved / "p5e4l-report.json"
    adapter_path = resolved / "output-adapter.pt"
    tokenizer_path = resolved / "tokenizer.vn97tk1"
    for path in (report_path, adapter_path, tokenizer_path):
        if not path.is_file():
            raise VN97P5E5AError(f"P5E4L seed missing {path.name}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("schema") != P5E4L_SCHEMA:
        raise VN97P5E5AError("invalid P5E4L report schema")
    if report.get("status") not in {
        "SEQUENCE_RESCUE_SIGNAL",
        "SEQUENCE_RESCUE_INSUFFICIENT",
    }:
        raise VN97P5E5AError("P5E4L seed status is not usable")
    if report.get("repaired_student_sha256") != repaired_meta.get(
        "student_sha256"
    ):
        raise VN97P5E5AError("P5E4L/P5E4C checkpoint identity mismatch")

    payload = torch.load(
        adapter_path,
        map_location="cpu",
        weights_only=False,
    )
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != P5E4L_ADAPTER_SCHEMA
        or int(payload.get("rank", -1)) != ADAPTER_RANK
        or int(payload.get("d_model", -1)) != d_model
        or int(payload.get("vocab_size", -1)) != vocab_size
    ):
        raise VN97P5E5AError("P5E4L adapter contract mismatch")
    if payload.get("repaired_student_sha256") != repaired_meta.get(
        "student_sha256"
    ):
        raise VN97P5E5AError("P5E4L adapter checkpoint identity mismatch")
    if payload.get("tokenizer_sha256") != repaired_meta.get(
        "tokenizer_sha256"
    ):
        raise VN97P5E5AError("P5E4L adapter tokenizer identity mismatch")

    tokenizer_sha = _sha256(tokenizer_path)
    if tokenizer_sha != repaired_meta.get("tokenizer_sha256"):
        raise VN97P5E5AError("P5E4L copied tokenizer hash mismatch")

    state = payload.get("adapter_state_dict")
    if not isinstance(state, dict):
        raise VN97P5E5AError("P5E4L adapter state missing")

    adapter = ResidualOutputAdapter(
        d_model=d_model,
        vocab_size=vocab_size,
        rank=ADAPTER_RANK,
    )
    adapter.load_state_dict(state)

    return adapter, {
        "root": str(resolved),
        "status": report.get("status"),
        "report_sha256": _sha256(report_path),
        "adapter_sha256": _sha256(adapter_path),
        "tokenizer_sha256": tokenizer_sha,
        "best_epoch": report.get("adapter", {}).get("best_epoch"),
        "ready_for_qat": bool(report.get("ready_for_qat", False)),
    }


def _seed(root: str, label: str, index: int = 0) -> str:
    return hashlib.sha256(
        (
            P5E5A_PROFILE_ID
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
        for probe in _build_probes(
            seed_hex=_seed(root_seed, "train", shard),
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


def _messages(probe):
    return (
        VN97ChatMessage(role="user", content=probe.prompt),
        VN97ChatMessage(role="assistant", content=probe.target),
    )


def _frozen_prefix_hash(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        trainable = (
            name.startswith("layers.28.")
            or name.startswith("layers.29.")
            or name.startswith("layers.30.")
            or name.startswith("layers.31.")
            or name.startswith("final_norm.")
        )
        if trainable:
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _trainable_state(model, adapter) -> dict[str, Any]:
    core: dict[str, torch.Tensor] = {}
    for index in range(TRAIN_START_LAYER, model.config.n_layers):
        for name, tensor in model.layers[index].state_dict().items():
            core[f"layers.{index}.{name}"] = tensor.detach().cpu()
    for name, tensor in model.final_norm.state_dict().items():
        core[f"final_norm.{name}"] = tensor.detach().cpu()
    return {
        "core_state_dict": core,
        "adapter_state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in adapter.state_dict().items()
        },
    }


def _load_trainable_state(model, adapter, state: dict[str, Any]) -> None:
    core = state["core_state_dict"]
    for index in range(TRAIN_START_LAYER, model.config.n_layers):
        prefix = f"layers.{index}."
        local = {
            name[len(prefix):]: tensor
            for name, tensor in core.items()
            if name.startswith(prefix)
        }
        model.layers[index].load_state_dict(local)
    norm_prefix = "final_norm."
    norm_state = {
        name[len(norm_prefix):]: tensor
        for name, tensor in core.items()
        if name.startswith(norm_prefix)
    }
    model.final_norm.load_state_dict(norm_state)
    adapter.load_state_dict(state["adapter_state_dict"])


def _configure_trainable(model, adapter) -> tuple[list, list]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for index in range(TRAIN_START_LAYER, model.config.n_layers):
        for parameter in model.layers[index].parameters():
            parameter.requires_grad_(True)
    for parameter in model.final_norm.parameters():
        parameter.requires_grad_(True)
    for parameter in adapter.parameters():
        parameter.requires_grad_(True)

    core_parameters = [
        parameter
        for index in range(TRAIN_START_LAYER, model.config.n_layers)
        for parameter in model.layers[index].parameters()
    ]
    core_parameters.extend(model.final_norm.parameters())
    head_parameters = list(adapter.parameters())
    return core_parameters, head_parameters


def _forward_trainable(model, adapter, input_ids: torch.Tensor):
    with torch.no_grad():
        x = model.embedding(input_ids)
        for index in range(TRAIN_START_LAYER):
            x, _ = model.layers[index](x, None)

    for index in range(TRAIN_START_LAYER, model.config.n_layers):
        layer = model.layers[index]
        normalized = layer.norm(x)
        y, _ = layer.core.forward_sequential_reference(
            normalized,
            None,
        )
        x = x + y

    hidden = model.final_norm(x)
    logits = model.lm_head(hidden) + adapter(hidden)
    return logits


def _weighted_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    mask = labels != -100
    if not bool(mask.any()):
        raise VN97P5E5AError("batch has no supervised tokens")

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


def _make_schedule(
    targeted_count: int,
    replay_count: int,
    epoch: int,
) -> list[tuple[str, int]]:
    rng = random.Random(SEED + epoch)
    schedule = [("targeted", i) for i in range(targeted_count)]
    schedule.extend(("replay", i) for i in range(replay_count))
    rng.shuffle(schedule)
    return schedule


def _optimizer(model, adapter):
    core_parameters, head_parameters = _configure_trainable(model, adapter)
    return torch.optim.AdamW(
        [
            {
                "params": core_parameters,
                "lr": CORE_LR,
                "weight_decay": WEIGHT_DECAY,
            },
            {
                "params": head_parameters,
                "lr": HEAD_LR,
                "weight_decay": WEIGHT_DECAY,
            },
        ]
    )


def _move_optimizer_state(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _train_epoch(
    *,
    model,
    adapter,
    optimizer,
    targeted_inputs,
    targeted_labels,
    replay_inputs,
    replay_labels,
    device: str,
    epoch: int,
    work_dir: Path,
    identity: str,
) -> dict[str, Any]:
    resolved = torch.device(device)
    schedule = _make_schedule(
        int(targeted_inputs.shape[0]),
        int(replay_inputs.shape[0]),
        epoch,
    )

    resume_path = work_dir / "state.p5e5a.pt"
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
        if (
            isinstance(resume, dict)
            and resume.get("identity") == identity
            and int(resume.get("epoch", -1)) == epoch
        ):
            _load_trainable_state(model, adapter, resume["trainable_state"])
            optimizer.load_state_dict(resume["optimizer_state_dict"])
            _move_optimizer_state(optimizer, resolved)
            start_step = int(resume.get("next_step", 0))
            loss_sum = float(resume.get("loss_sum", 0.0))
            targeted_steps = int(resume.get("targeted_steps", 0))
            replay_steps = int(resume.get("replay_steps", 0))
            print(
                "VN97 P5E5A RESUME "
                f"epoch={epoch} step={start_step}/{len(schedule)}",
                flush=True,
            )

    model.train()
    adapter.train()
    started = time.monotonic()

    for step_index in range(start_step, len(schedule)):
        kind, index = schedule[step_index]
        if kind == "targeted":
            x = targeted_inputs[index : index + 1]
            y = targeted_labels[index : index + 1]
        else:
            x = replay_inputs[index : index + 1]
            y = replay_labels[index : index + 1]

        x = x.to(device=resolved, dtype=torch.long)
        y = y.to(device=resolved, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        logits = _forward_trainable(model, adapter, x)
        if kind == "targeted":
            loss = _weighted_ce(logits, y)
            targeted_steps += 1
        else:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                y.reshape(-1),
                ignore_index=-100,
            )
            replay_steps += 1

        if not bool(torch.isfinite(loss)):
            raise VN97P5E5AError("late-core repair loss became non-finite")
        loss.backward()
        trainable = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable,
            MAX_GRAD_NORM,
        )
        if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
            raise VN97P5E5AError("late-core repair gradient became non-finite")
        optimizer.step()

        completed = step_index + 1
        loss_sum += float(loss.detach().cpu())

        if completed % PROGRESS_INTERVAL == 0 or completed == len(schedule):
            elapsed = time.monotonic() - started
            local_steps = max(1, completed - start_step)
            eta = (
                (len(schedule) - completed)
                * elapsed
                / local_steps
            )
            print(
                "VN97 P5E5A TRAIN "
                f"epoch={epoch} step={completed}/{len(schedule)} "
                f"percent={100.0 * completed / len(schedule):.2f} "
                f"kind={kind} loss={float(loss.detach().cpu()):.6f} "
                f"mean_loss={loss_sum / completed:.6f} "
                f"grad_norm={float(grad_norm):.6f} "
                f"elapsed_s={elapsed:.1f} eta_s={eta:.1f}",
                flush=True,
            )

        if (
            completed % CHECKPOINT_INTERVAL == 0
            and completed < len(schedule)
        ):
            work_dir.mkdir(parents=True, exist_ok=True)
            temp = resume_path.with_suffix(".tmp")
            torch.save(
                {
                    "schema": "VN97P5E5ARESUME1",
                    "identity": identity,
                    "epoch": epoch,
                    "next_step": completed,
                    "trainable_state": _trainable_state(model, adapter),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss_sum": loss_sum,
                    "targeted_steps": targeted_steps,
                    "replay_steps": replay_steps,
                },
                temp,
            )
            temp.replace(resume_path)

    resume_path.unlink(missing_ok=True)
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
        raise VN97P5E5AError("P5E5A requires CUDA")
    if args.max_epochs < 1 or args.max_epochs > MAX_EPOCHS:
        raise VN97P5E5AError(
            f"max-epochs must be in [1, {MAX_EPOCHS}]"
        )

    output = Path(args.output_dir)
    if output.exists() and (
        output.is_symlink()
        or not output.is_dir()
        or any(output.iterdir())
    ):
        raise VN97P5E5AError("output-dir must be new or empty")

    model, tokenizer, repaired_meta = _load_repaired(
        Path(args.repaired_dir)
    )
    if (
        model.config.n_layers != 32
        or model.config.d_model != 1536
        or model.config.d_state != 16
    ):
        raise VN97P5E5AError("P5E5A requires canonical 309M VN97")
    set_float_shadow_mode(model, True)

    adapter, p5e4l_meta = _load_p5e4l_seed(
        Path(args.p5e4l_dir),
        repaired_meta=repaired_meta,
        d_model=model.config.d_model,
        vocab_size=model.config.vocab_size,
    )

    model.to(args.device)
    adapter.to(args.device)
    frozen_before = _frozen_prefix_hash(model)

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
            + str(p5e4l_meta["adapter_sha256"])
            + "\0"
            + P5E5A_PROFILE_ID
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

    config = VN97TrainingConfig(
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        epochs=1,
        learning_rate=HEAD_LR,
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
        encode_chat_completion_messages(tokenizer, _messages(probe))
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
            config,
            pad_token_id=tokenizer.pad_id,
        )
    )
    replay_inputs, replay_labels = _windows_to_tensors(
        build_training_windows(
            replay_examples,
            config,
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

    seed_model = VN97OutputExpandedModel(model, adapter)
    seed_dev = _generation_eval(
        model=seed_model,
        tokenizer=tokenizer,
        probes=dev_probes,
        device=args.device,
        label="seed-dev",
    )
    seed_p3 = _p3_eval(
        model=model,
        adapter=adapter,
        inputs=p3_inputs,
        labels=p3_labels,
        device=args.device,
    )

    print(
        "VN97 P5E5A START "
        f"p5e4l_status={p5e4l_meta['status']} "
        f"trainable_layers={TRAINABLE_LAYERS} "
        f"train_start_layer={TRAIN_START_LAYER} "
        f"train_probes={len(train_probes)} "
        f"dev_probes={len(dev_probes)} "
        f"holdout_probes={len(holdout_probes)} "
        f"replay_records={len(replay_records)} "
        f"seed_dev_exact={seed_dev['exact_generation']}/{seed_dev['tasks']} "
        f"seed_dev_first_top1={seed_dev['first_top1_rate']:.6f} "
        f"seed_p3_loss={seed_p3['mean_loss']:.6f}",
        flush=True,
    )

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(
        json.dumps(
            {
                "profile": P5E5A_PROFILE_ID,
                "student": repaired_meta["student_sha256"],
                "adapter": p5e4l_meta["adapter_sha256"],
                "p3_training": p3_training_sha,
                "p3_validation": p3_validation_sha,
                "root_seed": root_seed,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    optimizer = _optimizer(model, adapter)
    best_key = (-1, -1.0, -1.0)
    best_epoch = 0
    best_state = _trainable_state(model, adapter)
    epochs: list[dict[str, Any]] = []

    for epoch in range(1, args.max_epochs + 1):
        train_result = _train_epoch(
            model=model,
            adapter=adapter,
            optimizer=optimizer,
            targeted_inputs=targeted_inputs,
            targeted_labels=targeted_labels,
            replay_inputs=replay_inputs,
            replay_labels=replay_labels,
            device=args.device,
            epoch=epoch,
            work_dir=work,
            identity=identity,
        )

        candidate = VN97OutputExpandedModel(model, adapter)
        dev = _generation_eval(
            model=candidate,
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
            best_state = _trainable_state(model, adapter)
            _write_torch_atomic(
                work / "best-late-core.pt",
                {
                    "schema": "VN97P5E5ABEST1",
                    "epoch": best_epoch,
                    "key": best_key,
                    "trainable_state": best_state,
                },
            )

        print(
            "VN97 P5E5A EPOCH "
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
            >= int(seed_dev["exact_generation"]) + MIN_EXACT_GAIN
        ):
            print(
                "VN97 P5E5A EARLY_STOP "
                f"epoch={epoch} reason=dev_exact_gate_reached",
                flush=True,
            )
            break

    _load_trainable_state(model, adapter, best_state)
    candidate = VN97OutputExpandedModel(model, adapter)

    seed_state = torch.load(
        Path(args.p5e4l_dir) / "output-adapter.pt",
        map_location="cpu",
        weights_only=False,
    )["adapter_state_dict"]
    seed_adapter = ResidualOutputAdapter(
        model.config.d_model,
        model.config.vocab_size,
        ADAPTER_RANK,
    ).to(args.device)
    seed_adapter.load_state_dict(seed_state)

    original_model, _, _ = _load_repaired(Path(args.repaired_dir))
    set_float_shadow_mode(original_model, True)
    original_model.to(args.device)
    seed_holdout_model = VN97OutputExpandedModel(
        original_model,
        seed_adapter,
    )

    seed_holdout = _generation_eval(
        model=seed_holdout_model,
        tokenizer=tokenizer,
        probes=holdout_probes,
        device=args.device,
        label="seed-holdout",
    )
    candidate_holdout = _generation_eval(
        model=candidate,
        tokenizer=tokenizer,
        probes=holdout_probes,
        device=args.device,
        label="candidate-holdout",
    )
    candidate_p3 = _p3_eval(
        model=model,
        adapter=adapter,
        inputs=p3_inputs,
        labels=p3_labels,
        device=args.device,
    )

    frozen_after = _frozen_prefix_hash(model)
    if frozen_after != frozen_before:
        raise VN97P5E5AError("frozen first-28-layer prefix changed")

    reasons: list[str] = []
    exact_gain = (
        int(candidate_holdout["exact_generation"])
        - int(seed_holdout["exact_generation"])
    )
    if float(candidate_holdout["exact_generation_rate"]) < MIN_QAT_EXACT_RATE:
        reasons.append("fresh_exact_rate_below_10pct")
    if exact_gain < MIN_EXACT_GAIN:
        reasons.append("fresh_exact_gain_below_8")
    if int(candidate_holdout["generation_errors"]) > int(
        seed_holdout["generation_errors"]
    ):
        reasons.append("fresh_generation_errors_increased")
    if float(candidate_p3["mean_loss"]) > float(seed_p3["mean_loss"]) * 1.03:
        reasons.append("p3_loss_over_3pct")
    if (
        float(candidate_p3["top1_accuracy"]) + 0.01
        < float(seed_p3["top1_accuracy"])
    ):
        reasons.append("p3_top1_drop_over_1pp")

    ready_for_qat = not reasons
    status = (
        "LATE_CORE_RESCUE_SIGNAL"
        if ready_for_qat
        else "LATE_CORE_RESCUE_INSUFFICIENT"
    )

    output.mkdir(parents=True, exist_ok=True)
    tokenizer_bytes = (
        Path(args.repaired_dir) / "tokenizer.vn97tk1"
    ).read_bytes()
    tokenizer_sha = _sha256_bytes(tokenizer_bytes)

    report = {
        "architecture": {
            "total_layers": model.config.n_layers,
            "trainable_layers": TRAINABLE_LAYERS,
            "train_start_layer": TRAIN_START_LAYER,
            "adapter_rank": ADAPTER_RANK,
            "embedding_frozen": True,
            "tied_lexical_head_frozen": True,
            "first_28_layers_frozen": True,
            "final_norm_trainable": True,
        },
        "curriculum": {
            "train_probes": len(train_probes),
            "dev_probes": len(dev_probes),
            "holdout_probes": len(holdout_probes),
            "p3_replay_records": P3_REPLAY_RECORDS,
            "max_epochs": args.max_epochs,
        },
        "epochs": epochs,
        "evaluations": {
            "seed_dev": seed_dev,
            "seed_holdout": seed_holdout,
            "candidate_holdout": candidate_holdout,
            "seed_p3": seed_p3,
            "candidate_p3": candidate_p3,
        },
        "frozen_prefix_sha256": frozen_after,
        "p3_training_sha256": p3_training_sha,
        "p3_validation_sha256": p3_validation_sha,
        "p5e4l_seed": p5e4l_meta,
        "profile_id": P5E5A_PROFILE_ID,
        "ready_for_qat": ready_for_qat,
        "reasons": reasons,
        "schema": P5E5A_SCHEMA,
        "selected_epoch": best_epoch,
        "status": status,
        "tokenizer_sha256": tokenizer_sha,
    }
    _atomic_write(
        output / "p5e5a-report.json",
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8") + b"\n",
    )

    package = {
        "schema": P5E5A_PACKAGE_SCHEMA,
        "profile_id": P5E5A_PROFILE_ID,
        "status": status,
        "ready_for_qat": ready_for_qat,
        "repaired_student_sha256": repaired_meta["student_sha256"],
        "seed_adapter_sha256": p5e4l_meta["adapter_sha256"],
        "tokenizer_sha256": tokenizer_sha,
        "frozen_prefix_sha256": frozen_after,
        "selected_epoch": best_epoch,
        **_trainable_state(model, adapter),
    }
    _write_torch_atomic(
        output / "late-core-rescue.pt",
        package,
    )
    _atomic_write(
        output / "tokenizer.vn97tk1",
        tokenizer_bytes,
    )
    sums = {
        "late-core-rescue.pt": _sha256_file(
            output / "late-core-rescue.pt"
        ),
        "p5e5a-report.json": _sha256_file(
            output / "p5e5a-report.json"
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
        "VN97P5E5A "
        f"status={status} "
        f"selected_epoch={best_epoch} "
        f"seed_holdout_exact="
        f"{seed_holdout['exact_generation']}/{seed_holdout['tasks']} "
        f"candidate_holdout_exact="
        f"{candidate_holdout['exact_generation']}/{candidate_holdout['tasks']} "
        f"candidate_holdout_rate="
        f"{candidate_holdout['exact_generation_rate']:.6f} "
        f"seed_first_top1={seed_holdout['first_top1_rate']:.6f} "
        f"candidate_first_top1={candidate_holdout['first_top1_rate']:.6f} "
        f"seed_p3_loss={seed_p3['mean_loss']:.6f} "
        f"candidate_p3_loss={candidate_p3['mean_loss']:.6f} "
        f"frozen_prefix_match={str(frozen_before == frozen_after).lower()} "
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
        print(f"vn97-p5e5a-repair: {exc}", file=sys.stderr)
        raise
