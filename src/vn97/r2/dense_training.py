from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Sequence

import torch
import torch.nn.functional as F

from ..training import IGNORE_INDEX, VN97TrainingWindow
from .checkpoint import save_r2_checkpoint, sha256_file
from .model import VN97R2Model


@dataclass(frozen=True)
class R2DenseTrainingConfig:
    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 9703
    checkpoint_every_steps: int = 100

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.checkpoint_every_steps <= 0:
            raise ValueError("checkpoint_every_steps must be positive")


@dataclass(frozen=True)
class R2DenseTrainingResult:
    steps: int
    target_tokens: int
    mean_loss: float
    final_loss: float
    best_epoch: int
    best_validation_loss: float
    best_checkpoint_sha256: str


def _device(value: str | torch.device) -> torch.device:
    if isinstance(value, torch.device):
        return value
    if value == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    return torch.device(value)


def _batch(
    windows: Sequence[VN97TrainingWindow],
    indices: Sequence[int],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [windows[index] for index in indices]
    return (
        torch.tensor(
            [row.input_ids for row in rows],
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            [row.labels for row in rows],
            dtype=torch.long,
            device=device,
        ),
    )


@torch.inference_mode()
def evaluate_dense_loss(
    model: VN97R2Model,
    windows: Sequence[VN97TrainingWindow],
    *,
    batch_size: int,
    device: str | torch.device = "auto",
) -> dict[str, float | int]:
    if not windows:
        raise ValueError("validation windows must not be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    resolved = _device(device)
    model.to(resolved)
    model.eval()

    loss_sum = 0.0
    target_tokens = 0
    correct = 0

    for start in range(0, len(windows), batch_size):
        indices = list(
            range(start, min(start + batch_size, len(windows)))
        )
        inputs, labels = _batch(
            windows,
            indices,
            device=resolved,
        )
        logits, _ = model(inputs, profile="deep")
        mask = labels != IGNORE_INDEX
        count = int(mask.sum().item())
        if count == 0:
            continue
        selected_logits = logits[mask]
        selected_labels = labels[mask]
        loss_sum += float(
            F.cross_entropy(
                selected_logits.float(),
                selected_labels,
                reduction="sum",
            ).item()
        )
        target_tokens += count
        correct += int(
            (
                selected_logits.argmax(dim=-1)
                == selected_labels
            ).sum().item()
        )

    if target_tokens <= 0:
        raise RuntimeError("validation produced no target tokens")
    return {
        "target_tokens": target_tokens,
        "mean_loss": loss_sum / target_tokens,
        "top1_accuracy": correct / target_tokens,
    }


def _run_identity(
    model: VN97R2Model,
    config: R2DenseTrainingConfig,
    *,
    dataset_identity: str,
) -> str:
    if not dataset_identity:
        raise ValueError("dataset_identity must be non-empty")
    payload = json.dumps(
        {
            "architecture": model.config.fingerprint(),
            "training": asdict(config),
            "dataset_identity": dataset_identity,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(b"VN97R2DENSE\0" + payload).hexdigest()


def _validate_resume_best_checkpoint(
    best_path: Path,
    *,
    best_epoch: int,
    best_checkpoint_sha256: str,
) -> str:
    if best_epoch >= 0:
        if not best_path.is_file():
            raise RuntimeError(
                "R2 dense resume references a missing best checkpoint"
            )
        actual_best_sha256 = sha256_file(best_path)
        if (
            best_checkpoint_sha256
            and actual_best_sha256 != best_checkpoint_sha256
        ):
            raise RuntimeError(
                "R2 dense resume best checkpoint SHA-256 mismatch"
            )
        return actual_best_sha256
    if best_checkpoint_sha256:
        raise RuntimeError(
            "R2 dense resume has checkpoint identity without best epoch"
        )
    return ""


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _save_resume(
    path: Path,
    *,
    identity: str,
    epoch: int,
    next_batch: int,
    model: VN97R2Model,
    optimizer: torch.optim.Optimizer,
    steps: int,
    target_tokens: int,
    loss_sum: float,
    final_loss: float,
    best_epoch: int,
    best_validation_loss: float,
    best_checkpoint_sha256: str,
) -> None:
    temp = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "schema": "VN97R2DENSERESUME1",
            "identity": identity,
            "epoch": epoch,
            "next_batch": next_batch,
            "model_state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "steps": steps,
            "target_tokens": target_tokens,
            "loss_sum": loss_sum,
            "final_loss": final_loss,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "best_checkpoint_sha256": best_checkpoint_sha256,
        },
        temp,
    )
    temp.replace(path)


def train_dense(
    model: VN97R2Model,
    training_windows: Sequence[VN97TrainingWindow],
    validation_windows: Sequence[VN97TrainingWindow],
    config: R2DenseTrainingConfig,
    *,
    work_dir: str | Path,
    best_checkpoint_path: str | Path,
    dataset_identity: str,
    device: str | torch.device = "auto",
) -> R2DenseTrainingResult:
    if not training_windows:
        raise ValueError("training windows must not be empty")
    if not validation_windows:
        raise ValueError("validation windows must not be empty")

    width = len(training_windows[0].input_ids)
    if any(len(item.input_ids) != width for item in training_windows):
        raise ValueError("training windows must share a common width")

    resolved = _device(device)
    torch.manual_seed(config.seed)
    if resolved.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    model.to(resolved)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    identity = _run_identity(
        model,
        config,
        dataset_identity=dataset_identity,
    )

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    resume_path = work / "dense-resume.pt"
    best_path = Path(best_checkpoint_path)

    start_epoch = 0
    start_batch = 0
    steps = 0
    target_tokens = 0
    loss_sum = 0.0
    final_loss = float("nan")
    best_epoch = -1
    best_validation_loss = float("inf")
    best_checkpoint_sha256 = ""

    if resume_path.is_file():
        resume = torch.load(
            resume_path,
            map_location="cpu",
            weights_only=False,
        )
        if (
            not isinstance(resume, dict)
            or resume.get("schema") != "VN97R2DENSERESUME1"
            or resume.get("identity") != identity
        ):
            raise RuntimeError("R2 dense resume identity mismatch")
        model.load_state_dict(resume["model_state_dict"])
        model.to(resolved)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        _move_optimizer_state(optimizer, resolved)
        start_epoch = int(resume["epoch"])
        start_batch = int(resume["next_batch"])
        steps = int(resume["steps"])
        target_tokens = int(resume["target_tokens"])
        loss_sum = float(resume["loss_sum"])
        final_loss = float(resume["final_loss"])
        best_epoch = int(resume["best_epoch"])
        best_validation_loss = float(
            resume["best_validation_loss"]
        )
        best_checkpoint_sha256 = str(
            resume.get("best_checkpoint_sha256", "")
        )
        best_checkpoint_sha256 = _validate_resume_best_checkpoint(
            best_path,
            best_epoch=best_epoch,
            best_checkpoint_sha256=best_checkpoint_sha256,
        )

    for epoch in range(start_epoch, config.epochs):
        order = list(range(len(training_windows)))
        if len(order) > 1:
            random.Random(config.seed + epoch).shuffle(order)
        batches = [
            order[start : start + config.batch_size]
            for start in range(0, len(order), config.batch_size)
        ]
        batch_begin = start_batch if epoch == start_epoch else 0
        if not 0 <= batch_begin <= len(batches):
            raise RuntimeError("R2 dense resume batch is out of range")

        model.train()
        for batch_index in range(batch_begin, len(batches)):
            inputs, labels = _batch(
                training_windows,
                batches[batch_index],
                device=resolved,
            )
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(inputs, profile="deep")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("R2 dense loss became non-finite")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.max_grad_norm,
            )
            if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
                raise RuntimeError(
                    "R2 dense gradient norm became non-finite"
                )
            optimizer.step()

            final_loss = float(loss.detach().cpu())
            loss_sum += final_loss
            target_tokens += int(
                (labels != IGNORE_INDEX).sum().item()
            )
            steps += 1

            if steps % config.checkpoint_every_steps == 0:
                _save_resume(
                    resume_path,
                    identity=identity,
                    epoch=epoch,
                    next_batch=batch_index + 1,
                    model=model,
                    optimizer=optimizer,
                    steps=steps,
                    target_tokens=target_tokens,
                    loss_sum=loss_sum,
                    final_loss=final_loss,
                    best_epoch=best_epoch,
                    best_validation_loss=best_validation_loss,
                    best_checkpoint_sha256=best_checkpoint_sha256,
                )

        validation = evaluate_dense_loss(
            model,
            validation_windows,
            batch_size=config.batch_size,
            device=resolved,
        )
        validation_loss = float(validation["mean_loss"])
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_checkpoint_sha256 = save_r2_checkpoint(
                best_path,
                model,
                stage="dense_pretrain",
                metadata={
                    "dataset_identity": dataset_identity,
                    "epoch": epoch,
                    "validation": validation,
                },
            )

        if epoch + 1 < config.epochs:
            _save_resume(
                resume_path,
                identity=identity,
                epoch=epoch + 1,
                next_batch=0,
                model=model,
                optimizer=optimizer,
                steps=steps,
                target_tokens=target_tokens,
                loss_sum=loss_sum,
                final_loss=final_loss,
                best_epoch=best_epoch,
                best_validation_loss=best_validation_loss,
                best_checkpoint_sha256=best_checkpoint_sha256,
            )
        start_batch = 0

    resume_path.unlink(missing_ok=True)

    if steps <= 0 or target_tokens <= 0:
        raise RuntimeError("R2 dense training completed without work")
    if best_epoch < 0 or not best_checkpoint_sha256:
        raise RuntimeError("R2 dense training selected no checkpoint")

    return R2DenseTrainingResult(
        steps=steps,
        target_tokens=target_tokens,
        mean_loss=loss_sum / steps,
        final_loss=final_loss,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        best_checkpoint_sha256=best_checkpoint_sha256,
    )
