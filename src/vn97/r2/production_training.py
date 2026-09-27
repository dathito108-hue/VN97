from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from ..training import IGNORE_INDEX, VN97TrainingWindow
from .checkpoint import (
    load_r2_checkpoint,
    save_r2_checkpoint,
    sha256_file,
)
from .model import VN97R2Model
from .pilot_contract import available_memory_bytes
from .production_contract import (
    R2ProductionCorpusManifest,
    R2ProductionTrainingRecipe,
    assert_production_training_contract,
    assert_r2_production_scale,
    assert_stage_transition,
)


PRODUCTION_RESUME_SCHEMA = "VN97R2PRODRESUME1"


@dataclass(frozen=True)
class R2ProductionTrainerConfig:
    epochs: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    seed: int = 9710
    checkpoint_every_optimizer_steps: int = 25
    initial_loss_scale: float = 4096.0
    min_loss_scale: float = 1.0
    max_loss_scale: float = 1_048_576.0
    loss_scale_growth_interval: int = 200
    require_measured_cuda_preflight: bool = True

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if not 0.0 <= self.beta1 < 1.0:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0.0 <= self.beta2 < 1.0:
            raise ValueError("beta2 must be in [0, 1)")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.checkpoint_every_optimizer_steps <= 0:
            raise ValueError(
                "checkpoint_every_optimizer_steps must be positive"
            )
        if self.initial_loss_scale <= 0.0:
            raise ValueError("initial_loss_scale must be positive")
        if self.min_loss_scale <= 0.0:
            raise ValueError("min_loss_scale must be positive")
        if self.max_loss_scale < self.min_loss_scale:
            raise ValueError("max_loss_scale must be >= min_loss_scale")
        if not (
            self.min_loss_scale
            <= self.initial_loss_scale
            <= self.max_loss_scale
        ):
            raise ValueError(
                "initial_loss_scale must be within configured bounds"
            )
        if self.loss_scale_growth_interval <= 0:
            raise ValueError(
                "loss_scale_growth_interval must be positive"
            )


@dataclass(frozen=True)
class R2MeasuredMemoryEvidence:
    architecture_fingerprint: str
    recipe_fingerprint: str
    sequence_length: int
    micro_batch_size: int
    device_name: str
    free_device_bytes_before: int
    total_device_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    safety_fraction: float
    passed: bool
    reason: str
    schema: str = "VN97R2MEMPROBE1"

    def __post_init__(self) -> None:
        if self.sequence_length <= 0 or self.micro_batch_size <= 0:
            raise ValueError("measured memory dimensions must be positive")
        if not 0.0 < self.safety_fraction <= 1.0:
            raise ValueError("safety_fraction must be in (0, 1]")
        for name in (
            "free_device_bytes_before",
            "total_device_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class R2ProductionTrainingResult:
    optimizer_steps: int
    micro_steps: int
    target_tokens: int
    mean_loss: float
    final_loss: float
    best_epoch: int
    best_validation_loss: float
    best_checkpoint_sha256: str
    completed: bool
    resume_checkpoint_sha256: str
    loss_scale: float
    skipped_nonfinite_steps: int


class CPUOffloadedAdamW:
    """Full-parameter AdamW with persistent optimizer moments on CPU.

    Model parameters stay FP32 on the execution device. CUDA autocast controls
    compute precision, while exp_avg/exp_avg_sq remain FP32 CPU tensors.
    Parameter values are copied to CPU one tensor at a time for the update and
    copied back, avoiding persistent Adam state in device memory.
    """

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        learning_rate: float,
        weight_decay: float,
        beta1: float,
        beta2: float,
        eps: float,
    ) -> None:
        self.parameters = tuple(
            (name, parameter)
            for name, parameter in named_parameters
            if parameter.requires_grad
        )
        if not self.parameters:
            raise ValueError("optimizer requires trainable parameters")
        if len({name for name, _ in self.parameters}) != len(self.parameters):
            raise ValueError("optimizer parameter names must be unique")
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.state: dict[str, dict[str, Any]] = {}

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        for _, parameter in self.parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    def _state_for(
        self,
        name: str,
        parameter: torch.nn.Parameter,
    ) -> dict[str, Any]:
        state = self.state.get(name)
        if state is None:
            shape = tuple(parameter.shape)
            state = {
                "step": 0,
                "exp_avg": torch.zeros(
                    shape,
                    dtype=torch.float32,
                    device="cpu",
                ),
                "exp_avg_sq": torch.zeros(
                    shape,
                    dtype=torch.float32,
                    device="cpu",
                ),
            }
            self.state[name] = state
        return state

    @torch.no_grad()
    def step(self) -> None:
        for name, parameter in self.parameters:
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach().float().cpu()
            if not bool(torch.isfinite(gradient).all()):
                raise RuntimeError(
                    f"non-finite gradient reached CPU AdamW for {name}"
                )

            state = self._state_for(name, parameter)
            state["step"] = int(state["step"]) + 1
            step = int(state["step"])
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            exp_avg.mul_(self.beta1).add_(
                gradient,
                alpha=1.0 - self.beta1,
            )
            exp_avg_sq.mul_(self.beta2).addcmul_(
                gradient,
                gradient,
                value=1.0 - self.beta2,
            )

            value = parameter.detach().float().cpu()
            if self.weight_decay:
                value.mul_(
                    1.0 - self.learning_rate * self.weight_decay
                )

            bias_correction1 = 1.0 - self.beta1**step
            bias_correction2 = 1.0 - self.beta2**step
            step_size = self.learning_rate / bias_correction1
            denom = exp_avg_sq.sqrt().div_(
                math.sqrt(bias_correction2)
            ).add_(self.eps)
            value.addcdiv_(
                exp_avg,
                denom,
                value=-step_size,
            )
            parameter.copy_(
                value.to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": "VN97R2CPUADAMW1",
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "parameter_names": [name for name, _ in self.parameters],
            "state": {
                name: {
                    "step": int(state["step"]),
                    "exp_avg": state["exp_avg"].clone(),
                    "exp_avg_sq": state["exp_avg_sq"].clone(),
                }
                for name, state in self.state.items()
            },
        }

    def load_state_dict(self, payload: Mapping[str, object]) -> None:
        if payload.get("schema") != "VN97R2CPUADAMW1":
            raise RuntimeError("CPU AdamW state schema mismatch")
        expected_names = [name for name, _ in self.parameters]
        if payload.get("parameter_names") != expected_names:
            raise RuntimeError("CPU AdamW parameter identity mismatch")
        expected_scalars = {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
        }
        for key, expected in expected_scalars.items():
            if float(payload.get(key, float("nan"))) != expected:
                raise RuntimeError(
                    f"CPU AdamW hyperparameter mismatch: {key}"
                )
        raw_state = payload.get("state")
        if not isinstance(raw_state, Mapping):
            raise RuntimeError("CPU AdamW state mapping missing")

        by_name = dict(self.parameters)
        loaded: dict[str, dict[str, Any]] = {}
        for name, raw in raw_state.items():
            if name not in by_name or not isinstance(raw, Mapping):
                raise RuntimeError("CPU AdamW state contains unknown parameter")
            parameter = by_name[name]
            exp_avg = raw.get("exp_avg")
            exp_avg_sq = raw.get("exp_avg_sq")
            if not torch.is_tensor(exp_avg) or not torch.is_tensor(exp_avg_sq):
                raise RuntimeError("CPU AdamW tensor state missing")
            expected_shape = tuple(parameter.shape)
            if (
                tuple(exp_avg.shape) != expected_shape
                or tuple(exp_avg_sq.shape) != expected_shape
            ):
                raise RuntimeError("CPU AdamW tensor shape mismatch")
            loaded[name] = {
                "step": int(raw.get("step", 0)),
                "exp_avg": exp_avg.detach().float().cpu().clone(),
                "exp_avg_sq": exp_avg_sq.detach().float().cpu().clone(),
            }
        self.state = loaded


def _resolve_device(value: str | torch.device) -> torch.device:
    if isinstance(value, torch.device):
        resolved = value
    elif value == "auto":
        resolved = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        resolved = torch.device(value)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return resolved


def _autocast_context(
    device: torch.device,
    precision: str,
):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = (
        torch.float16
        if precision == "fp16"
        else torch.bfloat16
    )
    return torch.autocast(
        device_type="cuda",
        dtype=dtype,
    )


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


def _run_identity(
    model: VN97R2Model,
    manifest: R2ProductionCorpusManifest,
    recipe: R2ProductionTrainingRecipe,
    trainer: R2ProductionTrainerConfig,
) -> str:
    payload = json.dumps(
        {
            "architecture": model.config.fingerprint(),
            "corpus": manifest.identity(),
            "recipe": recipe.fingerprint(),
            "trainer": asdict(trainer),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(
        b"VN97R2PRODTRAIN\0" + payload
    ).hexdigest()


def assert_stage_model_origin(
    manifest: R2ProductionCorpusManifest,
    parent_evidence: Mapping[str, object] | None,
) -> None:
    if manifest.stage == "dense_pretrain":
        if parent_evidence is not None:
            raise ValueError(
                "dense_pretrain must not receive parent checkpoint evidence"
            )
        return

    if parent_evidence is None:
        raise ValueError(
            "post-pretrain stage requires parent checkpoint evidence"
        )
    parent_sha = str(parent_evidence.get("sha256", ""))
    parent_stage = str(parent_evidence.get("stage", ""))
    if parent_sha != manifest.parent_checkpoint_sha256:
        raise ValueError("parent checkpoint SHA-256 does not match manifest")
    assert_stage_transition(
        stage=manifest.stage,
        parent_stage=parent_stage,
    )


def load_production_stage_model(
    manifest: R2ProductionCorpusManifest,
    *,
    config=None,
    parent_checkpoint_path: str | Path | None = None,
) -> tuple[VN97R2Model, dict[str, object] | None]:
    if manifest.stage == "dense_pretrain":
        if config is None:
            raise ValueError("dense_pretrain requires a model config")
        if parent_checkpoint_path is not None:
            raise ValueError(
                "dense_pretrain must not receive a parent checkpoint"
            )
        assert_r2_production_scale(config)
        return VN97R2Model(config), None

    if config is not None:
        raise ValueError(
            "post-pretrain stages derive config from the parent checkpoint"
        )
    if parent_checkpoint_path is None:
        raise ValueError("post-pretrain stage requires parent checkpoint")

    model, evidence = load_r2_checkpoint(
        parent_checkpoint_path,
        map_location="cpu",
    )
    assert_r2_production_scale(model.config)
    assert_stage_model_origin(manifest, evidence)
    return model, evidence


@torch.inference_mode()
def evaluate_production_loss(
    model: VN97R2Model,
    windows: Sequence[VN97TrainingWindow],
    recipe: R2ProductionTrainingRecipe,
    *,
    device: str | torch.device,
) -> dict[str, float | int]:
    if not windows:
        raise ValueError("validation windows must not be empty")
    resolved = _resolve_device(device)
    model.to(resolved)
    model.eval()

    total_loss = 0.0
    total_targets = 0
    correct = 0
    for start in range(0, len(windows), recipe.micro_batch_size):
        indices = list(
            range(
                start,
                min(start + recipe.micro_batch_size, len(windows)),
            )
        )
        inputs, labels = _batch(
            windows,
            indices,
            device=resolved,
        )
        with _autocast_context(resolved, recipe.precision):
            logits, _ = model(inputs, profile="deep")
        mask = labels != IGNORE_INDEX
        count = int(mask.sum().item())
        if count == 0:
            continue
        selected_logits = logits[mask].float()
        selected_labels = labels[mask]
        total_loss += float(
            F.cross_entropy(
                selected_logits,
                selected_labels,
                reduction="sum",
            ).item()
        )
        total_targets += count
        correct += int(
            (
                selected_logits.argmax(dim=-1)
                == selected_labels
            ).sum().item()
        )

    if total_targets <= 0:
        raise RuntimeError("production validation has no target tokens")
    return {
        "target_tokens": total_targets,
        "mean_loss": total_loss / total_targets,
        "top1_accuracy": correct / total_targets,
    }


def measure_cuda_training_preflight(
    model: VN97R2Model,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    recipe: R2ProductionTrainingRecipe,
    *,
    safety_fraction: float = 0.90,
    device: str | torch.device = "cuda",
) -> R2MeasuredMemoryEvidence:
    resolved = _resolve_device(device)
    if resolved.type != "cuda":
        raise ValueError("measured production preflight requires CUDA")
    if not 0.0 < safety_fraction <= 1.0:
        raise ValueError("safety_fraction must be in (0, 1]")
    if inputs.ndim != 2 or labels.shape != inputs.shape:
        raise ValueError("preflight inputs/labels must be [batch, sequence]")
    if inputs.shape[0] > recipe.micro_batch_size:
        raise ValueError("preflight batch exceeds recipe micro_batch_size")
    if inputs.shape[1] != recipe.sequence_length:
        raise ValueError("preflight sequence does not match recipe")

    assert_production_training_contract(model.config, recipe)
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError(
            "measured preflight requires a CPU-resident model at entry"
        )

    torch.cuda.empty_cache()
    free_before, total_bytes = torch.cuda.mem_get_info(resolved)
    torch.cuda.reset_peak_memory_stats(resolved)
    passed = False
    reason = "unknown"
    peak_allocated = 0
    peak_reserved = 0

    try:
        model.to(resolved)
        model.train()
        model.zero_grad(set_to_none=True)
        local_inputs = inputs.to(resolved)
        local_labels = labels.to(resolved)
        with _autocast_context(resolved, recipe.precision):
            logits, _ = model.forward_training(
                local_inputs,
                profile="deep",
                activation_checkpointing=recipe.activation_checkpointing,
            )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                local_labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        loss.backward()
        torch.cuda.synchronize(resolved)
        peak_allocated = int(
            torch.cuda.max_memory_allocated(resolved)
        )
        peak_reserved = int(
            torch.cuda.max_memory_reserved(resolved)
        )
        budget = int(int(free_before) * safety_fraction)
        passed = peak_reserved <= budget
        reason = (
            "within_measured_safety_budget"
            if passed
            else "peak_reserved_exceeds_safety_budget"
        )
    except torch.cuda.OutOfMemoryError:
        peak_allocated = int(
            torch.cuda.max_memory_allocated(resolved)
        )
        peak_reserved = int(
            torch.cuda.max_memory_reserved(resolved)
        )
        passed = False
        reason = "cuda_out_of_memory"
    finally:
        model.zero_grad(set_to_none=True)
        model.to("cpu")
        torch.cuda.empty_cache()

    return R2MeasuredMemoryEvidence(
        architecture_fingerprint=model.config.fingerprint(),
        recipe_fingerprint=recipe.fingerprint(),
        sequence_length=int(inputs.shape[1]),
        micro_batch_size=int(inputs.shape[0]),
        device_name=torch.cuda.get_device_name(resolved),
        free_device_bytes_before=int(free_before),
        total_device_bytes=int(total_bytes),
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        safety_fraction=safety_fraction,
        passed=passed,
        reason=reason,
    )


def _validate_measured_preflight(
    model: VN97R2Model,
    recipe: R2ProductionTrainingRecipe,
    evidence: R2MeasuredMemoryEvidence | None,
) -> None:
    if evidence is None:
        raise RuntimeError(
            "R2-D CUDA training requires measured memory preflight evidence"
        )
    if not evidence.passed:
        raise RuntimeError(
            f"R2-D measured memory preflight failed: {evidence.reason}"
        )
    if evidence.architecture_fingerprint != model.config.fingerprint():
        raise RuntimeError("measured preflight architecture mismatch")
    if evidence.recipe_fingerprint != recipe.fingerprint():
        raise RuntimeError("measured preflight recipe mismatch")
    if evidence.sequence_length != recipe.sequence_length:
        raise RuntimeError("measured preflight sequence mismatch")
    if evidence.micro_batch_size != recipe.micro_batch_size:
        raise RuntimeError("measured preflight batch mismatch")


def _save_resume(
    path: Path,
    *,
    identity: str,
    epoch: int,
    next_group: int,
    model: VN97R2Model,
    optimizer: CPUOffloadedAdamW,
    optimizer_steps: int,
    micro_steps: int,
    target_tokens: int,
    loss_sum: float,
    final_loss: float,
    best_epoch: int,
    best_validation_loss: float,
    best_checkpoint_sha256: str,
    loss_scale: float,
    stable_scaled_steps: int,
    skipped_nonfinite_steps: int,
) -> None:
    temp = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "schema": PRODUCTION_RESUME_SCHEMA,
            "identity": identity,
            "epoch": epoch,
            "next_group": next_group,
            "model_state_dict": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "optimizer_steps": optimizer_steps,
            "micro_steps": micro_steps,
            "target_tokens": target_tokens,
            "loss_sum": loss_sum,
            "final_loss": final_loss,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "best_checkpoint_sha256": best_checkpoint_sha256,
            "loss_scale": loss_scale,
            "stable_scaled_steps": stable_scaled_steps,
            "skipped_nonfinite_steps": skipped_nonfinite_steps,
        },
        temp,
    )
    temp.replace(path)


def _validate_best_checkpoint(
    path: Path,
    *,
    best_epoch: int,
    best_checkpoint_sha256: str,
) -> str:
    if best_epoch < 0:
        if best_checkpoint_sha256:
            raise RuntimeError(
                "resume contains best checkpoint identity without epoch"
            )
        return ""
    if not path.is_file():
        raise RuntimeError("resume references missing best checkpoint")
    actual = sha256_file(path)
    if best_checkpoint_sha256 and actual != best_checkpoint_sha256:
        raise RuntimeError("best checkpoint SHA-256 mismatch on resume")
    return actual


def train_production_stage(
    model: VN97R2Model,
    training_windows: Sequence[VN97TrainingWindow],
    validation_windows: Sequence[VN97TrainingWindow],
    manifest: R2ProductionCorpusManifest,
    recipe: R2ProductionTrainingRecipe,
    trainer: R2ProductionTrainerConfig,
    *,
    work_dir: str | Path,
    best_checkpoint_path: str | Path,
    parent_evidence: Mapping[str, object] | None = None,
    device: str | torch.device = "auto",
    measured_preflight: R2MeasuredMemoryEvidence | None = None,
    max_run_seconds: float | None = None,
) -> R2ProductionTrainingResult:
    if not training_windows or not validation_windows:
        raise ValueError("production train/validation windows must be non-empty")
    if max_run_seconds is not None and max_run_seconds <= 0.0:
        raise ValueError("max_run_seconds must be positive when provided")

    assert_r2_production_scale(model.config)
    assert_stage_model_origin(manifest, parent_evidence)

    for split_name, windows in (
        ("training", training_windows),
        ("validation", validation_windows),
    ):
        if any(
            len(window.input_ids) != recipe.sequence_length
            or len(window.labels) != recipe.sequence_length
            for window in windows
        ):
            raise ValueError(
                f"{split_name} windows do not match recipe sequence_length"
            )

    resolved = _resolve_device(device)
    if not recipe.optimizer_state_offload:
        raise ValueError(
            "R2-D3 trainer requires optimizer_state_offload=true"
        )
    if (
        resolved.type == "cuda"
        and recipe.precision == "bf16"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError(
            "requested BF16 is not supported by this CUDA device"
        )

    available_device = None
    if resolved.type == "cuda":
        available_device = int(torch.cuda.mem_get_info(resolved)[0])
    available_host = available_memory_bytes()
    assert_production_training_contract(
        model.config,
        recipe,
        available_device_bytes=available_device,
        available_host_bytes=available_host,
    )
    if (
        resolved.type == "cuda"
        and trainer.require_measured_cuda_preflight
    ):
        _validate_measured_preflight(model, recipe, measured_preflight)

    torch.manual_seed(trainer.seed)
    if resolved.type == "cuda":
        torch.cuda.manual_seed_all(trainer.seed)

    model.to(resolved)
    optimizer = CPUOffloadedAdamW(
        model.named_parameters(),
        learning_rate=trainer.learning_rate,
        weight_decay=trainer.weight_decay,
        beta1=trainer.beta1,
        beta2=trainer.beta2,
        eps=trainer.eps,
    )
    identity = _run_identity(model, manifest, recipe, trainer)

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    resume_path = work / "production-resume.pt"
    best_path = Path(best_checkpoint_path)

    start_epoch = 0
    start_group = 0
    optimizer_steps = 0
    micro_steps = 0
    target_tokens = 0
    loss_sum = 0.0
    final_loss = float("nan")
    best_epoch = -1
    best_validation_loss = float("inf")
    best_checkpoint_sha256 = ""
    use_loss_scaling = (
        resolved.type == "cuda" and recipe.precision == "fp16"
    )
    loss_scale = (
        trainer.initial_loss_scale if use_loss_scaling else 1.0
    )
    stable_scaled_steps = 0
    skipped_nonfinite_steps = 0

    if resume_path.is_file():
        resume = torch.load(
            resume_path,
            map_location="cpu",
            weights_only=False,
        )
        if (
            not isinstance(resume, dict)
            or resume.get("schema") != PRODUCTION_RESUME_SCHEMA
            or resume.get("identity") != identity
        ):
            raise RuntimeError("R2 production resume identity mismatch")
        model.load_state_dict(resume["model_state_dict"], strict=True)
        model.to(resolved)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        start_epoch = int(resume["epoch"])
        start_group = int(resume["next_group"])
        optimizer_steps = int(resume["optimizer_steps"])
        micro_steps = int(resume["micro_steps"])
        target_tokens = int(resume["target_tokens"])
        loss_sum = float(resume["loss_sum"])
        final_loss = float(resume["final_loss"])
        best_epoch = int(resume["best_epoch"])
        best_validation_loss = float(
            resume["best_validation_loss"]
        )
        best_checkpoint_sha256 = _validate_best_checkpoint(
            best_path,
            best_epoch=best_epoch,
            best_checkpoint_sha256=str(
                resume.get("best_checkpoint_sha256", "")
            ),
        )
        loss_scale = float(resume["loss_scale"])
        stable_scaled_steps = int(resume["stable_scaled_steps"])
        skipped_nonfinite_steps = int(
            resume["skipped_nonfinite_steps"]
        )

    run_started = time.monotonic()
    paused = False
    pause_next_group = start_group

    for epoch in range(start_epoch, trainer.epochs):
        order = list(range(len(training_windows)))
        if len(order) > 1:
            random.Random(trainer.seed + epoch).shuffle(order)

        micro_batches = [
            order[start : start + recipe.micro_batch_size]
            for start in range(
                0,
                len(order),
                recipe.micro_batch_size,
            )
        ]
        groups = [
            micro_batches[
                start : start + recipe.gradient_accumulation_steps
            ]
            for start in range(
                0,
                len(micro_batches),
                recipe.gradient_accumulation_steps,
            )
        ]
        group_begin = start_group if epoch == start_epoch else 0
        if not 0 <= group_begin <= len(groups):
            raise RuntimeError("production resume group is out of range")

        model.train()
        for group_index in range(group_begin, len(groups)):
            group = groups[group_index]
            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.0
            group_targets = 0

            for indices in group:
                inputs, labels = _batch(
                    training_windows,
                    indices,
                    device=resolved,
                )
                with _autocast_context(resolved, recipe.precision):
                    logits, _ = model.forward_training(
                        inputs,
                        profile="deep",
                        activation_checkpointing=(
                            recipe.activation_checkpointing
                        ),
                    )
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]).float(),
                        labels.reshape(-1),
                        ignore_index=IGNORE_INDEX,
                    )
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(
                        "R2 production loss became non-finite"
                    )
                scaled = (
                    loss
                    / len(group)
                    * loss_scale
                )
                scaled.backward()

                value = float(loss.detach().cpu())
                group_loss += value
                loss_sum += value
                final_loss = value
                count = int(
                    (labels != IGNORE_INDEX).sum().item()
                )
                group_targets += count
                target_tokens += count
                micro_steps += 1

            if use_loss_scaling and loss_scale != 1.0:
                inverse = 1.0 / loss_scale
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(inverse)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                trainer.max_grad_norm,
            )
            finite_gradients = bool(
                torch.isfinite(torch.as_tensor(grad_norm)).item()
            )
            if not finite_gradients:
                optimizer.zero_grad(set_to_none=True)
                skipped_nonfinite_steps += 1
                stable_scaled_steps = 0
                if use_loss_scaling:
                    loss_scale = max(
                        trainer.min_loss_scale,
                        loss_scale / 2.0,
                    )
            else:
                optimizer.step()
                optimizer_steps += 1
                if use_loss_scaling:
                    stable_scaled_steps += 1
                    if (
                        stable_scaled_steps
                        >= trainer.loss_scale_growth_interval
                    ):
                        loss_scale = min(
                            trainer.max_loss_scale,
                            loss_scale * 2.0,
                        )
                        stable_scaled_steps = 0

            optimizer.zero_grad(set_to_none=True)

            if (
                optimizer_steps > 0
                and optimizer_steps
                % trainer.checkpoint_every_optimizer_steps
                == 0
            ):
                _save_resume(
                    resume_path,
                    identity=identity,
                    epoch=epoch,
                    next_group=group_index + 1,
                    model=model,
                    optimizer=optimizer,
                    optimizer_steps=optimizer_steps,
                    micro_steps=micro_steps,
                    target_tokens=target_tokens,
                    loss_sum=loss_sum,
                    final_loss=final_loss,
                    best_epoch=best_epoch,
                    best_validation_loss=best_validation_loss,
                    best_checkpoint_sha256=best_checkpoint_sha256,
                    loss_scale=loss_scale,
                    stable_scaled_steps=stable_scaled_steps,
                    skipped_nonfinite_steps=skipped_nonfinite_steps,
                )

            if (
                max_run_seconds is not None
                and time.monotonic() - run_started >= max_run_seconds
            ):
                paused = True
                pause_next_group = group_index + 1
                break

        validation = evaluate_production_loss(
            model,
            validation_windows,
            recipe,
            device=resolved,
        )
        validation_loss = float(validation["mean_loss"])
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_checkpoint_sha256 = save_r2_checkpoint(
                best_path,
                model,
                stage=manifest.stage,
                metadata={
                    "production_corpus_identity": manifest.identity(),
                    "production_recipe_fingerprint": recipe.fingerprint(),
                    "production_trainer": asdict(trainer),
                    "optimizer_steps": optimizer_steps,
                    "micro_steps": micro_steps,
                    "validation": validation,
                    "parent_checkpoint_sha256": (
                        manifest.parent_checkpoint_sha256
                    ),
                },
            )

        if paused:
            # Persist again after validation so best-checkpoint epoch/SHA
            # cannot lag behind a checkpoint promoted by this paused epoch.
            _save_resume(
                resume_path,
                identity=identity,
                epoch=epoch,
                next_group=pause_next_group,
                model=model,
                optimizer=optimizer,
                optimizer_steps=optimizer_steps,
                micro_steps=micro_steps,
                target_tokens=target_tokens,
                loss_sum=loss_sum,
                final_loss=final_loss,
                best_epoch=best_epoch,
                best_validation_loss=best_validation_loss,
                best_checkpoint_sha256=best_checkpoint_sha256,
                loss_scale=loss_scale,
                stable_scaled_steps=stable_scaled_steps,
                skipped_nonfinite_steps=skipped_nonfinite_steps,
            )
            break

        if epoch + 1 < trainer.epochs:
            _save_resume(
                resume_path,
                identity=identity,
                epoch=epoch + 1,
                next_group=0,
                model=model,
                optimizer=optimizer,
                optimizer_steps=optimizer_steps,
                micro_steps=micro_steps,
                target_tokens=target_tokens,
                loss_sum=loss_sum,
                final_loss=final_loss,
                best_epoch=best_epoch,
                best_validation_loss=best_validation_loss,
                best_checkpoint_sha256=best_checkpoint_sha256,
                loss_scale=loss_scale,
                stable_scaled_steps=stable_scaled_steps,
                skipped_nonfinite_steps=skipped_nonfinite_steps,
            )
        start_group = 0

    if optimizer_steps <= 0 or micro_steps <= 0 or target_tokens <= 0:
        raise RuntimeError("R2 production training completed without work")
    if best_epoch < 0 or not best_checkpoint_sha256:
        raise RuntimeError("R2 production training selected no checkpoint")

    if paused:
        if not resume_path.is_file():
            raise RuntimeError(
                "R2 production pause did not persist resume state"
            )
        return R2ProductionTrainingResult(
            optimizer_steps=optimizer_steps,
            micro_steps=micro_steps,
            target_tokens=target_tokens,
            mean_loss=loss_sum / micro_steps,
            final_loss=final_loss,
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
            best_checkpoint_sha256=best_checkpoint_sha256,
            completed=False,
            resume_checkpoint_sha256=sha256_file(resume_path),
            loss_scale=loss_scale,
            skipped_nonfinite_steps=skipped_nonfinite_steps,
        )

    resume_path.unlink(missing_ok=True)
    return R2ProductionTrainingResult(
        optimizer_steps=optimizer_steps,
        micro_steps=micro_steps,
        target_tokens=target_tokens,
        mean_loss=loss_sum / micro_steps,
        final_loss=final_loss,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        best_checkpoint_sha256=best_checkpoint_sha256,
        completed=True,
        resume_checkpoint_sha256="",
        loss_scale=loss_scale,
        skipped_nonfinite_steps=skipped_nonfinite_steps,
    )
