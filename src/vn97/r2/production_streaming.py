from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F

from ..training import (
    IGNORE_INDEX,
    VN97ChatMessage,
    VN97TrainingWindow,
    encode_chat_completion_messages,
    make_training_windows,
)
from .checkpoint import save_r2_checkpoint, sha256_file
from .data_bridge import load_vn97tk1
from .model import VN97R2Model
from .pilot_contract import available_memory_bytes
from .production_contract import (
    R2ProductionCorpusManifest,
    R2ProductionShard,
    R2ProductionTrainingRecipe,
    assert_production_training_contract,
    assert_r2_production_scale,
)
from .production_corpus_scale import (
    R2D6_DEFAULT_MIN_TOKENS_PER_PARAMETER,
    R2D6_DEFAULT_TARGET_TOKENS_PER_PARAMETER,
    verify_r2d6_corpus_index,
)
from .production_training import (
    CPUOffloadedAdamW,
    R2MeasuredMemoryEvidence,
    R2ProductionTrainerConfig,
    _assert_model_provenance,
    _autocast_context,
    _resolve_device,
    _validate_best_checkpoint,
    _validate_measured_preflight,
)


R2D7_STREAMING_RESUME_SCHEMA = "VN97R2STREAMRESUME1"
R2D7_PREFLIGHT_RECEIPT_SCHEMA = "VN97R2D5PREFLIGHT1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


@dataclass(frozen=True)
class R2StreamingCursor:
    epoch: int
    shard_position: int
    record_index: int
    window_index: int

    def __post_init__(self) -> None:
        if min(
            self.epoch,
            self.shard_position,
            self.record_index,
            self.window_index,
        ) < 0:
            raise ValueError("R2-D7 cursor values must be non-negative")

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class R2StreamingTrainingResult:
    optimizer_steps: int
    micro_steps: int
    consumed_windows: int
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
    next_cursor: R2StreamingCursor


def production_manifest_from_r2d6(
    index: Mapping[str, object],
) -> R2ProductionCorpusManifest:
    tokenizer_sha = _require_sha256(
        index.get("tokenizer_sha256"),
        label="R2-D7 tokenizer SHA-256",
    )
    raw_shards = index.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValueError("R2-D7 D6 shard list is invalid")

    shards: list[R2ProductionShard] = []
    for raw in raw_shards:
        if not isinstance(raw, Mapping):
            raise ValueError("R2-D7 D6 shard entry is invalid")
        split = raw.get("split")
        if split == "release":
            continue
        mapped_split = {
            "training": "train",
            "validation": "validation",
        }.get(str(split))
        if mapped_split is None:
            raise ValueError("R2-D7 encountered unsupported D6 split")
        families = raw.get("task_families")
        if (
            not isinstance(families, list)
            or not families
            or any(not isinstance(item, str) or not item for item in families)
        ):
            raise ValueError("R2-D7 D6 task families are invalid")
        shards.append(
            R2ProductionShard(
                split=mapped_split,
                sha256=_require_sha256(
                    raw.get("sha256"),
                    label="R2-D7 shard SHA-256",
                ),
                bytes=int(raw.get("bytes", 0)),
                records=int(raw.get("records", 0)),
                task_families=tuple(families),
            )
        )

    return R2ProductionCorpusManifest(
        stage="dense_pretrain",
        tokenizer_sha256=tokenizer_sha,
        shards=tuple(shards),
        parent_checkpoint_sha256=None,
    )


def _split_shards(
    index: Mapping[str, object],
    split: str,
) -> tuple[dict[str, object], ...]:
    raw = index.get("shards")
    if not isinstance(raw, list):
        raise ValueError("R2-D7 D6 shard list is missing")
    selected: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("R2-D7 shard entry is invalid")
        if item.get("split") == split:
            selected.append(item)
    if not selected:
        raise ValueError(f"R2-D7 has no {split} shards")
    selected.sort(
        key=lambda item: (
            str(item["source_manifest_id"]),
            str(item["shard_id"]),
        )
    )
    return tuple(selected)


def _epoch_shard_order(
    training_shards: Sequence[dict[str, object]],
    *,
    seed: int,
    epoch: int,
) -> tuple[dict[str, object], ...]:
    order = list(training_shards)
    if len(order) > 1:
        random.Random(seed + epoch).shuffle(order)
    return tuple(order)


def _epoch_order_digest(
    order: Sequence[Mapping[str, object]],
) -> str:
    return hashlib.sha256(
        b"VN97R2D7ORDER1\0"
        + _canonical_json(
            [str(item["shard_id"]) for item in order]
        )
    ).hexdigest()


def _parse_chat_record(
    line: str,
    *,
    path: Path,
    line_number: int,
) -> tuple[VN97ChatMessage, ...]:
    try:
        value = json.loads(
            line,
            parse_constant=lambda raw: (
                _ for _ in ()
            ).throw(ValueError(raw)),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            f"R2-D7 invalid JSONL at {path}:{line_number}"
        ) from exc
    if (
        not isinstance(value, dict)
        or set(value) != {"messages"}
        or not isinstance(value["messages"], list)
        or not value["messages"]
    ):
        raise ValueError(
            f"R2-D7 chat record is invalid at {path}:{line_number}"
        )
    messages: list[VN97ChatMessage] = []
    for raw in value["messages"]:
        if (
            not isinstance(raw, dict)
            or set(raw) != {"role", "content"}
        ):
            raise ValueError(
                f"R2-D7 chat message is invalid at {path}:{line_number}"
            )
        messages.append(
            VN97ChatMessage(
                role=raw["role"],
                content=raw["content"],
            )
        )
    return tuple(messages)


def _iter_chat_records(
    path: Path,
    *,
    start_record: int = 0,
) -> Iterator[tuple[int, tuple[VN97ChatMessage, ...]]]:
    if start_record < 0:
        raise ValueError("R2-D7 start_record must be non-negative")
    if path.is_symlink() or not path.is_file():
        raise ValueError("R2-D7 shard path must be a regular file")
    record_index = 0
    try:
        with path.open(
            "r",
            encoding="utf-8",
            errors="strict",
        ) as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                current = record_index
                record_index += 1
                if current < start_record:
                    continue
                yield (
                    current,
                    _parse_chat_record(
                        line,
                        path=path,
                        line_number=line_number,
                    ),
                )
    except UnicodeDecodeError as exc:
        raise ValueError("R2-D7 shard is not strict UTF-8") from exc


def _next_window_cursor(
    *,
    epoch: int,
    shard_position: int,
    record_index: int,
    window_index: int,
    windows_in_record: int,
    records_in_shard: int,
    shards_in_epoch: int,
) -> R2StreamingCursor:
    if window_index + 1 < windows_in_record:
        return R2StreamingCursor(
            epoch,
            shard_position,
            record_index,
            window_index + 1,
        )
    if record_index + 1 < records_in_shard:
        return R2StreamingCursor(
            epoch,
            shard_position,
            record_index + 1,
            0,
        )
    if shard_position + 1 < shards_in_epoch:
        return R2StreamingCursor(
            epoch,
            shard_position + 1,
            0,
            0,
        )
    return R2StreamingCursor(epoch + 1, 0, 0, 0)


def iter_epoch_training_windows(
    package_dir: Path,
    index: Mapping[str, object],
    tokenizer,
    *,
    cursor: R2StreamingCursor,
    seed: int,
    sequence_length: int,
) -> Iterator[tuple[VN97TrainingWindow, R2StreamingCursor]]:
    training_shards = _split_shards(index, "training")
    order = _epoch_shard_order(
        training_shards,
        seed=seed,
        epoch=cursor.epoch,
    )
    if cursor.shard_position >= len(order):
        raise RuntimeError("R2-D7 cursor shard position is out of range")

    root = package_dir.resolve(strict=True)
    for shard_position in range(cursor.shard_position, len(order)):
        shard = order[shard_position]
        records_in_shard = int(shard["records"])
        if records_in_shard <= 0:
            raise RuntimeError("R2-D7 shard record count is invalid")
        start_record = (
            cursor.record_index
            if shard_position == cursor.shard_position
            else 0
        )
        start_window = (
            cursor.window_index
            if shard_position == cursor.shard_position
            else 0
        )
        if start_record >= records_in_shard:
            raise RuntimeError("R2-D7 cursor record is out of range")

        shard_path = root / str(shard["filename"])
        seen_records = 0
        for record_index, messages in _iter_chat_records(
            shard_path,
            start_record=start_record,
        ):
            seen_records += 1
            example = encode_chat_completion_messages(
                tokenizer,
                messages,
            )
            windows = make_training_windows(
                example,
                sequence_length=sequence_length,
                stride=sequence_length,
                pad_token_id=tokenizer.pad_id,
            )
            begin = (
                start_window
                if record_index == start_record
                else 0
            )
            if begin >= len(windows):
                raise RuntimeError(
                    "R2-D7 cursor window is out of range"
                )
            for window_index in range(begin, len(windows)):
                next_cursor = _next_window_cursor(
                    epoch=cursor.epoch,
                    shard_position=shard_position,
                    record_index=record_index,
                    window_index=window_index,
                    windows_in_record=len(windows),
                    records_in_shard=records_in_shard,
                    shards_in_epoch=len(order),
                )
                yield windows[window_index], next_cursor
            start_window = 0

        expected_remaining = records_in_shard - start_record
        if seen_records != expected_remaining:
            raise RuntimeError(
                "R2-D7 shard record count changed after D6 verification"
            )


def _iter_validation_windows(
    package_dir: Path,
    index: Mapping[str, object],
    tokenizer,
    *,
    sequence_length: int,
) -> Iterator[VN97TrainingWindow]:
    root = package_dir.resolve(strict=True)
    for shard in _split_shards(index, "validation"):
        path = root / str(shard["filename"])
        expected_records = int(shard["records"])
        observed = 0
        for _, messages in _iter_chat_records(path):
            observed += 1
            example = encode_chat_completion_messages(
                tokenizer,
                messages,
            )
            yield from make_training_windows(
                example,
                sequence_length=sequence_length,
                stride=sequence_length,
                pad_token_id=tokenizer.pad_id,
            )
        if observed != expected_records:
            raise RuntimeError(
                "R2-D7 validation shard record count changed"
            )


def _windows_to_batch(
    windows: Sequence[VN97TrainingWindow],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not windows:
        raise ValueError("R2-D7 micro-batch must not be empty")
    return (
        torch.tensor(
            [item.input_ids for item in windows],
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            [item.labels for item in windows],
            dtype=torch.long,
            device=device,
        ),
    )


@torch.inference_mode()
def evaluate_streaming_validation(
    model: VN97R2Model,
    package_dir: Path,
    index: Mapping[str, object],
    tokenizer,
    recipe: R2ProductionTrainingRecipe,
    *,
    device: str | torch.device,
) -> dict[str, float | int]:
    resolved = _resolve_device(device)
    model.to(resolved)
    model.eval()

    total_loss = 0.0
    total_targets = 0
    correct = 0
    batch: list[VN97TrainingWindow] = []

    def consume(rows: Sequence[VN97TrainingWindow]) -> None:
        nonlocal total_loss, total_targets, correct
        inputs, labels = _windows_to_batch(
            rows,
            device=resolved,
        )
        with _autocast_context(resolved, recipe.precision):
            logits, _ = model(inputs, profile="deep")
        mask = labels != IGNORE_INDEX
        count = int(mask.sum().item())
        if count <= 0:
            return
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

    for window in _iter_validation_windows(
        package_dir,
        index,
        tokenizer,
        sequence_length=recipe.sequence_length,
    ):
        batch.append(window)
        if len(batch) == recipe.micro_batch_size:
            consume(batch)
            batch = []
    if batch:
        consume(batch)

    if total_targets <= 0:
        raise RuntimeError("R2-D7 validation has no target tokens")
    return {
        "target_tokens": total_targets,
        "mean_loss": total_loss / total_targets,
        "top1_accuracy": correct / total_targets,
    }


def _stream_run_identity(
    model: VN97R2Model,
    index: Mapping[str, object],
    manifest: R2ProductionCorpusManifest,
    recipe: R2ProductionTrainingRecipe,
    trainer: R2ProductionTrainerConfig,
) -> str:
    payload = {
        "architecture": model.config.fingerprint(),
        "d6_index_id": _require_sha256(
            index.get("index_id"),
            label="R2-D7 D6 index ID",
        ),
        "production_manifest": manifest.identity(),
        "recipe": recipe.fingerprint(),
        "resume_schema": R2D7_STREAMING_RESUME_SCHEMA,
        "trainer": asdict(trainer),
    }
    return hashlib.sha256(
        b"VN97R2D7TRAIN\0" + _canonical_json(payload)
    ).hexdigest()


def _save_stream_resume(
    path: Path,
    *,
    identity: str,
    d6_index_id: str,
    cursor: R2StreamingCursor,
    epoch_order_digest: str,
    model: VN97R2Model,
    optimizer: CPUOffloadedAdamW,
    optimizer_steps: int,
    micro_steps: int,
    consumed_windows: int,
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
            "schema": R2D7_STREAMING_RESUME_SCHEMA,
            "identity": identity,
            "d6_index_id": d6_index_id,
            "cursor": cursor.as_dict(),
            "epoch_order_digest": epoch_order_digest,
            "accumulation_open": False,
            "model_state_dict": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "optimizer_steps": optimizer_steps,
            "micro_steps": micro_steps,
            "consumed_windows": consumed_windows,
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


def _cursor_order_digest(
    index: Mapping[str, object],
    *,
    cursor: R2StreamingCursor,
    seed: int,
    epochs: int,
) -> str:
    if cursor.epoch >= epochs:
        return ""
    return _epoch_order_digest(
        _epoch_shard_order(
            _split_shards(index, "training"),
            seed=seed,
            epoch=cursor.epoch,
        )
    )


def _load_stream_resume(
    path: Path,
    *,
    identity: str,
    d6_index_id: str,
    index: Mapping[str, object],
    seed: int,
    epochs: int,
) -> dict[str, object]:
    resume = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    if (
        not isinstance(resume, dict)
        or resume.get("schema") != R2D7_STREAMING_RESUME_SCHEMA
        or resume.get("identity") != identity
        or resume.get("d6_index_id") != d6_index_id
    ):
        raise RuntimeError("R2-D7 streaming resume identity mismatch")
    if resume.get("accumulation_open") is not False:
        raise RuntimeError(
            "R2-D7 resume was not saved at an optimizer boundary"
        )
    raw_cursor = resume.get("cursor")
    if not isinstance(raw_cursor, dict):
        raise RuntimeError("R2-D7 resume cursor is missing")
    try:
        cursor = R2StreamingCursor(**raw_cursor)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("R2-D7 resume cursor is invalid") from exc
    if cursor.epoch > epochs:
        raise RuntimeError("R2-D7 resume epoch is out of range")
    expected_order = _cursor_order_digest(
        index,
        cursor=cursor,
        seed=seed,
        epochs=epochs,
    )
    if resume.get("epoch_order_digest") != expected_order:
        raise RuntimeError("R2-D7 resume shard order mismatch")
    resume["cursor_object"] = cursor
    return resume


def load_r2d5_memory_receipt(
    path: Path,
    *,
    architecture_fingerprint: str,
    recipe: R2ProductionTrainingRecipe,
) -> R2MeasuredMemoryEvidence:
    try:
        payload = json.loads(
            path.resolve(strict=True).read_text(
                encoding="utf-8",
                errors="strict",
            )
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "R2-D7 preflight receipt must be UTF-8 JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("R2-D7 preflight receipt must be an object")
    if payload.get("schema") != R2D7_PREFLIGHT_RECEIPT_SCHEMA:
        raise ValueError("R2-D7 preflight receipt schema mismatch")
    receipt_id = _require_sha256(
        payload.get("receipt_id"),
        label="R2-D7 preflight receipt ID",
    )
    body = dict(payload)
    body.pop("receipt_id", None)
    expected = hashlib.sha256(
        b"VN97R2D5PREFLIGHT1\0"
        + _canonical_json(body)
    ).hexdigest()
    if receipt_id != expected:
        raise ValueError("R2-D7 preflight receipt identity mismatch")
    if payload.get("training_allowed") is not False:
        raise ValueError("R2-D7 preflight receipt must be no-promotion")
    if payload.get("passed") is not True:
        raise ValueError("R2-D7 preflight receipt did not pass")
    if payload.get("architecture_fingerprint") != architecture_fingerprint:
        raise ValueError("R2-D7 preflight architecture mismatch")
    if payload.get("recipe_fingerprint") != recipe.fingerprint():
        raise ValueError("R2-D7 preflight recipe mismatch")
    if int(payload.get("sequence_length", -1)) != recipe.sequence_length:
        raise ValueError("R2-D7 preflight sequence mismatch")
    if int(payload.get("micro_batch_size", -1)) != recipe.micro_batch_size:
        raise ValueError("R2-D7 preflight micro-batch mismatch")

    device_name = str(payload.get("device_name", ""))
    reason = str(payload.get("reason", ""))
    free_bytes = int(payload.get("free_device_bytes_before", -1))
    total_bytes = int(payload.get("total_device_bytes", -1))
    peak_allocated = int(payload.get("peak_allocated_bytes", -1))
    peak_reserved = int(payload.get("peak_reserved_bytes", -1))
    safety_fraction = float(
        payload.get("safety_fraction", float("nan"))
    )
    if not device_name:
        raise ValueError("R2-D7 preflight device name is missing")
    if reason != "within_measured_safety_budget":
        raise ValueError("R2-D7 preflight pass reason is invalid")
    if (
        free_bytes < 0
        or total_bytes <= 0
        or free_bytes > total_bytes
        or peak_allocated < 0
        or peak_reserved < peak_allocated
        or not math.isfinite(safety_fraction)
        or not 0.0 < safety_fraction <= 1.0
        or peak_reserved > int(free_bytes * safety_fraction)
    ):
        raise ValueError(
            "R2-D7 preflight memory evidence is internally inconsistent"
        )

    return R2MeasuredMemoryEvidence(
        architecture_fingerprint=architecture_fingerprint,
        recipe_fingerprint=recipe.fingerprint(),
        sequence_length=recipe.sequence_length,
        micro_batch_size=recipe.micro_batch_size,
        device_name=device_name,
        free_device_bytes_before=free_bytes,
        total_device_bytes=total_bytes,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        safety_fraction=safety_fraction,
        passed=True,
        reason=reason,
    )


def _validate_stream_package(
    model: VN97R2Model,
    package_dir: Path,
    index: Mapping[str, object],
    recipe: R2ProductionTrainingRecipe,
) -> None:
    assert_r2_production_scale(model.config)
    if index.get("architecture_fingerprint") != model.config.fingerprint():
        raise ValueError("R2-D7 D6 architecture fingerprint mismatch")
    if int(index.get("sequence_length", -1)) != recipe.sequence_length:
        raise ValueError("R2-D7 D6 sequence length mismatch")
    scale = index.get("scale")
    if not isinstance(scale, Mapping):
        raise ValueError("R2-D7 D6 scale evidence is missing")
    minimum_ratio = float(
        scale.get("minimum_tokens_per_parameter", float("nan"))
    )
    target_ratio = float(
        scale.get("target_tokens_per_parameter", float("nan"))
    )
    if (
        not math.isclose(
            minimum_ratio,
            R2D6_DEFAULT_MIN_TOKENS_PER_PARAMETER,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            target_ratio,
            R2D6_DEFAULT_TARGET_TOKENS_PER_PARAMETER,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise RuntimeError(
            "R2-D7 requires the canonical D6 8/20 token-per-parameter "
            "scale policy"
        )
    if scale.get("scale_floor_passed") is not True:
        raise RuntimeError(
            "R2-D7 refuses production training below the D6 scale floor"
        )
    if int(scale.get("parameter_count", -1)) != model.config.estimated_parameter_count():
        raise ValueError("R2-D7 D6 parameter count mismatch")
    tokenizer_sha = _require_sha256(
        index.get("tokenizer_sha256"),
        label="R2-D7 tokenizer SHA-256",
    )
    if sha256_file(
        package_dir.resolve(strict=True) / "tokenizer.vn97tk1"
    ) != tokenizer_sha:
        raise ValueError("R2-D7 tokenizer identity mismatch")


def train_streaming_production_stage(
    model: VN97R2Model,
    package_dir: str | Path,
    recipe: R2ProductionTrainingRecipe,
    trainer: R2ProductionTrainerConfig,
    *,
    work_dir: str | Path,
    best_checkpoint_path: str | Path,
    device: str | torch.device = "auto",
    measured_preflight: R2MeasuredMemoryEvidence | None = None,
    max_run_seconds: float | None = None,
) -> R2StreamingTrainingResult:
    if max_run_seconds is not None and max_run_seconds <= 0.0:
        raise ValueError("max_run_seconds must be positive when provided")

    package = Path(package_dir)
    index = verify_r2d6_corpus_index(package)
    _validate_stream_package(model, package, index, recipe)

    manifest = production_manifest_from_r2d6(index)
    _assert_model_provenance(
        model,
        manifest,
        trainer,
        None,
    )

    resolved = _resolve_device(device)
    if not recipe.optimizer_state_offload:
        raise ValueError(
            "R2-D7 trainer requires optimizer_state_offload=true"
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
    assert_production_training_contract(
        model.config,
        recipe,
        available_device_bytes=available_device,
        available_host_bytes=available_memory_bytes(),
    )
    if (
        resolved.type == "cuda"
        and trainer.require_measured_cuda_preflight
    ):
        _validate_measured_preflight(
            model,
            recipe,
            measured_preflight,
        )
        current_device_name = torch.cuda.get_device_name(resolved)
        if (
            measured_preflight is None
            or measured_preflight.device_name != current_device_name
        ):
            raise RuntimeError(
                "R2-D7 CUDA device differs from measured preflight device"
            )

    tokenizer = load_vn97tk1(
        package.resolve(strict=True) / "tokenizer.vn97tk1"
    )
    if tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError("R2-D7 tokenizer/model vocabulary mismatch")

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
    identity = _stream_run_identity(
        model,
        index,
        manifest,
        recipe,
        trainer,
    )
    d6_index_id = str(index["index_id"])

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    resume_path = work / "streaming-production-resume.pt"
    best_path = Path(best_checkpoint_path)

    cursor = R2StreamingCursor(0, 0, 0, 0)
    optimizer_steps = 0
    micro_steps = 0
    consumed_windows = 0
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
        resume = _load_stream_resume(
            resume_path,
            identity=identity,
            d6_index_id=d6_index_id,
            index=index,
            seed=trainer.seed,
            epochs=trainer.epochs,
        )
        model.load_state_dict(
            resume["model_state_dict"],
            strict=True,
        )
        model.to(resolved)
        optimizer.load_state_dict(
            resume["optimizer_state_dict"]
        )
        cursor = resume["cursor_object"]
        optimizer_steps = int(resume["optimizer_steps"])
        micro_steps = int(resume["micro_steps"])
        consumed_windows = int(resume["consumed_windows"])
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
        stable_scaled_steps = int(
            resume["stable_scaled_steps"]
        )
        skipped_nonfinite_steps = int(
            resume["skipped_nonfinite_steps"]
        )

    run_started = time.monotonic()
    paused = False

    while cursor.epoch < trainer.epochs:
        epoch = cursor.epoch
        stream = iter_epoch_training_windows(
            package,
            index,
            tokenizer,
            cursor=cursor,
            seed=trainer.seed,
            sequence_length=recipe.sequence_length,
        )
        model.train()

        while cursor.epoch == epoch:
            group: list[
                tuple[
                    list[VN97TrainingWindow],
                    R2StreamingCursor,
                ]
            ] = []
            exhausted = False
            for _ in range(recipe.gradient_accumulation_steps):
                rows: list[VN97TrainingWindow] = []
                next_cursor = cursor
                for _ in range(recipe.micro_batch_size):
                    try:
                        window, next_cursor = next(stream)
                    except StopIteration:
                        exhausted = True
                        break
                    rows.append(window)
                if rows:
                    group.append((rows, next_cursor))
                if exhausted:
                    break

            if not group:
                if cursor.epoch == epoch:
                    raise RuntimeError(
                        "R2-D7 stream ended before epoch cursor advanced"
                    )
                break

            optimizer.zero_grad(set_to_none=True)
            for rows, next_cursor in group:
                inputs, labels = _windows_to_batch(
                    rows,
                    device=resolved,
                )
                with _autocast_context(
                    resolved,
                    recipe.precision,
                ):
                    logits, _ = model.forward_training(
                        inputs,
                        profile="deep",
                        activation_checkpointing=(
                            recipe.activation_checkpointing
                        ),
                    )
                    loss = F.cross_entropy(
                        logits.reshape(
                            -1,
                            logits.shape[-1],
                        ).float(),
                        labels.reshape(-1),
                        ignore_index=IGNORE_INDEX,
                    )
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(
                        "R2-D7 production loss became non-finite"
                    )
                (
                    loss
                    / len(group)
                    * loss_scale
                ).backward()

                value = float(loss.detach().cpu())
                loss_sum += value
                final_loss = value
                target_tokens += int(
                    (labels != IGNORE_INDEX).sum().item()
                )
                micro_steps += 1
                consumed_windows += len(rows)
                cursor = next_cursor

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
                torch.isfinite(
                    torch.as_tensor(grad_norm)
                ).item()
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

            checkpoint_due = (
                (
                    optimizer_steps > 0
                    and optimizer_steps
                    % trainer.checkpoint_every_optimizer_steps
                    == 0
                )
                or not finite_gradients
            )
            if checkpoint_due and cursor.epoch == epoch:
                _save_stream_resume(
                    resume_path,
                    identity=identity,
                    d6_index_id=d6_index_id,
                    cursor=cursor,
                    epoch_order_digest=_cursor_order_digest(
                        index,
                        cursor=cursor,
                        seed=trainer.seed,
                        epochs=trainer.epochs,
                    ),
                    model=model,
                    optimizer=optimizer,
                    optimizer_steps=optimizer_steps,
                    micro_steps=micro_steps,
                    consumed_windows=consumed_windows,
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
                and time.monotonic() - run_started
                >= max_run_seconds
            ):
                paused = True
                break

            if cursor.epoch > epoch:
                break

        epoch_completed = cursor.epoch > epoch
        if epoch_completed:
            validation = evaluate_streaming_validation(
                model,
                package,
                index,
                tokenizer,
                recipe,
                device=resolved,
            )
            validation_loss = float(
                validation["mean_loss"]
            )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_checkpoint_sha256 = save_r2_checkpoint(
                    best_path,
                    model,
                    stage="dense_pretrain",
                    metadata={
                        "r2d6_index_id": d6_index_id,
                        "streaming_run_identity": identity,
                        "production_manifest_identity": (
                            manifest.identity()
                        ),
                        "production_recipe_fingerprint": (
                            recipe.fingerprint()
                        ),
                        "production_trainer": asdict(trainer),
                        "optimizer_steps": optimizer_steps,
                        "micro_steps": micro_steps,
                        "consumed_windows": consumed_windows,
                        "validation": validation,
                        "parent_checkpoint_sha256": None,
                    },
                )

        if paused and cursor.epoch >= trainer.epochs:
            # The final epoch has already been validated. Do not force a
            # no-op resume turn merely because the wall-clock budget expired
            # on the final optimizer boundary.
            paused = False

        if paused:
            _save_stream_resume(
                resume_path,
                identity=identity,
                d6_index_id=d6_index_id,
                cursor=cursor,
                epoch_order_digest=_cursor_order_digest(
                    index,
                    cursor=cursor,
                    seed=trainer.seed,
                    epochs=trainer.epochs,
                ),
                model=model,
                optimizer=optimizer,
                optimizer_steps=optimizer_steps,
                micro_steps=micro_steps,
                consumed_windows=consumed_windows,
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

        if epoch_completed and cursor.epoch < trainer.epochs:
            _save_stream_resume(
                resume_path,
                identity=identity,
                d6_index_id=d6_index_id,
                cursor=cursor,
                epoch_order_digest=_cursor_order_digest(
                    index,
                    cursor=cursor,
                    seed=trainer.seed,
                    epochs=trainer.epochs,
                ),
                model=model,
                optimizer=optimizer,
                optimizer_steps=optimizer_steps,
                micro_steps=micro_steps,
                consumed_windows=consumed_windows,
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

    if micro_steps <= 0 or consumed_windows <= 0 or target_tokens <= 0:
        raise RuntimeError("R2-D7 streaming training completed without work")

    completed = cursor.epoch >= trainer.epochs and not paused
    if completed:
        if best_epoch < 0 or not best_checkpoint_sha256:
            raise RuntimeError(
                "R2-D7 completed without selecting a checkpoint"
            )
        resume_path.unlink(missing_ok=True)
        resume_sha = ""
    else:
        if not resume_path.is_file():
            raise RuntimeError(
                "R2-D7 pause did not persist resume state"
            )
        resume_sha = sha256_file(resume_path)

    return R2StreamingTrainingResult(
        optimizer_steps=optimizer_steps,
        micro_steps=micro_steps,
        consumed_windows=consumed_windows,
        target_tokens=target_tokens,
        mean_loss=loss_sum / micro_steps,
        final_loss=final_loss,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        best_checkpoint_sha256=best_checkpoint_sha256,
        completed=completed,
        resume_checkpoint_sha256=resume_sha,
        loss_scale=loss_scale,
        skipped_nonfinite_steps=skipped_nonfinite_steps,
        next_cursor=cursor,
    )
