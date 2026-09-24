from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import io
import json
import math
from pathlib import Path
import random
from typing import Any, Iterator

import torch
import torch.nn.functional as F

from .evaluation import VN97EvaluationResult
from .model import VN97LanguageCore
from .training import (
    IGNORE_INDEX,
    VN97TrainingConfig,
    VN97TrainingResult,
)
from .training_cli import (
    _atomic_write,
    _read_bounded_regular_file,
)


class VN97P3TensorCacheError(RuntimeError):
    pass


_REQUIRED_CACHE_FILES = {
    "cache.vn97p3cache1.json",
    "tokenizer.vn97tk1",
    "training-windows.pt",
    "validation-windows.pt",
}
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_TENSOR_FILE_BYTES = 512 * 1024 * 1024


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(
    path: Path,
    *,
    max_bytes: int,
) -> tuple[str, int]:
    data = _read_bounded_regular_file(
        path,
        max_bytes=max_bytes,
    )
    return _sha256_bytes(data), len(data)


def tensor_bundle_bytes(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> bytes:
    if (
        input_ids.ndim != 2
        or labels.ndim != 2
        or input_ids.shape != labels.shape
        or input_ids.numel() <= 0
    ):
        raise VN97P3TensorCacheError(
            "cached input/label tensors must be non-empty matching matrices"
        )
    if input_ids.device.type != "cpu" or labels.device.type != "cpu":
        raise VN97P3TensorCacheError(
            "cached tensors must be CPU tensors"
        )
    if input_ids.dtype != torch.int32 or labels.dtype != torch.int32:
        raise VN97P3TensorCacheError(
            "cached tensors must use int32"
        )
    buffer = io.BytesIO()
    torch.save(
        {
            "input_ids": input_ids.contiguous(),
            "labels": labels.contiguous(),
        },
        buffer,
    )
    return buffer.getvalue()


def load_tensor_bundle(
    path: Path,
    *,
    expected_sha256: str,
    expected_windows: int,
    expected_sequence_length: int,
    expected_target_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    actual_sha256, _ = _sha256_file(
        path,
        max_bytes=_MAX_TENSOR_FILE_BYTES,
    )
    if actual_sha256 != expected_sha256:
        raise VN97P3TensorCacheError(
            f"tensor cache SHA-256 mismatch: {path.name}"
        )
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise VN97P3TensorCacheError(
            f"could not load tensor cache: {path.name}"
        ) from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"input_ids", "labels"}
        or not isinstance(payload["input_ids"], torch.Tensor)
        or not isinstance(payload["labels"], torch.Tensor)
    ):
        raise VN97P3TensorCacheError(
            "tensor cache payload fields are invalid"
        )
    input_ids = payload["input_ids"]
    labels = payload["labels"]
    if (
        input_ids.dtype != torch.int32
        or labels.dtype != torch.int32
        or input_ids.device.type != "cpu"
        or labels.device.type != "cpu"
        or input_ids.ndim != 2
        or labels.ndim != 2
        or input_ids.shape != labels.shape
        or tuple(input_ids.shape)
        != (
            expected_windows,
            expected_sequence_length,
        )
    ):
        raise VN97P3TensorCacheError(
            "tensor cache shape/dtype is invalid"
        )
    target_tokens = int(
        (labels != IGNORE_INDEX).sum().item()
    )
    if target_tokens != expected_target_tokens:
        raise VN97P3TensorCacheError(
            "tensor cache target-token count mismatch"
        )
    return input_ids, labels


def load_cache_manifest(
    cache_dir: Path,
) -> dict[str, Any]:
    if cache_dir.is_symlink():
        raise VN97P3TensorCacheError(
            "P3 cache directory must not be a symlink"
        )
    root = cache_dir.resolve(strict=True)
    if not root.is_dir():
        raise VN97P3TensorCacheError(
            "P3 cache path must be a directory"
        )
    if {item.name for item in root.iterdir()} != _REQUIRED_CACHE_FILES:
        raise VN97P3TensorCacheError(
            "P3 cache directory has unexpected file set"
        )
    data = _read_bounded_regular_file(
        root / "cache.vn97p3cache1.json",
        max_bytes=_MAX_MANIFEST_BYTES,
    )
    duplicates: list[str] = []

    def hook(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                duplicates.append(key)
            out[key] = value
        return out

    try:
        text = data.decode("utf-8", errors="strict")
        body = text[:-1] if text.endswith("\n") else text
        value = json.loads(
            body,
            object_pairs_hook=hook,
            parse_constant=lambda raw: (
                _ for _ in ()
            ).throw(ValueError(raw)),
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise VN97P3TensorCacheError(
            "VN97P3CACHE1 must be strict UTF-8 JSON"
        ) from exc
    if duplicates or not isinstance(value, dict):
        raise VN97P3TensorCacheError(
            "VN97P3CACHE1 must be one object without duplicate keys"
        )
    if value.get("schema") != "VN97P3CACHE1":
        raise VN97P3TensorCacheError(
            "P3 cache schema mismatch"
        )
    expected_text = (
        _canonical_json(value).decode("utf-8")
        + ("\n" if text.endswith("\n") else "")
    )
    if expected_text != text:
        raise VN97P3TensorCacheError(
            "VN97P3CACHE1 must use canonical JSON"
        )
    result_id = value.get("cache_id")
    if (
        not isinstance(result_id, str)
        or len(result_id) != 64
        or any(ch not in "0123456789abcdef" for ch in result_id)
    ):
        raise VN97P3TensorCacheError(
            "P3 cache identity is invalid"
        )
    identity = dict(value)
    identity.pop("cache_id", None)
    expected_id = _sha256_bytes(
        b"VN97P3CACHE1\0"
        + _canonical_json(identity)
    )
    if result_id != expected_id:
        raise VN97P3TensorCacheError(
            "P3 cache identity mismatch"
        )
    return value


def _prepare_cpu_batch(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    batch_indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    index_tensor = torch.tensor(
        batch_indices,
        dtype=torch.long,
    )
    inputs = input_ids.index_select(
        0,
        index_tensor,
    ).to(
        dtype=torch.long,
    )
    batch_labels = labels.index_select(
        0,
        index_tensor,
    ).to(
        dtype=torch.long,
    )
    if torch.cuda.is_available():
        inputs = inputs.pin_memory()
        batch_labels = batch_labels.pin_memory()
    return inputs, batch_labels


def _iter_prefetched_cpu_batches(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    indices: list[int],
    batch_size: int,
    workers: int,
) -> Iterator[
    tuple[torch.Tensor, torch.Tensor]
]:
    if batch_size <= 0:
        raise VN97P3TensorCacheError(
            "prefetch batch size must be positive"
        )
    if workers < 0 or workers > 8:
        raise VN97P3TensorCacheError(
            "CPU prefetch workers must be in [0, 8]"
        )

    batches = [
        indices[start : start + batch_size]
        for start in range(
            0,
            len(indices),
            batch_size,
        )
    ]
    if workers == 0:
        for batch_indices in batches:
            yield _prepare_cpu_batch(
                input_ids,
                labels,
                batch_indices,
            )
        return

    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="vn97-p3-prefetch",
    ) as executor:
        pending: deque[
            Future[
                tuple[
                    torch.Tensor,
                    torch.Tensor,
                ]
            ]
        ] = deque()
        next_batch = 0
        queue_depth = workers + 1

        while (
            next_batch < len(batches)
            and len(pending) < queue_depth
        ):
            pending.append(
                executor.submit(
                    _prepare_cpu_batch,
                    input_ids,
                    labels,
                    batches[next_batch],
                )
            )
            next_batch += 1

        while pending:
            future = pending.popleft()
            yield future.result()

            if next_batch < len(batches):
                pending.append(
                    executor.submit(
                        _prepare_cpu_batch,
                        input_ids,
                        labels,
                        batches[next_batch],
                    )
                )
                next_batch += 1


def train_vn97_from_tensors(
    model: VN97LanguageCore,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    config: VN97TrainingConfig,
    *,
    device: str | torch.device,
    cpu_prefetch_workers: int = 1,
    micro_batch_size: int | None = None,
) -> VN97TrainingResult:
    if not isinstance(model, VN97LanguageCore):
        raise TypeError("model must be VN97LanguageCore")
    if (
        input_ids.ndim != 2
        or labels.ndim != 2
        or input_ids.shape != labels.shape
        or input_ids.shape[1] != config.sequence_length
        or input_ids.shape[0] <= 0
    ):
        raise VN97P3TensorCacheError(
            "training tensor cache shape is invalid"
        )
    if micro_batch_size is None:
        micro_batch_size = config.batch_size
    if (
        micro_batch_size <= 0
        or micro_batch_size > config.batch_size
        or config.batch_size % micro_batch_size != 0
    ):
        raise VN97P3TensorCacheError(
            "micro_batch_size must be a positive divisor of logical batch_size"
        )

    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise VN97P3TensorCacheError(
            "P3 tensor training requires CUDA"
        )

    torch.cuda.set_device(
        resolved.index
        if resolved.index is not None
        else torch.cuda.current_device()
    )
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    model.to(resolved)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    rng = random.Random(config.seed)
    losses: list[float] = []
    trained_tokens = 0
    steps = 0
    window_count = int(input_ids.shape[0])

    for _epoch in range(config.epochs):
        indices = list(range(window_count))
        if config.shuffle:
            rng.shuffle(indices)

        for (
            cpu_inputs,
            cpu_labels,
        ) in _iter_prefetched_cpu_batches(
            input_ids,
            labels,
            indices=indices,
            batch_size=config.batch_size,
            workers=cpu_prefetch_workers,
        ):
            optimizer.zero_grad(
                set_to_none=True
            )

            logical_targets = int(
                (
                    cpu_labels
                    != IGNORE_INDEX
                ).sum().item()
            )
            if logical_targets <= 0:
                raise RuntimeError(
                    "logical training batch contains no supervised targets"
                )

            logical_loss_sum = 0.0
            logical_trained_tokens = 0

            for micro_start in range(
                0,
                int(cpu_inputs.shape[0]),
                micro_batch_size,
            ):
                micro_end = min(
                    micro_start + micro_batch_size,
                    int(cpu_inputs.shape[0]),
                )
                inputs = cpu_inputs[
                    micro_start:micro_end
                ].to(
                    device=resolved,
                    non_blocking=True,
                )
                batch_labels = cpu_labels[
                    micro_start:micro_end
                ].to(
                    device=resolved,
                    non_blocking=True,
                )

                logits, _ = model(inputs)
                if (
                    logits.ndim != 3
                    or logits.shape[:2]
                    != inputs.shape
                    or logits.shape[-1]
                    != model.config.vocab_size
                ):
                    raise RuntimeError(
                        "VN97 model returned invalid training logits shape"
                    )

                micro_loss_sum = F.cross_entropy(
                    logits.reshape(
                        -1,
                        logits.shape[-1],
                    ),
                    batch_labels.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction="sum",
                )
                if not bool(
                    torch.isfinite(
                        micro_loss_sum
                    )
                ):
                    raise RuntimeError(
                        "VN97 training loss became non-finite"
                    )

                micro_targets = int(
                    (
                        batch_labels
                        != IGNORE_INDEX
                    ).sum().item()
                )
                if micro_targets <= 0:
                    raise RuntimeError(
                        "micro batch contains no supervised targets"
                    )

                (
                    micro_loss_sum
                    / logical_targets
                ).backward()

                logical_loss_sum += float(
                    micro_loss_sum.detach().cpu()
                )
                logical_trained_tokens += (
                    micro_targets
                )

                del inputs
                del batch_labels
                del logits
                del micro_loss_sum

            if (
                logical_trained_tokens
                != logical_targets
            ):
                raise RuntimeError(
                    "micro-batch target count changed logical batch semantics"
                )

            grad_norm = (
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.max_grad_norm,
                )
            )
            if not bool(
                torch.isfinite(
                    torch.as_tensor(
                        grad_norm
                    )
                )
            ):
                raise RuntimeError(
                    "VN97 training gradient norm became non-finite"
                )
            optimizer.step()

            value = (
                logical_loss_sum
                / logical_targets
            )
            losses.append(value)
            trained_tokens += logical_targets
            steps += 1

    model.eval()
    if (
        not losses
        or steps <= 0
        or trained_tokens <= 0
    ):
        raise RuntimeError(
            "VN97 tensor training completed without supervised updates"
        )
    return VN97TrainingResult(
        steps=steps,
        target_tokens=trained_tokens,
        mean_loss=sum(losses)
        / len(losses),
        final_loss=losses[-1],
    )


def evaluate_vn97_from_tensors(
    model: VN97LanguageCore,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    device: str | torch.device,
    cpu_prefetch_workers: int = 1,
    micro_batch_size: int | None = None,
) -> VN97EvaluationResult:
    if (
        input_ids.ndim != 2
        or labels.ndim != 2
        or input_ids.shape != labels.shape
        or input_ids.shape[0] <= 0
        or batch_size <= 0
    ):
        raise VN97P3TensorCacheError(
            "evaluation tensor cache shape is invalid"
        )
    if micro_batch_size is None:
        micro_batch_size = batch_size
    if (
        micro_batch_size <= 0
        or micro_batch_size > batch_size
        or batch_size % micro_batch_size != 0
    ):
        raise VN97P3TensorCacheError(
            "evaluation micro_batch_size must be a positive divisor of batch_size"
        )

    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise VN97P3TensorCacheError(
            "P3 tensor evaluation requires CUDA"
        )

    model.to(resolved)
    model.eval()
    total_loss = 0.0
    total_targets = 0
    total_correct = 0
    window_count = int(
        input_ids.shape[0]
    )

    with torch.no_grad():
        indices = list(
            range(window_count)
        )
        for (
            cpu_inputs,
            cpu_labels,
        ) in _iter_prefetched_cpu_batches(
            input_ids,
            labels,
            indices=indices,
            batch_size=batch_size,
            workers=cpu_prefetch_workers,
        ):
            for micro_start in range(
                0,
                int(cpu_inputs.shape[0]),
                micro_batch_size,
            ):
                micro_end = min(
                    micro_start + micro_batch_size,
                    int(cpu_inputs.shape[0]),
                )
                inputs = cpu_inputs[
                    micro_start:micro_end
                ].to(
                    device=resolved,
                    non_blocking=True,
                )
                batch_labels = cpu_labels[
                    micro_start:micro_end
                ].to(
                    device=resolved,
                    non_blocking=True,
                )
                logits, _ = model(inputs)
                if (
                    logits.ndim != 3
                    or logits.shape[:2]
                    != inputs.shape
                    or logits.shape[-1]
                    != model.config.vocab_size
                ):
                    raise RuntimeError(
                        "VN97 model returned invalid evaluation logits shape"
                    )
                active = (
                    batch_labels
                    != IGNORE_INDEX
                )
                target_count = int(
                    active.sum().item()
                )
                if target_count <= 0:
                    raise RuntimeError(
                        "evaluation micro batch contains no supervised targets"
                    )
                loss = F.cross_entropy(
                    logits.reshape(
                        -1,
                        logits.shape[-1],
                    ),
                    batch_labels.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction="sum",
                )
                if not bool(
                    torch.isfinite(loss)
                ):
                    raise RuntimeError(
                        "VN97 validation loss became non-finite"
                    )
                predictions = (
                    logits.argmax(dim=-1)
                )
                correct = int(
                    (
                        (
                            predictions
                            == batch_labels
                        )
                        & active
                    ).sum().item()
                )
                total_loss += float(
                    loss.detach().cpu()
                )
                total_targets += target_count
                total_correct += correct

                del inputs
                del batch_labels
                del logits
                del loss

    return VN97EvaluationResult(
        windows=window_count,
        target_tokens=total_targets,
        mean_loss=(
            total_loss
            / total_targets
        ),
        top1_accuracy=(
            total_correct
            / total_targets
        ),
    )
