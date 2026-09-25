from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import stat
import tempfile
import time
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
_MAX_RESUME_CHECKPOINT_BYTES = 2 * 1024 * 1024 * 1024
_RESUME_SCHEMA = "VN97P3RESUME1"


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


def _resume_metadata_path(
    checkpoint_path: Path,
) -> Path:
    return checkpoint_path.with_name(
        "progress.vn97p3resume1.json"
    )


def _atomic_torch_save(
    path: Path,
    payload: object,
) -> None:
    if path.is_symlink():
        raise VN97P3TensorCacheError(
            "resume checkpoint target must not be a symlink"
        )
    parent = path.parent
    parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    if parent.is_symlink():
        raise VN97P3TensorCacheError(
            "resume checkpoint parent must not be a symlink"
        )

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=parent,
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        torch.save(
            payload,
            temp_path,
        )
        with temp_path.open("rb+") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        if path.is_symlink():
            raise VN97P3TensorCacheError(
                "resume checkpoint target became a symlink"
            )
        os.replace(
            temp_path,
            path,
        )
        dir_fd = os.open(
            parent.resolve(strict=True),
            os.O_RDONLY
            | getattr(
                os,
                "O_DIRECTORY",
                0,
            ),
        )
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        temp_path.unlink(
            missing_ok=True
        )


def _load_torch_checkpoint(
    path: Path,
) -> object:
    flags = (
        os.O_RDONLY
        | getattr(
            os,
            "O_NOFOLLOW",
            0,
        )
    )
    try:
        fd = os.open(
            path,
            flags,
        )
    except OSError as exc:
        raise VN97P3TensorCacheError(
            "resume checkpoint could not be opened safely"
        ) from exc
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(
                info.st_mode
            )
            or info.st_size <= 0
            or info.st_size
            > _MAX_RESUME_CHECKPOINT_BYTES
        ):
            raise VN97P3TensorCacheError(
                "resume checkpoint file bounds are invalid"
            )
        with os.fdopen(
            fd,
            "rb",
            closefd=True,
        ) as stream:
            fd = -1
            try:
                return torch.load(
                    stream,
                    map_location="cpu",
                    weights_only=True,
                )
            except Exception as exc:
                raise VN97P3TensorCacheError(
                    "resume checkpoint could not be decoded"
                ) from exc
    finally:
        if fd >= 0:
            os.close(fd)


def _encode_python_rng_state(
    state: tuple[
        int,
        tuple[int, ...],
        float | None,
    ],
) -> list[object]:
    return [
        int(state[0]),
        [
            int(value)
            for value in state[1]
        ],
        state[2],
    ]


def _decode_python_rng_state(
    value: object,
) -> tuple[
    int,
    tuple[int, ...],
    float | None,
]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or type(value[0]) is not int
        or not isinstance(
            value[1],
            list,
        )
        or any(
            type(item) is not int
            for item in value[1]
        )
        or (
            value[2] is not None
            and type(value[2])
            is not float
        )
    ):
        raise VN97P3TensorCacheError(
            "resume Python RNG state is invalid"
        )
    return (
        value[0],
        tuple(value[1]),
        value[2],
    )


def _write_resume_progress(
    checkpoint_path: Path,
    *,
    resume_identity: str,
    epoch_index: int,
    next_batch_index: int,
    steps: int,
    total_steps: int,
    trained_tokens: int,
    final_loss: float,
) -> None:
    percent = (
        100.0
        * steps
        / total_steps
        if total_steps > 0
        else 0.0
    )
    body = {
        "epoch_index": epoch_index,
        "final_loss": final_loss,
        "next_batch_index":
            next_batch_index,
        "percent": percent,
        "resume_identity":
            resume_identity,
        "schema":
            "VN97P3RESUMEPROGRESS1",
        "steps": steps,
        "total_steps": total_steps,
        "trained_tokens":
            trained_tokens,
    }
    _atomic_write(
        _resume_metadata_path(
            checkpoint_path
        ),
        _canonical_json(body)
        + b"\n",
    )


def _save_resume_checkpoint(
    checkpoint_path: Path,
    *,
    resume_identity: str,
    model: VN97LanguageCore,
    optimizer: torch.optim.Optimizer,
    rng: random.Random,
    epoch_index: int,
    next_batch_index: int,
    epoch_indices: list[int],
    steps: int,
    total_steps: int,
    trained_tokens: int,
    loss_sum: float,
    loss_count: int,
    final_loss: float,
    resolved_device: torch.device,
) -> None:
    payload = {
        "cuda_rng_state":
            torch.cuda.get_rng_state(
                resolved_device
            ).cpu(),
        "epoch_index":
            epoch_index,
        "epoch_indices":
            list(epoch_indices),
        "final_loss":
            final_loss,
        "loss_count":
            loss_count,
        "loss_sum":
            loss_sum,
        "model_state":
            model.state_dict(),
        "next_batch_index":
            next_batch_index,
        "optimizer_state":
            optimizer.state_dict(),
        "python_rng_state":
            _encode_python_rng_state(
                rng.getstate()
            ),
        "resume_identity":
            resume_identity,
        "schema":
            _RESUME_SCHEMA,
        "steps":
            steps,
        "torch_rng_state":
            torch.get_rng_state(),
        "trained_tokens":
            trained_tokens,
    }
    _atomic_torch_save(
        checkpoint_path,
        payload,
    )
    _write_resume_progress(
        checkpoint_path,
        resume_identity=
            resume_identity,
        epoch_index=epoch_index,
        next_batch_index=
            next_batch_index,
        steps=steps,
        total_steps=total_steps,
        trained_tokens=
            trained_tokens,
        final_loss=final_loss,
    )


def _load_resume_checkpoint(
    checkpoint_path: Path,
    *,
    expected_identity: str,
    model: VN97LanguageCore,
    optimizer: torch.optim.Optimizer,
    rng: random.Random,
    resolved_device: torch.device,
    config: VN97TrainingConfig,
    window_count: int,
    total_steps: int,
) -> dict[str, object]:
    value = _load_torch_checkpoint(
        checkpoint_path
    )
    expected_keys = {
        "cuda_rng_state",
        "epoch_index",
        "epoch_indices",
        "final_loss",
        "loss_count",
        "loss_sum",
        "model_state",
        "next_batch_index",
        "optimizer_state",
        "python_rng_state",
        "resume_identity",
        "schema",
        "steps",
        "torch_rng_state",
        "trained_tokens",
    }
    if (
        not isinstance(value, dict)
        or set(value)
        != expected_keys
        or value.get("schema")
        != _RESUME_SCHEMA
        or value.get(
            "resume_identity"
        )
        != expected_identity
    ):
        raise VN97P3TensorCacheError(
            "resume checkpoint identity/schema mismatch"
        )

    epoch_index = value[
        "epoch_index"
    ]
    next_batch_index = value[
        "next_batch_index"
    ]
    steps = value["steps"]
    trained_tokens = value[
        "trained_tokens"
    ]
    loss_count = value[
        "loss_count"
    ]
    loss_sum = value[
        "loss_sum"
    ]
    final_loss = value[
        "final_loss"
    ]
    epoch_indices = value[
        "epoch_indices"
    ]

    if (
        type(epoch_index) is not int
        or not 0
        <= epoch_index
        <= config.epochs
        or type(next_batch_index)
        is not int
        or next_batch_index < 0
        or type(steps) is not int
        or not 0 <= steps
        <= total_steps
        or type(trained_tokens)
        is not int
        or trained_tokens < 0
        or type(loss_count)
        is not int
        or loss_count != steps
        or not isinstance(
            loss_sum,
            (int, float),
        )
        or not math.isfinite(
            float(loss_sum)
        )
        or not isinstance(
            final_loss,
            (int, float),
        )
        or (
            steps > 0
            and not math.isfinite(
                float(final_loss)
            )
        )
        or not isinstance(
            epoch_indices,
            list,
        )
        or any(
            type(item) is not int
            or not 0
            <= item
            < window_count
            for item
            in epoch_indices
        )
    ):
        raise VN97P3TensorCacheError(
            "resume checkpoint counters are invalid"
        )

    steps_per_epoch = math.ceil(
        window_count
        / config.batch_size
    )
    if (
        epoch_index
        == config.epochs
        and (
            next_batch_index != 0
            or epoch_indices
        )
    ):
        raise VN97P3TensorCacheError(
            "completed resume checkpoint has invalid epoch state"
        )
    if epoch_index < config.epochs:
        if epoch_indices:
            if (
                len(epoch_indices)
                != window_count
                or next_batch_index
                > steps_per_epoch
            ):
                raise VN97P3TensorCacheError(
                    "active resume checkpoint has invalid epoch indices"
                )
            expected_steps = (
                epoch_index
                * steps_per_epoch
                + next_batch_index
            )
        else:
            if next_batch_index != 0:
                raise VN97P3TensorCacheError(
                    "epoch-boundary resume checkpoint has invalid batch index"
                )
            expected_steps = (
                epoch_index
                * steps_per_epoch
            )
        if steps != expected_steps:
            raise VN97P3TensorCacheError(
                "resume checkpoint step/epoch position mismatch"
            )

    model_state = value[
        "model_state"
    ]
    optimizer_state = value[
        "optimizer_state"
    ]
    if (
        not isinstance(
            model_state,
            dict,
        )
        or not isinstance(
            optimizer_state,
            dict,
        )
        or not isinstance(
            value[
                "torch_rng_state"
            ],
            torch.Tensor,
        )
        or not isinstance(
            value[
                "cuda_rng_state"
            ],
            torch.Tensor,
        )
    ):
        raise VN97P3TensorCacheError(
            "resume checkpoint tensor state is invalid"
        )

    model.load_state_dict(
        model_state,
        strict=True,
    )
    optimizer.load_state_dict(
        optimizer_state
    )
    for state in optimizer.state.values():
        for key, item in list(
            state.items()
        ):
            if isinstance(
                item,
                torch.Tensor,
            ):
                state[key] = item.to(
                    resolved_device
                )
    rng.setstate(
        _decode_python_rng_state(
            value[
                "python_rng_state"
            ]
        )
    )
    torch.set_rng_state(
        value[
            "torch_rng_state"
        ].cpu()
    )
    torch.cuda.set_rng_state(
        value[
            "cuda_rng_state"
        ].cpu(),
        resolved_device,
    )

    return {
        "epoch_index":
            epoch_index,
        "epoch_indices":
            epoch_indices,
        "final_loss":
            float(final_loss),
        "loss_count":
            loss_count,
        "loss_sum":
            float(loss_sum),
        "next_batch_index":
            next_batch_index,
        "steps":
            steps,
        "trained_tokens":
            trained_tokens,
    }


def train_vn97_from_tensors(
    model: VN97LanguageCore,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    config: VN97TrainingConfig,
    *,
    device: str | torch.device,
    cpu_prefetch_workers: int = 1,
    micro_batch_size: int | None = None,
    progress_label: str = "candidate",
    progress_interval_steps: int = 50,
    resume_checkpoint_path: Path | None = None,
    resume_identity: str | None = None,
    checkpoint_interval_steps: int = 250,
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
    if progress_interval_steps <= 0:
        raise VN97P3TensorCacheError(
            "progress_interval_steps must be positive"
        )
    if checkpoint_interval_steps <= 0:
        raise VN97P3TensorCacheError(
            "checkpoint_interval_steps must be positive"
        )
    if (
        resume_checkpoint_path
        is None
    ) != (
        resume_identity is None
    ):
        raise VN97P3TensorCacheError(
            "resume checkpoint path and identity must be provided together"
        )
    if (
        resume_identity is not None
        and (
            len(resume_identity)
            != 64
            or any(
                ch
                not in "0123456789abcdef"
                for ch
                in resume_identity
            )
        )
    ):
        raise VN97P3TensorCacheError(
            "resume identity must be lowercase SHA-256"
        )
    if not progress_label:
        raise VN97P3TensorCacheError(
            "progress_label must not be empty"
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
    trained_tokens = 0
    steps = 0
    loss_sum = 0.0
    loss_count = 0
    final_loss = math.nan
    window_count = int(input_ids.shape[0])
    steps_per_epoch = math.ceil(
        window_count / config.batch_size
    )
    total_steps = (
        steps_per_epoch * config.epochs
    )
    resumed_steps = 0
    resume_epoch_index = 0
    resume_next_batch_index = 0
    resume_epoch_indices: list[int] = []

    if (
        resume_checkpoint_path
        is not None
        and resume_checkpoint_path.exists()
    ):
        state = _load_resume_checkpoint(
            resume_checkpoint_path,
            expected_identity=
                str(resume_identity),
            model=model,
            optimizer=optimizer,
            rng=rng,
            resolved_device=resolved,
            config=config,
            window_count=window_count,
            total_steps=total_steps,
        )
        resume_epoch_index = int(
            state["epoch_index"]
        )
        resume_next_batch_index = int(
            state[
                "next_batch_index"
            ]
        )
        resume_epoch_indices = list(
            state["epoch_indices"]
        )
        steps = int(
            state["steps"]
        )
        trained_tokens = int(
            state[
                "trained_tokens"
            ]
        )
        loss_sum = float(
            state["loss_sum"]
        )
        loss_count = int(
            state["loss_count"]
        )
        final_loss = float(
            state["final_loss"]
        )
        resumed_steps = steps
        print(
            "VN97 P3 RESUME "
            f"{progress_label} "
            f"step={steps}/{total_steps} "
            f"percent={100.0 * steps / total_steps:.2f}",
            flush=True,
        )

    progress_started = time.monotonic()

    if resume_epoch_index == config.epochs:
        if (
            steps != total_steps
            or loss_count <= 0
            or trained_tokens <= 0
            or not math.isfinite(
                final_loss
            )
        ):
            raise VN97P3TensorCacheError(
                "completed resume checkpoint has incomplete training counters"
            )
        return VN97TrainingResult(
            steps=steps,
            target_tokens=
                trained_tokens,
            mean_loss=(
                loss_sum
                / loss_count
            ),
            final_loss=final_loss,
        )

    for epoch_index in range(
        resume_epoch_index,
        config.epochs,
    ):
        if (
            epoch_index
            == resume_epoch_index
            and resume_epoch_indices
        ):
            indices = (
                resume_epoch_indices
            )
            batch_index = (
                resume_next_batch_index
            )
        else:
            indices = list(
                range(window_count)
            )
            if config.shuffle:
                rng.shuffle(indices)
            batch_index = 0

        remaining_indices = indices[
            batch_index
            * config.batch_size:
        ]

        for (
            cpu_inputs,
            cpu_labels,
        ) in _iter_prefetched_cpu_batches(
            input_ids,
            labels,
            indices=remaining_indices,
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
            loss_sum += value
            loss_count += 1
            final_loss = value
            trained_tokens += logical_targets
            steps += 1
            batch_index += 1

            if (
                steps == 1
                or steps % progress_interval_steps == 0
                or steps == total_steps
            ):
                elapsed = max(
                    time.monotonic()
                    - progress_started,
                    1e-9,
                )
                session_steps = max(
                    steps
                    - resumed_steps,
                    1,
                )
                steps_per_second = (
                    session_steps
                    / elapsed
                )
                remaining_steps = max(
                    total_steps - steps,
                    0,
                )
                eta_seconds = (
                    remaining_steps
                    / steps_per_second
                    if steps_per_second > 0.0
                    else math.inf
                )
                percent = (
                    100.0
                    * steps
                    / total_steps
                )
                running_mean_loss = (
                    loss_sum
                    / loss_count
                )
                print(
                    "VN97 P3 PROGRESS "
                    f"{progress_label} "
                    f"epoch={epoch_index + 1}/{config.epochs} "
                    f"step={steps}/{total_steps} "
                    f"percent={percent:.2f} "
                    f"elapsed_s={elapsed:.1f} "
                    f"eta_s={eta_seconds:.1f} "
                    f"loss={value:.6f} "
                    f"mean_loss={running_mean_loss:.6f}",
                    flush=True,
                )

            if (
                resume_checkpoint_path
                is not None
                and (
                    steps
                    % checkpoint_interval_steps
                    == 0
                )
            ):
                _save_resume_checkpoint(
                    resume_checkpoint_path,
                    resume_identity=
                        str(
                            resume_identity
                        ),
                    model=model,
                    optimizer=optimizer,
                    rng=rng,
                    epoch_index=
                        epoch_index,
                    next_batch_index=
                        batch_index,
                    epoch_indices=
                        indices,
                    steps=steps,
                    total_steps=
                        total_steps,
                    trained_tokens=
                        trained_tokens,
                    loss_sum=
                        loss_sum,
                    loss_count=
                        loss_count,
                    final_loss=
                        final_loss,
                    resolved_device=
                        resolved,
                )
                print(
                    "VN97 P3 CHECKPOINT "
                    f"{progress_label} "
                    f"step={steps}/{total_steps} "
                    f"path={resume_checkpoint_path}",
                    flush=True,
                )

        if (
            resume_checkpoint_path
            is not None
        ):
            _save_resume_checkpoint(
                resume_checkpoint_path,
                resume_identity=
                    str(resume_identity),
                model=model,
                optimizer=optimizer,
                rng=rng,
                epoch_index=
                    epoch_index + 1,
                next_batch_index=0,
                epoch_indices=[],
                steps=steps,
                total_steps=
                    total_steps,
                trained_tokens=
                    trained_tokens,
                loss_sum=loss_sum,
                loss_count=
                    loss_count,
                final_loss=
                    final_loss,
                resolved_device=
                    resolved,
            )

        resume_epoch_indices = []
        resume_next_batch_index = 0

    model.eval()
    if (
        loss_count <= 0
        or steps <= 0
        or trained_tokens <= 0
        or not math.isfinite(
            final_loss
        )
    ):
        raise RuntimeError(
            "VN97 tensor training completed without supervised updates"
        )
    if steps != total_steps:
        raise RuntimeError(
            "VN97 tensor training ended before all logical steps completed"
        )
    return VN97TrainingResult(
        steps=steps,
        target_tokens=trained_tokens,
        mean_loss=(
            loss_sum
            / loss_count
        ),
        final_loss=final_loss,
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
