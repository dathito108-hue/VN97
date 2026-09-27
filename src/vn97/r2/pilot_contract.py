from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

from ..training import VN97ChatMessage
from .config import VN97R2Config


R2_PILOT_MIN_PARAMETERS = 50_000_000
R2_PILOT_MAX_PARAMETERS = 150_000_000


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def chat_record_digest(
    messages: Sequence[VN97ChatMessage],
) -> str:
    if not messages:
        raise ValueError("chat record must not be empty")
    payload = [
        {"role": message.role, "content": message.content}
        for message in messages
    ]
    return hashlib.sha256(
        b"VN97R2CHAT1\0" + _canonical_json(payload)
    ).hexdigest()


@dataclass(frozen=True)
class R2PilotCorpusEvidence:
    training_records: int
    validation_records: int
    unique_training_records: int
    unique_validation_records: int
    overlap_records: int
    training_fingerprint: str
    validation_fingerprint: str

    def __post_init__(self) -> None:
        if self.training_records <= 0 or self.validation_records <= 0:
            raise ValueError("pilot corpus splits must be non-empty")
        if self.overlap_records < 0:
            raise ValueError("overlap_records must be non-negative")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def build_pilot_corpus_evidence(
    training_records: Sequence[Sequence[VN97ChatMessage]],
    validation_records: Sequence[Sequence[VN97ChatMessage]],
    *,
    reject_overlap: bool = True,
    reject_duplicates: bool = True,
) -> R2PilotCorpusEvidence:
    if not training_records or not validation_records:
        raise ValueError("pilot train/validation records must be non-empty")

    train_digests = tuple(
        chat_record_digest(record)
        for record in training_records
    )
    validation_digests = tuple(
        chat_record_digest(record)
        for record in validation_records
    )
    train_set = set(train_digests)
    validation_set = set(validation_digests)
    overlap = train_set & validation_set

    if reject_duplicates:
        duplicate_train = len(train_digests) - len(train_set)
        duplicate_validation = (
            len(validation_digests) - len(validation_set)
        )
        if duplicate_train or duplicate_validation:
            raise ValueError(
                "R2 pilot exact duplicate records detected: "
                f"train_duplicates={duplicate_train}, "
                f"validation_duplicates={duplicate_validation}"
            )

    if reject_overlap and overlap:
        first = sorted(overlap)[0]
        raise ValueError(
            "R2 pilot train/validation leakage detected: "
            f"{len(overlap)} overlapping record(s), first={first[:16]}"
        )

    def split_fingerprint(
        label: bytes,
        digests: Sequence[str],
    ) -> str:
        payload = _canonical_json(sorted(digests))
        return hashlib.sha256(label + b"\0" + payload).hexdigest()

    return R2PilotCorpusEvidence(
        training_records=len(training_records),
        validation_records=len(validation_records),
        unique_training_records=len(train_set),
        unique_validation_records=len(validation_set),
        overlap_records=len(overlap),
        training_fingerprint=split_fingerprint(
            b"VN97R2TRAIN",
            train_digests,
        ),
        validation_fingerprint=split_fingerprint(
            b"VN97R2VALID",
            validation_digests,
        ),
    )


def assert_r2_pilot_scale(
    config: VN97R2Config,
    *,
    minimum_parameters: int = R2_PILOT_MIN_PARAMETERS,
    maximum_parameters: int = R2_PILOT_MAX_PARAMETERS,
) -> int:
    if minimum_parameters <= 0 or maximum_parameters < minimum_parameters:
        raise ValueError("invalid pilot parameter bounds")
    parameters = config.estimated_parameter_count()
    if not minimum_parameters <= parameters <= maximum_parameters:
        raise ValueError(
            "R2 pilot configuration is outside the locked 50-150M class: "
            f"parameters={parameters}, expected=[{minimum_parameters}, "
            f"{maximum_parameters}]"
        )
    return parameters


@dataclass(frozen=True)
class R2PilotResourceEstimate:
    parameter_count: int
    optimizer_model_grad_bytes: int
    activation_bytes: int
    runtime_overhead_bytes: int
    recommended_ram_bytes: int
    available_ram_bytes: int | None

    @property
    def fits_available_ram(self) -> bool | None:
        if self.available_ram_bytes is None:
            return None
        return self.available_ram_bytes >= self.recommended_ram_bytes

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["fits_available_ram"] = self.fits_available_ram
        return value


def _linux_cgroup_available_bytes() -> int | None:
    maximum = Path("/sys/fs/cgroup/memory.max")
    current = Path("/sys/fs/cgroup/memory.current")
    try:
        raw_max = maximum.read_text(encoding="utf-8").strip()
        raw_current = current.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if raw_max == "max":
        return None
    try:
        limit = int(raw_max)
        used = int(raw_current)
    except ValueError:
        return None
    return max(limit - used, 0)


def available_memory_bytes() -> int | None:
    candidates: list[int] = []

    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if pages > 0 and page_size > 0:
            candidates.append(pages * page_size)
    except (OSError, ValueError, TypeError):
        pass

    try:
        for line in Path("/proc/meminfo").read_text(
            encoding="utf-8"
        ).splitlines():
            if line.startswith("MemAvailable:"):
                kib = int(line.split()[1])
                candidates.append(kib * 1024)
                break
    except (OSError, ValueError, IndexError):
        pass

    cgroup = _linux_cgroup_available_bytes()
    if cgroup is not None:
        candidates.append(cgroup)

    positive = [value for value in candidates if value > 0]
    return min(positive) if positive else None


def estimate_pilot_training_resources(
    config: VN97R2Config,
    *,
    sequence_length: int,
    batch_size: int,
    available_ram: int | None = None,
) -> R2PilotResourceEstimate:
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    parameter_count = config.estimated_parameter_count()

    # Dense FP32 AdamW lower-order storage:
    # model (4 B) + grad (4 B) + first/second moments (8 B).
    optimizer_model_grad = parameter_count * 16

    # The selective scan materializes sequence states during autograd. This is
    # deliberately conservative: three state-sized tensors plus projection/
    # gate intermediates across all layers. It is a preflight budget, not a
    # profiler replacement.
    state_values = (
        batch_size
        * sequence_length
        * config.n_layers
        * config.d_inner
        * config.d_state
    )
    projection_values = (
        batch_size
        * sequence_length
        * config.n_layers
        * (config.d_model + 4 * config.d_inner)
    )
    activation_bytes = (
        state_values * 4 * 3
        + projection_values * 4 * 2
    )

    runtime_overhead = max(
        512 * 1024 * 1024,
        parameter_count * 4,
    )
    raw = (
        optimizer_model_grad
        + activation_bytes
        + runtime_overhead
    )
    recommended = (raw * 5 + 3) // 4

    resolved_available = (
        available_memory_bytes()
        if available_ram is None
        else available_ram
    )
    return R2PilotResourceEstimate(
        parameter_count=parameter_count,
        optimizer_model_grad_bytes=optimizer_model_grad,
        activation_bytes=activation_bytes,
        runtime_overhead_bytes=runtime_overhead,
        recommended_ram_bytes=recommended,
        available_ram_bytes=resolved_available,
    )
