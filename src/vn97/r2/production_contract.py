from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable

from .config import VN97R2Config


R2_PRODUCTION_MIN_PARAMETERS = 900_000_000
R2_PRODUCTION_MAX_PARAMETERS = 1_300_000_000
R2_PRODUCTION_STAGES = (
    "dense_pretrain",
    "instruction_reasoning",
    "tool_action",
    "capability",
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def assert_r2_production_scale(
    config: VN97R2Config,
    *,
    minimum_parameters: int = R2_PRODUCTION_MIN_PARAMETERS,
    maximum_parameters: int = R2_PRODUCTION_MAX_PARAMETERS,
) -> int:
    if minimum_parameters <= 0 or maximum_parameters < minimum_parameters:
        raise ValueError("invalid production parameter bounds")
    parameters = config.estimated_parameter_count()
    if not minimum_parameters <= parameters <= maximum_parameters:
        raise ValueError(
            "R2 production configuration is outside the locked 0.9B-1.3B "
            f"class: parameters={parameters}, expected=[{minimum_parameters}, "
            f"{maximum_parameters}]"
        )
    return parameters


@dataclass(frozen=True)
class R2ProductionShard:
    split: str
    sha256: str
    bytes: int
    records: int
    task_families: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        if len(self.sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.sha256
        ):
            raise ValueError("sha256 must be a lowercase SHA-256 digest")
        if self.bytes <= 0 or self.records <= 0:
            raise ValueError("shard bytes and records must be positive")
        if any(not item for item in self.task_families):
            raise ValueError("task family names must be non-empty")


@dataclass(frozen=True)
class R2ProductionCorpusManifest:
    stage: str
    tokenizer_sha256: str
    shards: tuple[R2ProductionShard, ...]
    parent_checkpoint_sha256: str | None = None
    schema: str = "VN97R2PRODCORPUS1"

    def __post_init__(self) -> None:
        if self.stage not in R2_PRODUCTION_STAGES:
            raise ValueError("unsupported R2 production stage")
        if len(self.tokenizer_sha256) != 64:
            raise ValueError("tokenizer_sha256 must be a SHA-256 digest")
        if not self.shards:
            raise ValueError("production corpus must contain shards")
        train = [item for item in self.shards if item.split == "train"]
        validation = [
            item for item in self.shards if item.split == "validation"
        ]
        if not train or not validation:
            raise ValueError(
                "production corpus requires train and validation shards"
            )
        digests = [item.sha256 for item in self.shards]
        if len(digests) != len(set(digests)):
            raise ValueError("duplicate production corpus shard digest")
        if self.stage == "dense_pretrain":
            if self.parent_checkpoint_sha256 is not None:
                raise ValueError(
                    "dense_pretrain must start without a parent checkpoint"
                )
        else:
            if (
                self.parent_checkpoint_sha256 is None
                or len(self.parent_checkpoint_sha256) != 64
            ):
                raise ValueError(
                    "post-pretrain stages require parent checkpoint SHA-256"
                )

    def identity(self) -> str:
        payload = {
            "schema": self.schema,
            "stage": self.stage,
            "tokenizer_sha256": self.tokenizer_sha256,
            "parent_checkpoint_sha256": self.parent_checkpoint_sha256,
            "shards": [asdict(item) for item in self.shards],
        }
        return hashlib.sha256(
            b"VN97R2PRODCORPUS\0" + _canonical_json(payload)
        ).hexdigest()


@dataclass(frozen=True)
class R2ProductionTrainingRecipe:
    sequence_length: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    precision: str = "bf16"
    activation_checkpointing: bool = True
    memory_efficient_scan: bool = True
    optimizer_state_offload: bool = False
    full_parameter_training: bool = True
    quantization_used: bool = False

    def __post_init__(self) -> None:
        if self.sequence_length < 64:
            raise ValueError("production sequence_length must be at least 64")
        if self.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16 or bf16")

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    def fingerprint(self) -> str:
        return hashlib.sha256(
            b"VN97R2PRODRECIPE\0" + _canonical_json(asdict(self))
        ).hexdigest()


@dataclass(frozen=True)
class R2ProductionResourceEstimate:
    parameter_count: int
    device_weight_bytes: int
    device_gradient_bytes: int
    device_optimizer_bytes: int
    host_optimizer_bytes: int
    recurrent_state_bytes: int
    minimum_device_bytes_before_activations: int
    minimum_host_bytes_before_dataset: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def estimate_production_training_resources(
    config: VN97R2Config,
    recipe: R2ProductionTrainingRecipe,
) -> R2ProductionResourceEstimate:
    parameters = assert_r2_production_scale(config)
    value_bytes = 4 if recipe.precision == "fp32" else 2

    weight_bytes = parameters * value_bytes
    # Keep gradients in model precision for the conservative lower bound.
    gradient_bytes = parameters * value_bytes
    # AdamW moments plus FP32 master weights are 12 B/parameter.
    optimizer_bytes = parameters * 12

    if recipe.optimizer_state_offload:
        device_optimizer_bytes = 0
        host_optimizer_bytes = optimizer_bytes
    else:
        device_optimizer_bytes = optimizer_bytes
        host_optimizer_bytes = 0

    recurrent_state = config.recurrent_state_bytes(
        recipe.micro_batch_size,
        bytes_per_value=value_bytes,
    )
    minimum_device = (
        weight_bytes
        + gradient_bytes
        + device_optimizer_bytes
        + recurrent_state
    )

    return R2ProductionResourceEstimate(
        parameter_count=parameters,
        device_weight_bytes=weight_bytes,
        device_gradient_bytes=gradient_bytes,
        device_optimizer_bytes=device_optimizer_bytes,
        host_optimizer_bytes=host_optimizer_bytes,
        recurrent_state_bytes=recurrent_state,
        minimum_device_bytes_before_activations=minimum_device,
        minimum_host_bytes_before_dataset=host_optimizer_bytes,
    )


def assert_production_training_contract(
    config: VN97R2Config,
    recipe: R2ProductionTrainingRecipe,
    *,
    available_device_bytes: int | None = None,
    available_host_bytes: int | None = None,
) -> R2ProductionResourceEstimate:
    resources = estimate_production_training_resources(config, recipe)

    if not recipe.full_parameter_training:
        raise ValueError(
            "R2-D requires full-parameter dense training; adapter-only "
            "training cannot replace the canonical model"
        )
    if recipe.quantization_used:
        raise ValueError(
            "R2-D forbids QAT/INT4/ternary before fresh dense validation"
        )
    if not recipe.activation_checkpointing:
        raise ValueError(
            "R2-D production training requires activation checkpointing"
        )
    if not recipe.memory_efficient_scan:
        raise ValueError(
            "R2-D production training requires the memory-efficient "
            "selective-scan path; eager affine-scan autograd is pilot-only"
        )

    if (
        available_device_bytes is not None
        and resources.minimum_device_bytes_before_activations
        > available_device_bytes
    ):
        raise RuntimeError(
            "R2-D device memory is insufficient even before activation "
            "storage; enable optimizer offload or use a larger GPU"
        )
    if (
        available_host_bytes is not None
        and resources.minimum_host_bytes_before_dataset
        > available_host_bytes
    ):
        raise RuntimeError(
            "R2-D host memory is insufficient for optimizer offload"
        )
    return resources


def assert_stage_transition(
    *,
    stage: str,
    parent_stage: str | None,
) -> None:
    if stage not in R2_PRODUCTION_STAGES:
        raise ValueError("unsupported R2 production stage")
    index = R2_PRODUCTION_STAGES.index(stage)
    expected_parent = None if index == 0 else R2_PRODUCTION_STAGES[index - 1]
    if parent_stage != expected_parent:
        raise ValueError(
            f"invalid R2-D stage transition: stage={stage}, "
            f"parent={parent_stage}, expected_parent={expected_parent}"
        )


def corpus_task_families(
    manifest: R2ProductionCorpusManifest,
) -> tuple[str, ...]:
    families: set[str] = set()
    for shard in manifest.shards:
        families.update(shard.task_families)
    return tuple(sorted(families))
