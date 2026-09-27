from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable

import torch

from .model import VN97R2Model, VN97R2State


class AdaptiveExecutionMode(str, Enum):
    SEQUENTIAL = "sequential"
    HYBRID = "hybrid"
    PARALLEL = "parallel"


class OrtExecutionProvider(str, Enum):
    QNN = "qnn"
    NNAPI = "nnapi"
    XNNPACK = "xnnpack"
    CPU = "cpu"


@dataclass(frozen=True)
class AndroidDeviceProfile:
    sdk_int: int
    logical_cores: int
    available_memory_bytes: int
    thermal_status: int = 0
    hardware: str = ""
    soc_manufacturer: str = ""
    soc_model: str = ""
    nnapi_available: bool = True
    xnnpack_available: bool = True
    qnn_available: bool = False

    def __post_init__(self) -> None:
        if self.sdk_int < 26:
            raise ValueError("VN97 Android runtime requires API 26+")
        if self.logical_cores <= 0:
            raise ValueError("logical_cores must be positive")
        if self.available_memory_bytes <= 0:
            raise ValueError("available_memory_bytes must be positive")
        if not 0 <= self.thermal_status <= 6:
            raise ValueError("thermal_status must be in [0, 6]")

    @property
    def is_qualcomm(self) -> bool:
        joined = " ".join(
            (
                self.hardware,
                self.soc_manufacturer,
                self.soc_model,
            )
        ).lower()
        return any(
            marker in joined
            for marker in ("qualcomm", "snapdragon", "qcom", "sm8350")
        )


@dataclass(frozen=True)
class AdaptiveExecutionRequest:
    sequence_length: int
    batch_size: int = 1
    deadline_ms: float = 250.0
    realtime: bool = False
    state_dependency: float = 0.5
    prefer_accelerator: bool = True

    def __post_init__(self) -> None:
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.deadline_ms <= 0.0 or not math.isfinite(self.deadline_ms):
            raise ValueError("deadline_ms must be finite and positive")
        if not 0.0 <= self.state_dependency <= 1.0:
            raise ValueError("state_dependency must be in [0, 1]")


@dataclass(frozen=True)
class AdaptiveRuntimeTelemetry:
    moving_latency_ms: float | None = None
    provider_failures: tuple[OrtExecutionProvider, ...] = ()
    thermal_status: int | None = None
    available_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        if (
            self.moving_latency_ms is not None
            and (
                self.moving_latency_ms <= 0.0
                or not math.isfinite(self.moving_latency_ms)
            )
        ):
            raise ValueError(
                "moving_latency_ms must be finite and positive"
            )
        if (
            self.thermal_status is not None
            and not 0 <= self.thermal_status <= 6
        ):
            raise ValueError("thermal_status must be in [0, 6]")
        if (
            self.available_memory_bytes is not None
            and self.available_memory_bytes <= 0
        ):
            raise ValueError(
                "available_memory_bytes must be positive"
            )


@dataclass(frozen=True)
class AdaptiveExecutionDecision:
    mode: AdaptiveExecutionMode
    chunk_sizes: tuple[int, ...]
    providers: tuple[OrtExecutionProvider, ...]
    xnnpack_threads: int
    reason: str

    def __post_init__(self) -> None:
        if not self.chunk_sizes:
            raise ValueError("chunk_sizes must not be empty")
        if any(size <= 0 for size in self.chunk_sizes):
            raise ValueError("chunk_sizes must be positive")
        if not self.providers:
            raise ValueError("providers must not be empty")
        if self.providers[-1] is not OrtExecutionProvider.CPU:
            raise ValueError("CPU must remain the final fallback")
        if self.xnnpack_threads <= 0:
            raise ValueError("xnnpack_threads must be positive")


def variable_chunk_plan(
    sequence_length: int,
    *,
    maximum_chunk_size: int,
    first_chunk_size: int | None = None,
) -> tuple[int, ...]:
    """Create a deterministic ramped chunk schedule.

    Small initial chunks lower first-work latency. Subsequent chunks grow by
    powers of two up to the selected maximum. State is carried exactly between
    chunks, so changing chunk boundaries changes only execution schedule.
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if maximum_chunk_size <= 0:
        raise ValueError("maximum_chunk_size must be positive")
    first = (
        min(maximum_chunk_size, 8)
        if first_chunk_size is None
        else first_chunk_size
    )
    if first <= 0 or first > maximum_chunk_size:
        raise ValueError(
            "first_chunk_size must be in [1, maximum_chunk_size]"
        )

    remaining = sequence_length
    current = min(first, remaining)
    chunks: list[int] = []
    while remaining > 0:
        take = min(current, remaining)
        chunks.append(take)
        remaining -= take
        current = min(maximum_chunk_size, current * 2)
    return tuple(chunks)


class VN97AdaptiveExecutionScheduler:
    """Hardware-aware scheduler for one VN97 model/state contract.

    The scheduler changes only how the same recurrent/scan equations are
    executed. It never routes to another AI model.
    """

    _LOW_MEMORY = 512 * 1024 * 1024
    _COMFORTABLE_MEMORY = 1536 * 1024 * 1024

    def decide(
        self,
        device: AndroidDeviceProfile,
        request: AdaptiveExecutionRequest,
        telemetry: AdaptiveRuntimeTelemetry | None = None,
    ) -> AdaptiveExecutionDecision:
        telemetry = telemetry or AdaptiveRuntimeTelemetry()
        thermal = (
            device.thermal_status
            if telemetry.thermal_status is None
            else telemetry.thermal_status
        )
        memory = (
            device.available_memory_bytes
            if telemetry.available_memory_bytes is None
            else telemetry.available_memory_bytes
        )

        failed = set(telemetry.provider_failures)
        providers: list[OrtExecutionProvider] = []
        if request.prefer_accelerator:
            if (
                device.qnn_available
                and device.is_qualcomm
                and OrtExecutionProvider.QNN not in failed
            ):
                providers.append(OrtExecutionProvider.QNN)
            if (
                device.nnapi_available
                and device.sdk_int >= 27
                and OrtExecutionProvider.NNAPI not in failed
            ):
                providers.append(OrtExecutionProvider.NNAPI)
            if (
                device.xnnpack_available
                and OrtExecutionProvider.XNNPACK not in failed
            ):
                providers.append(OrtExecutionProvider.XNNPACK)
        elif (
            device.xnnpack_available
            and OrtExecutionProvider.XNNPACK not in failed
        ):
            providers.append(OrtExecutionProvider.XNNPACK)
        providers.append(OrtExecutionProvider.CPU)

        threads = max(1, min(4, device.logical_cores))
        if thermal >= 4:
            threads = min(threads, 2)

        latency_pressure = (
            telemetry.moving_latency_ms is not None
            and telemetry.moving_latency_ms > request.deadline_ms
        )
        strongly_recurrent = request.state_dependency >= 0.85
        short = request.sequence_length <= 4

        if request.realtime or short or strongly_recurrent:
            return AdaptiveExecutionDecision(
                mode=AdaptiveExecutionMode.SEQUENTIAL,
                chunk_sizes=(1,) * request.sequence_length,
                providers=tuple(providers),
                xnnpack_threads=threads,
                reason=(
                    "realtime_or_high_state_dependency_selects_recurrent_path"
                ),
            )

        if memory < self._LOW_MEMORY or thermal >= 5:
            maximum = 8
            reason = "memory_or_thermal_pressure_selects_small_hybrid_chunks"
        elif request.sequence_length < 128 or thermal >= 3:
            maximum = 32
            reason = "moderate_sequence_or_thermal_load_selects_hybrid"
        elif (
            memory >= self._COMFORTABLE_MEMORY
            and request.sequence_length >= 256
            and request.state_dependency <= 0.35
            and not latency_pressure
        ):
            return AdaptiveExecutionDecision(
                mode=AdaptiveExecutionMode.PARALLEL,
                chunk_sizes=(request.sequence_length,),
                providers=tuple(providers),
                xnnpack_threads=threads,
                reason="long_low_dependency_prefill_selects_parallel_scan",
            )
        else:
            maximum = 64
            reason = "default_variable_hybrid_scan"

        if latency_pressure:
            maximum = max(8, maximum // 2)
            reason += "_latency_feedback_reduces_chunk"

        return AdaptiveExecutionDecision(
            mode=AdaptiveExecutionMode.HYBRID,
            chunk_sizes=variable_chunk_plan(
                request.sequence_length,
                maximum_chunk_size=maximum,
                first_chunk_size=min(8, maximum),
            ),
            providers=tuple(providers),
            xnnpack_threads=threads,
            reason=reason,
        )


@torch.inference_mode()
def execute_reference_schedule(
    model: VN97R2Model,
    input_ids: torch.Tensor,
    decision: AdaptiveExecutionDecision,
    state: VN97R2State | None = None,
    *,
    profile: str | int | None = None,
) -> tuple[torch.Tensor, VN97R2State]:
    """Reference executor proving schedule-invariant VN97 semantics.

    This is not the Android ORT implementation. It is the canonical numerical
    oracle used by E1 tests and later ONNX export validation.
    """
    if input_ids.ndim != 2 or input_ids.shape[1] <= 0:
        raise ValueError("input_ids must be [batch, positive_sequence]")
    if sum(decision.chunk_sizes) != input_ids.shape[1]:
        raise ValueError(
            "decision chunk_sizes must exactly cover input sequence"
        )

    if decision.mode is AdaptiveExecutionMode.PARALLEL:
        return model(input_ids, state, profile=profile)

    if decision.mode is AdaptiveExecutionMode.SEQUENTIAL:
        outputs: list[torch.Tensor] = []
        current = state
        for position in range(input_ids.shape[1]):
            logits, current = model.step(
                input_ids[:, position],
                current,
                profile=profile,
            )
            outputs.append(logits)
        if current is None:
            raise AssertionError("sequential state was not produced")
        return torch.stack(outputs, dim=1), current

    outputs = []
    current = state
    start = 0
    for size in decision.chunk_sizes:
        end = start + size
        logits, current = model(
            input_ids[:, start:end],
            current,
            profile=profile,
        )
        outputs.append(logits)
        start = end
    if current is None:
        raise AssertionError("hybrid state was not produced")
    return torch.cat(outputs, dim=1), current


def provider_names(
    providers: Iterable[OrtExecutionProvider],
) -> tuple[str, ...]:
    return tuple(provider.value for provider in providers)
