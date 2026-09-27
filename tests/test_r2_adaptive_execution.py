from __future__ import annotations

import torch

from vn97.r2.adaptive_execution import (
    AdaptiveExecutionDecision,
    AdaptiveExecutionMode,
    AdaptiveExecutionRequest,
    AdaptiveRuntimeTelemetry,
    AndroidDeviceProfile,
    OrtExecutionProvider,
    VN97AdaptiveExecutionScheduler,
    execute_reference_schedule,
    variable_chunk_plan,
)
from vn97.r2.config import r2_smoke_config
from vn97.r2.model import VN97R2Model


def _model() -> VN97R2Model:
    torch.manual_seed(9701)
    model = VN97R2Model(r2_smoke_config(vocab_size=257))
    model.eval()
    return model


def _decision(
    mode: AdaptiveExecutionMode,
    chunks: tuple[int, ...],
) -> AdaptiveExecutionDecision:
    return AdaptiveExecutionDecision(
        mode=mode,
        chunk_sizes=chunks,
        providers=(
            OrtExecutionProvider.NNAPI,
            OrtExecutionProvider.XNNPACK,
            OrtExecutionProvider.CPU,
        ),
        xnnpack_threads=4,
        reason="test",
    )


def test_variable_chunk_plan_ramps_and_exactly_covers_sequence() -> None:
    assert variable_chunk_plan(
        100,
        maximum_chunk_size=32,
        first_chunk_size=8,
    ) == (8, 16, 32, 32, 12)
    for length in (1, 2, 7, 8, 9, 31, 32, 33, 127, 256):
        chunks = variable_chunk_plan(
            length,
            maximum_chunk_size=64,
        )
        assert sum(chunks) == length
        assert all(1 <= value <= 64 for value in chunks)


def test_s21_fe_exynos_prefers_nnapi_then_xnnpack_cpu() -> None:
    scheduler = VN97AdaptiveExecutionScheduler()
    device = AndroidDeviceProfile(
        sdk_int=36,
        logical_cores=8,
        available_memory_bytes=3 * 1024**3,
        hardware="exynos2100",
        soc_manufacturer="samsung",
        soc_model="Exynos 2100",
        nnapi_available=True,
        xnnpack_available=True,
        qnn_available=False,
    )
    decision = scheduler.decide(
        device,
        AdaptiveExecutionRequest(
            sequence_length=192,
            state_dependency=0.5,
        ),
    )
    assert decision.mode is AdaptiveExecutionMode.HYBRID
    assert decision.providers == (
        OrtExecutionProvider.NNAPI,
        OrtExecutionProvider.XNNPACK,
        OrtExecutionProvider.CPU,
    )
    assert decision.xnnpack_threads == 4
    assert sum(decision.chunk_sizes) == 192


def test_snapdragon_can_select_qnn_only_when_explicitly_available() -> None:
    scheduler = VN97AdaptiveExecutionScheduler()
    base = dict(
        sdk_int=36,
        logical_cores=8,
        available_memory_bytes=3 * 1024**3,
        hardware="qcom",
        soc_manufacturer="Qualcomm",
        soc_model="SM8350",
        nnapi_available=True,
        xnnpack_available=True,
    )
    enabled = scheduler.decide(
        AndroidDeviceProfile(
            **base,
            qnn_available=True,
        ),
        AdaptiveExecutionRequest(sequence_length=64),
    )
    assert enabled.providers[0] is OrtExecutionProvider.QNN

    disabled = scheduler.decide(
        AndroidDeviceProfile(
            **base,
            qnn_available=False,
        ),
        AdaptiveExecutionRequest(sequence_length=64),
    )
    assert OrtExecutionProvider.QNN not in disabled.providers


def test_realtime_always_uses_recurrent_sequential_path() -> None:
    scheduler = VN97AdaptiveExecutionScheduler()
    decision = scheduler.decide(
        AndroidDeviceProfile(
            sdk_int=35,
            logical_cores=8,
            available_memory_bytes=2 * 1024**3,
        ),
        AdaptiveExecutionRequest(
            sequence_length=17,
            deadline_ms=16.7,
            realtime=True,
            state_dependency=0.95,
        ),
    )
    assert decision.mode is AdaptiveExecutionMode.SEQUENTIAL
    assert decision.chunk_sizes == (1,) * 17


def test_long_low_dependency_prefill_can_use_parallel_scan() -> None:
    scheduler = VN97AdaptiveExecutionScheduler()
    decision = scheduler.decide(
        AndroidDeviceProfile(
            sdk_int=35,
            logical_cores=8,
            available_memory_bytes=2 * 1024**3,
            thermal_status=1,
        ),
        AdaptiveExecutionRequest(
            sequence_length=512,
            deadline_ms=800.0,
            state_dependency=0.2,
        ),
    )
    assert decision.mode is AdaptiveExecutionMode.PARALLEL
    assert decision.chunk_sizes == (512,)


def test_thermal_pressure_reduces_chunks_and_threads() -> None:
    scheduler = VN97AdaptiveExecutionScheduler()
    decision = scheduler.decide(
        AndroidDeviceProfile(
            sdk_int=35,
            logical_cores=8,
            available_memory_bytes=3 * 1024**3,
            thermal_status=5,
        ),
        AdaptiveExecutionRequest(
            sequence_length=96,
            state_dependency=0.5,
        ),
    )
    assert decision.mode is AdaptiveExecutionMode.HYBRID
    assert max(decision.chunk_sizes) <= 8
    assert decision.xnnpack_threads <= 2


def test_failed_nnapi_is_removed_from_provider_order() -> None:
    scheduler = VN97AdaptiveExecutionScheduler()
    decision = scheduler.decide(
        AndroidDeviceProfile(
            sdk_int=35,
            logical_cores=8,
            available_memory_bytes=2 * 1024**3,
        ),
        AdaptiveExecutionRequest(sequence_length=64),
        AdaptiveRuntimeTelemetry(
            provider_failures=(OrtExecutionProvider.NNAPI,),
        ),
    )
    assert OrtExecutionProvider.NNAPI not in decision.providers
    assert decision.providers[-1] is OrtExecutionProvider.CPU


def test_sequential_hybrid_parallel_are_numerically_equivalent() -> None:
    model = _model()
    torch.manual_seed(9702)
    input_ids = torch.randint(
        0,
        model.config.vocab_size,
        (2, 73),
        dtype=torch.long,
    )

    parallel_logits, parallel_state = execute_reference_schedule(
        model,
        input_ids,
        _decision(
            AdaptiveExecutionMode.PARALLEL,
            (73,),
        ),
    )
    hybrid_chunks = variable_chunk_plan(
        73,
        maximum_chunk_size=32,
        first_chunk_size=8,
    )
    hybrid_logits, hybrid_state = execute_reference_schedule(
        model,
        input_ids,
        _decision(
            AdaptiveExecutionMode.HYBRID,
            hybrid_chunks,
        ),
    )
    sequential_logits, sequential_state = execute_reference_schedule(
        model,
        input_ids,
        _decision(
            AdaptiveExecutionMode.SEQUENTIAL,
            (1,) * 73,
        ),
    )

    torch.testing.assert_close(
        hybrid_logits,
        parallel_logits,
        rtol=2e-5,
        atol=2e-5,
    )
    torch.testing.assert_close(
        sequential_logits,
        parallel_logits,
        rtol=3e-5,
        atol=3e-5,
    )
    assert hybrid_state.active_layers == parallel_state.active_layers
    assert sequential_state.active_layers == parallel_state.active_layers

    for reference, hybrid, sequential in zip(
        parallel_state.layers,
        hybrid_state.layers,
        sequential_state.layers,
    ):
        torch.testing.assert_close(
            hybrid.conv,
            reference.conv,
            rtol=2e-5,
            atol=2e-5,
        )
        torch.testing.assert_close(
            hybrid.ssm,
            reference.ssm,
            rtol=2e-5,
            atol=2e-5,
        )
        torch.testing.assert_close(
            sequential.conv,
            reference.conv,
            rtol=3e-5,
            atol=3e-5,
        )
        torch.testing.assert_close(
            sequential.ssm,
            reference.ssm,
            rtol=3e-5,
            atol=3e-5,
        )


def test_fast_profile_uses_same_weights_under_all_schedules() -> None:
    model = _model()
    input_ids = torch.tensor(
        [[7, 11, 13, 17, 19, 23, 29, 31, 37]],
        dtype=torch.long,
    )
    parallel, parallel_state = execute_reference_schedule(
        model,
        input_ids,
        _decision(AdaptiveExecutionMode.PARALLEL, (9,)),
        profile="fast",
    )
    hybrid, hybrid_state = execute_reference_schedule(
        model,
        input_ids,
        _decision(AdaptiveExecutionMode.HYBRID, (2, 4, 3)),
        profile="fast",
    )
    assert parallel_state.active_layers == model.config.fast_layers
    assert hybrid_state.active_layers == model.config.fast_layers
    torch.testing.assert_close(
        hybrid,
        parallel,
        rtol=2e-5,
        atol=2e-5,
    )
