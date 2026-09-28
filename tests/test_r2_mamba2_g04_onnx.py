from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vn97.r2.mamba2_onnx import (
    Mamba2OnnxConfig,
    VN97Mamba2StepOnnx,
    export_mamba2_step_onnx,
    validate_ort_step_parity,
    verify_mamba2_g04_bundle,
)
from vn97.r2.mamba2_ssd_reference import (
    Mamba2LayerReferenceState,
    Mamba2ReferenceConfig,
    mamba2_model_step_ref,
)


def _configs() -> tuple[Mamba2OnnxConfig, Mamba2ReferenceConfig]:
    onnx = Mamba2OnnxConfig(
        d_model=8,
        n_layers=2,
        vocab_size=31,
        d_state=4,
        d_conv=3,
        expand=2,
        head_dim=4,
        n_groups=1,
    )
    ref = Mamba2ReferenceConfig(
        d_model=8,
        d_state=4,
        d_conv=3,
        expand=2,
        head_dim=4,
        n_groups=1,
    )
    return onnx, ref


def _source_state(
    config: Mamba2OnnxConfig,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(9740)
    state: dict[str, torch.Tensor] = {
        "backbone.embedding.weight": torch.randn(
            config.vocab_size,
            config.d_model,
            generator=generator,
        ) * 0.05,
        "backbone.norm_f.weight": (
            torch.randn(
                config.d_model,
                generator=generator,
            ) * 0.03
            + 1.0
        ),
    }
    for index in range(config.n_layers):
        prefix = f"backbone.layers.{index}"
        mixer = f"{prefix}.mixer"
        in_proj_dim = (
            2 * config.d_inner
            + 2 * config.n_groups * config.d_state
            + config.n_heads
        )
        state[f"{prefix}.norm.weight"] = (
            torch.randn(
                config.d_model,
                generator=generator,
            ) * 0.03
            + 1.0
        )
        state[f"{mixer}.in_proj.weight"] = (
            torch.randn(
                in_proj_dim,
                config.d_model,
                generator=generator,
            ) * 0.04
        )
        state[f"{mixer}.conv1d.weight"] = (
            torch.randn(
                config.conv_dim,
                1,
                config.d_conv,
                generator=generator,
            ) * 0.04
        )
        state[f"{mixer}.conv1d.bias"] = (
            torch.randn(
                config.conv_dim,
                generator=generator,
            ) * 0.01
        )
        state[f"{mixer}.dt_bias"] = (
            torch.randn(
                config.n_heads,
                generator=generator,
            ) * 0.03
        )
        state[f"{mixer}.A_log"] = (
            torch.randn(
                config.n_heads,
                generator=generator,
            ) * 0.03
        )
        state[f"{mixer}.D"] = (
            torch.randn(
                config.n_heads,
                generator=generator,
            ) * 0.03
        )
        state[f"{mixer}.norm.weight"] = (
            torch.randn(
                config.d_inner,
                generator=generator,
            ) * 0.03
            + 1.0
        )
        state[f"{mixer}.out_proj.weight"] = (
            torch.randn(
                config.d_model,
                config.d_inner,
                generator=generator,
            ) * 0.04
        )
    return state


def _logical_state(
    source: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    logical: dict[str, torch.Tensor] = {
        "vn97.core.embedding.weight":
            source["backbone.embedding.weight"],
        "vn97.core.final_norm.weight":
            source["backbone.norm_f.weight"],
    }
    for name, value in source.items():
        if name.startswith("backbone.layers."):
            logical[
                "vn97.core." + name.removeprefix("backbone.")
            ] = value
    return logical


def _reference_states(
    config: Mamba2OnnxConfig,
    conv: torch.Tensor,
    ssm: torch.Tensor,
) -> tuple[Mamba2LayerReferenceState, ...]:
    return tuple(
        Mamba2LayerReferenceState(
            conv=conv[index],
            ssm=ssm[index],
        )
        for index in range(config.n_layers)
    )


def test_explicit_state_step_matches_native_reference() -> None:
    config, ref_config = _configs()
    source = _source_state(config)
    model = VN97Mamba2StepOnnx(
        config,
        _logical_state(source),
    ).eval()

    conv, ssm = model.initial_state(2)
    generator = torch.Generator().manual_seed(9741)
    conv = torch.randn(
        conv.shape,
        generator=generator,
    ) * 0.01
    ssm = torch.randn(
        ssm.shape,
        generator=generator,
    ) * 0.01
    tokens = torch.tensor([3, 7], dtype=torch.long)

    logits, next_conv, next_ssm = model(
        tokens,
        conv,
        ssm,
    )
    ref_logits, ref_states = mamba2_model_step_ref(
        tokens,
        _reference_states(config, conv, ssm),
        source,
        config=ref_config,
        n_layers=config.n_layers,
    )
    ref_conv = torch.stack(
        [state.conv for state in ref_states],
        dim=0,
    )
    ref_ssm = torch.stack(
        [state.ssm for state in ref_states],
        dim=0,
    )

    torch.testing.assert_close(
        logits,
        ref_logits,
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(
        next_conv,
        ref_conv,
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(
        next_ssm,
        ref_ssm,
        atol=1.0e-6,
        rtol=1.0e-6,
    )



def test_explicit_state_step_fp16_matches_native_reference() -> None:
    config, ref_config = _configs()
    source = {
        name: value.half()
        for name, value in _source_state(config).items()
    }
    model = VN97Mamba2StepOnnx(
        config,
        _logical_state(source),
    ).eval()

    conv, ssm = model.initial_state(1)
    generator = torch.Generator().manual_seed(9743)
    conv = (
        torch.randn(conv.shape, generator=generator) * 0.01
    ).half()
    ssm = (
        torch.randn(ssm.shape, generator=generator) * 0.01
    ).half()
    tokens = torch.tensor([5], dtype=torch.long)

    logits, next_conv, next_ssm = model(tokens, conv, ssm)
    ref_logits, ref_states = mamba2_model_step_ref(
        tokens,
        _reference_states(config, conv, ssm),
        source,
        config=ref_config,
        n_layers=config.n_layers,
    )
    ref_conv = torch.stack([state.conv for state in ref_states], dim=0)
    ref_ssm = torch.stack([state.ssm for state in ref_states], dim=0)

    torch.testing.assert_close(
        logits.float(),
        ref_logits.float(),
        atol=2.0e-3,
        rtol=2.0e-3,
    )
    torch.testing.assert_close(
        next_conv.float(),
        ref_conv.float(),
        atol=2.0e-3,
        rtol=2.0e-3,
    )
    torch.testing.assert_close(
        next_ssm.float(),
        ref_ssm.float(),
        atol=2.0e-3,
        rtol=2.0e-3,
    )


def _onnx_runtime() -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")


def test_tiny_external_data_step_export_matches_ort(
    tmp_path: Path,
) -> None:
    _onnx_runtime()
    config, _ = _configs()
    model = VN97Mamba2StepOnnx(
        config,
        _logical_state(_source_state(config)),
    ).eval()
    bundle = tmp_path / "g04"
    manifest = export_mamba2_step_onnx(
        model,
        bundle,
        capsule_id="a" * 64,
        capsule_manifest_sha256="b" * 64,
        source_weight_sha256="c" * 64,
        example_batch_size=1,
        export_external_data=True,
    )

    assert manifest["schema"] == "VN97M2G04ONNX1"
    assert manifest["graph_kind"] == "step"
    assert manifest["parallel_prefill_ready"] is False
    assert manifest["quantization_used"] is False
    assert manifest["production_activation_authorized"] is False
    assert manifest["state_contract"]["conv_state"]["shape"] == [
        2,
        1,
        config.conv_dim,
        config.d_conv,
    ]
    assert manifest["state_contract"]["ssm_state"]["shape"] == [
        2,
        1,
        config.n_heads,
        config.head_dim,
        config.d_state,
    ]
    verified = verify_mamba2_g04_bundle(bundle)
    assert verified["manifest_id"] == manifest["manifest_id"]

    metrics = validate_ort_step_parity(
        model,
        bundle,
        batch_size=1,
        seed=9742,
    )
    assert metrics["max_logits_abs_error"] < 1.0e-4
    assert metrics["max_conv_state_abs_error"] < 1.0e-5
    assert metrics["max_ssm_state_abs_error"] < 1.0e-5


def test_official_27b_state_budget_is_explicit() -> None:
    config = Mamba2OnnxConfig(
        d_model=2560,
        n_layers=64,
        vocab_size=50288,
        d_state=128,
        d_conv=4,
        expand=2,
        head_dim=64,
        n_groups=1,
    )
    # FP16/BF16 state budget for batch=1.
    conv_elements = (
        config.n_layers
        * config.conv_dim
        * config.d_conv
    )
    ssm_elements = (
        config.n_layers
        * config.n_heads
        * config.head_dim
        * config.d_state
    )
    assert conv_elements == 1_376_256
    assert ssm_elements == 41_943_040
    assert (conv_elements + ssm_elements) * 2 == 86_638_592
    assert (conv_elements + ssm_elements) * 4 == 173_277_184


def test_bundle_verifier_rejects_graph_tamper(
    tmp_path: Path,
) -> None:
    _onnx_runtime()
    config, _ = _configs()
    model = VN97Mamba2StepOnnx(
        config,
        _logical_state(_source_state(config)),
    ).eval()
    bundle = tmp_path / "g04"
    export_mamba2_step_onnx(
        model,
        bundle,
        capsule_id="a" * 64,
        capsule_manifest_sha256="b" * 64,
        source_weight_sha256="c" * 64,
        export_external_data=True,
    )
    with (bundle / "step.onnx").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="byte size|SHA-256"):
        verify_mamba2_g04_bundle(bundle)
