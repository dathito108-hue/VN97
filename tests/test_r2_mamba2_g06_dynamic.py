from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vn97.r2.mamba2_dynamic_onnx import (
    VN97Mamba2DynamicSequenceOnnx,
    export_dynamic_sequence_onnx,
    validate_dynamic_native_parity,
    validate_dynamic_ort_parity,
    verify_g06_bundle,
)
from vn97.r2.mamba2_onnx import (
    Mamba2OnnxConfig,
    VN97Mamba2StepOnnx,
)


def _config() -> Mamba2OnnxConfig:
    return Mamba2OnnxConfig(
        d_model=8,
        n_layers=2,
        vocab_size=31,
        d_state=4,
        d_conv=3,
        expand=2,
        head_dim=4,
        n_groups=1,
    )


def _logical_state(
    config: Mamba2OnnxConfig,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(9760)
    state: dict[str, torch.Tensor] = {
        "vn97.core.embedding.weight": torch.randn(
            config.vocab_size,
            config.d_model,
            generator=generator,
        ) * 0.05,
        "vn97.core.final_norm.weight": (
            torch.randn(
                config.d_model,
                generator=generator,
            ) * 0.03
            + 1.0
        ),
    }
    in_proj_dim = (
        2 * config.d_inner
        + 2 * config.d_state
        + config.n_heads
    )
    for index in range(config.n_layers):
        prefix = f"vn97.core.layers.{index}"
        mixer = f"{prefix}.mixer"
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


def _model(max_sequence_length: int = 8) -> VN97Mamba2DynamicSequenceOnnx:
    config = _config()
    step = VN97Mamba2StepOnnx(
        config,
        _logical_state(config),
    ).eval()
    return VN97Mamba2DynamicSequenceOnnx(
        step,
        max_sequence_length=max_sequence_length,
    ).eval()


@pytest.mark.parametrize("length", [1, 3, 8])
@pytest.mark.parametrize("nonzero_state", [False, True])
def test_dynamic_native_matches_repeated_step(
    length: int,
    nonzero_state: bool,
) -> None:
    metrics = validate_dynamic_native_parity(
        _model(),
        sequence_length=length,
        nonzero_state=nonzero_state,
        seed=9761 + length + int(nonzero_state) * 20,
    )
    assert metrics["max_logits_abs_error"] < 2.0e-5
    assert metrics["max_conv_state_abs_error"] < 1.0e-6
    assert metrics["max_ssm_state_abs_error"] < 2.0e-5


def _onnx_runtime() -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")


def test_one_exported_graph_accepts_decode_tail_and_full_prefill(
    tmp_path: Path,
) -> None:
    _onnx_runtime()
    model = _model()
    bundle = tmp_path / "dynamic"
    manifest = export_dynamic_sequence_onnx(
        model,
        bundle,
        capsule_id="a" * 64,
        capsule_manifest_sha256="b" * 64,
        source_weight_sha256="c" * 64,
        example_sequence_length=5,
        external_data=True,
    )
    assert manifest["sequence_length_min"] == 1
    assert manifest["sequence_length_max"] == 8
    assert manifest["single_weight_graph"] is True
    assert manifest["decode_sequence_length"] == 1
    assert manifest["logits_dtype"] == "float32"
    assert manifest["production_activation_authorized"] is False

    for length in (1, 3, 8):
        metrics = validate_dynamic_ort_parity(
            model,
            bundle,
            sequence_length=length,
            seed=9770 + length,
        )
        assert metrics["max_logits_abs_error"] < 2.0e-4
        assert metrics["max_conv_state_abs_error"] < 2.0e-5
        assert metrics["max_ssm_state_abs_error"] < 2.0e-4

    verified = verify_g06_bundle(bundle)
    assert verified["manifest_id"] == manifest["manifest_id"]


def test_dynamic_bundle_rejects_graph_tamper(
    tmp_path: Path,
) -> None:
    _onnx_runtime()
    model = _model()
    bundle = tmp_path / "dynamic"
    export_dynamic_sequence_onnx(
        model,
        bundle,
        capsule_id="a" * 64,
        capsule_manifest_sha256="b" * 64,
        source_weight_sha256="c" * 64,
        example_sequence_length=5,
        external_data=True,
    )
    with (bundle / "recurrent-dynamic.onnx").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="size|hash"):
        verify_g06_bundle(bundle)
