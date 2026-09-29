from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vn97.r2.mamba2_onnx import (
    Mamba2OnnxConfig,
    VN97Mamba2StepOnnx,
)
from vn97.r2.mamba2_parallel_onnx import (
    VN97Mamba2ParallelChunkOnnx,
    export_parallel_recurrent_onnx,
    repeated_step_reference,
    validate_ort_parallel_parity,
    validate_parallel_parity,
    verify_g05_bundle,
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
    generator = torch.Generator().manual_seed(9750)
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


def _model(chunk_size: int = 8) -> VN97Mamba2ParallelChunkOnnx:
    config = _config()
    step = VN97Mamba2StepOnnx(
        config,
        _logical_state(config),
    ).eval()
    return VN97Mamba2ParallelChunkOnnx(
        step,
        chunk_size=chunk_size,
    ).eval()


@pytest.mark.parametrize("valid_length", [1, 3, 8])
def test_parallel_chunk_matches_repeated_step_from_zero_state(
    valid_length: int,
) -> None:
    model = _model()
    metrics = validate_parallel_parity(
        model,
        valid_length=valid_length,
        seed=9751 + valid_length,
    )
    assert metrics["max_logits_abs_error"] < 2.0e-5
    assert metrics["max_conv_state_abs_error"] < 1.0e-6
    assert metrics["max_ssm_state_abs_error"] < 2.0e-5


@pytest.mark.parametrize("valid_length", [1, 5, 8])
def test_parallel_chunk_matches_repeated_step_from_nonzero_state(
    valid_length: int,
) -> None:
    model = _model()
    conv, ssm = model.initial_state()
    generator = torch.Generator().manual_seed(9760 + valid_length)
    conv = torch.randn(
        conv.shape,
        generator=generator,
    ) * 0.01
    ssm = torch.randn(
        ssm.shape,
        generator=generator,
    ) * 0.01

    generator = torch.Generator().manual_seed(9770 + valid_length)
    tokens = torch.randint(
        0,
        model.config.vocab_size,
        (1, model.chunk_size),
        generator=generator,
        dtype=torch.long,
    )
    actual = model(
        tokens,
        torch.tensor([valid_length], dtype=torch.long),
        conv,
        ssm,
    )
    expected = repeated_step_reference(
        model.step,
        tokens,
        valid_length=valid_length,
        conv_state=conv,
        ssm_state=ssm,
    )
    torch.testing.assert_close(
        actual[0],
        expected[0],
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    torch.testing.assert_close(
        actual[1],
        expected[1],
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(
        actual[2],
        expected[2],
        atol=2.0e-5,
        rtol=2.0e-5,
    )


def test_valid_length_one_is_same_decode_contract() -> None:
    model = _model()
    generator = torch.Generator().manual_seed(9780)
    tokens = torch.randint(
        0,
        model.config.vocab_size,
        (1, model.chunk_size),
        generator=generator,
        dtype=torch.long,
    )
    conv, ssm = model.initial_state()
    parallel = model(
        tokens,
        torch.tensor([1], dtype=torch.long),
        conv,
        ssm,
    )
    step = model.step(tokens[:, 0], conv, ssm)
    torch.testing.assert_close(
        parallel[0][:, 0, :],
        step[0],
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    assert torch.count_nonzero(parallel[0][:, 1:, :]).item() == 0
    torch.testing.assert_close(parallel[1], step[1])
    torch.testing.assert_close(
        parallel[2],
        step[2],
        atol=2.0e-5,
        rtol=2.0e-5,
    )


def _onnx_runtime() -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")


@pytest.mark.parametrize("valid_length", [1, 5, 8])
def test_unified_parallel_onnx_matches_repeated_step(
    tmp_path: Path,
    valid_length: int,
) -> None:
    _onnx_runtime()
    model = _model()
    bundle = tmp_path / f"bundle-{valid_length}"
    manifest = export_parallel_recurrent_onnx(
        model,
        bundle,
        capsule_id="a" * 64,
        capsule_manifest_sha256="b" * 64,
        source_weight_sha256="c" * 64,
        external_data=True,
    )
    assert manifest["single_weight_graph"] is True
    assert manifest["decode_via_valid_length_one"] is True
    assert manifest["parallel_prefill_ready"] is True
    assert manifest["max_chunk_size"] == 8
    assert manifest["production_activation_authorized"] is False
    assert len(
        [
            item
            for item in manifest["graph_files"]
            if item["filename"].endswith(".onnx")
        ]
    ) == 1

    metrics = validate_ort_parallel_parity(
        model,
        bundle,
        valid_length=valid_length,
        seed=9790 + valid_length,
    )
    assert metrics["max_logits_abs_error"] < 2.0e-4
    assert metrics["max_conv_state_abs_error"] < 2.0e-5
    assert metrics["max_ssm_state_abs_error"] < 2.0e-4

    verified = verify_g05_bundle(bundle)
    assert verified["manifest_id"] == manifest["manifest_id"]


def test_g05_verifier_rejects_graph_tamper(
    tmp_path: Path,
) -> None:
    _onnx_runtime()
    model = _model()
    bundle = tmp_path / "bundle"
    export_parallel_recurrent_onnx(
        model,
        bundle,
        capsule_id="a" * 64,
        capsule_manifest_sha256="b" * 64,
        source_weight_sha256="c" * 64,
        external_data=True,
    )
    with (bundle / "recurrent-8.onnx").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="byte size|hash"):
        verify_g05_bundle(bundle)


@pytest.mark.parametrize('valid_length', [1, 3, 8])
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_state_scan_preserves_each_token_rounding(valid_length, dtype):
    model = _model().to(dtype=dtype)
    layer = model.step.layers[0]
    cfg = model.config
    generator = torch.Generator().manual_seed(9734)
    def random(*shape):
        return torch.randn(*shape, generator=generator).to(dtype)
    dt = torch.sigmoid(random(1, 8, cfg.n_heads))
    b = random(1, 8, cfg.d_state)
    c = random(1, 8, cfg.d_state)
    x = random(1, 8, cfg.n_heads, cfg.head_dim)
    initial = random(1, cfg.n_heads, cfg.head_dim, cfg.d_state)
    before = initial.clone()
    actual_y, actual_state = model._state_scan(
        layer, dt, b, c, x, initial, torch.tensor([valid_length]),
    )
    # Independent recurrence, deliberately not the new shared helper.
    state = initial.clone()
    outputs = []
    for index in range(valid_length):
        decay = torch.exp(dt[:, index].float() * -torch.exp(layer.a_log.float()).unsqueeze(0))
        drive = torch.einsum('bh,bn,bhp->bhpn', dt[:, index], b[:, index], x[:, index])
        state = (state.float() * decay[:, :, None, None] + drive).to(dtype)
        outputs.append(torch.einsum('bhpn,bn->bhp', state, c[:, index]))
    expected_y = torch.zeros_like(actual_y)
    expected_y[:, :valid_length] = torch.stack(outputs, dim=1)
    torch.testing.assert_close(actual_y, expected_y, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, state, rtol=0, atol=0)
    torch.testing.assert_close(initial, before, rtol=0, atol=0)
    # Padding may contain arbitrary data: it must not advance state or readout.
    dt[:, valid_length:] = float('nan')
    y_padded, state_padded = model._state_scan(
        layer, dt, b, c, x, initial, torch.tensor([valid_length]),
    )
    torch.testing.assert_close(y_padded, expected_y, rtol=0, atol=0)
    torch.testing.assert_close(state_padded, state, rtol=0, atol=0)


def test_fp16_export_preserves_recurrent_continuation(tmp_path):
    _onnx_runtime()
    import numpy as np
    import onnxruntime as ort
    model = _model().half()
    bundle = tmp_path / 'fp16'
    manifest = export_parallel_recurrent_onnx(
        model, bundle, capsule_id='a' * 64, capsule_manifest_sha256='b' * 64,
        source_weight_sha256='c' * 64, external_data=True,
    )
    assert manifest['parallel_algorithm'] == 'parallel_projection_conv_token_rounded_state_scan'
    session = ort.InferenceSession(str(bundle / 'recurrent-8.onnx'), providers=['CPUExecutionProvider'])
    conv, ssm = model.initial_state()
    ort_conv, ort_ssm = conv.numpy(), ssm.numpy()
    generator = torch.Generator().manual_seed(9744)
    with torch.inference_mode():
        for length in (8, 1, 3):
            tokens = torch.randint(0, model.config.vocab_size, (1, 8), generator=generator)
            expected, conv, ssm = repeated_step_reference(
                model.step, tokens, valid_length=length, conv_state=conv, ssm_state=ssm,
            )
            actual, ort_conv, ort_ssm = session.run(None, {
                'input_ids': tokens.numpy(), 'valid_length': np.asarray([length], dtype=np.int64),
                'conv_state': ort_conv, 'ssm_state': ort_ssm,
            })
            for left, right in ((actual, expected), (ort_conv, conv), (ort_ssm, ssm)):
                assert np.isfinite(left).all()
                # Same absolute gate as the full-weight numerical audit.
                np.testing.assert_allclose(left, right.numpy(), rtol=0, atol=0.002)
            assert np.count_nonzero(actual[:, length:]) == 0
