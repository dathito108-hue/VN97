from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vn97.r2.checkpoint import save_r2_checkpoint
from vn97.r2.config import r2_smoke_config
from vn97.r2.model import VN97R2Model
from vn97.r2.onnx_export import (
    R2_ONNX_BUNDLE_SCHEMA,
    R2OnnxInvocation,
    compile_onnx_invocation_plan,
    export_r2_checkpoint_onnx_bundle,
    export_r2_onnx_bundle,
    ort_run_graph,
    stack_r2_state,
    unstack_r2_state,
    validate_ort_parity,
    verify_r2_onnx_bundle,
)


def _model() -> VN97R2Model:
    torch.manual_seed(9720)
    model = VN97R2Model(r2_smoke_config(vocab_size=257))
    model.eval()
    return model


def _onnx_runtime() -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")


def test_state_stack_round_trip_preserves_all_layers() -> None:
    model = _model()
    state = model.initial_state(
        2,
        device="cpu",
        dtype=torch.float32,
        profile="deep",
    )
    conv, ssm = stack_r2_state(state)
    restored = unstack_r2_state(
        conv,
        ssm,
        active_layers=state.active_layers,
    )

    assert conv.shape == (
        model.config.n_layers,
        2,
        model.config.d_inner,
        model.config.d_conv - 1,
    )
    assert ssm.shape == (
        model.config.n_layers,
        2,
        model.config.d_inner,
        model.config.d_state,
    )
    assert restored.active_layers == state.active_layers
    for left, right in zip(state.layers, restored.layers):
        torch.testing.assert_close(left.conv, right.conv)
        torch.testing.assert_close(left.ssm, right.ssm)


def test_invocation_plan_uses_available_chunks_then_step_tail() -> None:
    plan = compile_onnx_invocation_plan(
        (8, 16, 32, 12, 64),
        available_chunk_sizes=(32, 8),
    )
    assert sum(item.sequence_length for item in plan) == 132
    assert plan[:3] == (
        R2OnnxInvocation("chunk", 8, "chunk-8.onnx"),
        R2OnnxInvocation("chunk", 8, "chunk-8.onnx"),
        R2OnnxInvocation("chunk", 8, "chunk-8.onnx"),
    )
    # 32 is covered by one graph; 12 becomes 8 + four recurrent steps.
    assert R2OnnxInvocation("chunk", 32, "chunk-32.onnx") in plan
    assert sum(
        1
        for item in plan
        if item.kind == "step"
    ) == 4
    assert plan[-2:] == (
        R2OnnxInvocation("chunk", 32, "chunk-32.onnx"),
        R2OnnxInvocation("chunk", 32, "chunk-32.onnx"),
    )


def test_invocation_plan_can_run_parallel_prefill_with_one_graph() -> None:
    plan = compile_onnx_invocation_plan(
        (512,),
        available_chunk_sizes=(32,),
    )
    assert len(plan) == 16
    assert all(
        item == R2OnnxInvocation(
            "chunk",
            32,
            "chunk-32.onnx",
        )
        for item in plan
    )


def test_export_step_and_chunk8_match_pytorch_and_carry_state(
    tmp_path: Path,
) -> None:
    _onnx_runtime()
    model = _model()
    bundle = tmp_path / "bundle"
    manifest = export_r2_onnx_bundle(
        model,
        bundle,
        profile="deep",
        chunk_sizes=(8,),
    )

    assert manifest["schema"] == R2_ONNX_BUNDLE_SCHEMA
    assert manifest["active_layers"] == model.config.n_layers
    assert manifest["supported_chunk_sizes"] == [8]
    assert manifest["quantization_used"] is False
    assert manifest["production_lowering_used"] is False
    assert verify_r2_onnx_bundle(bundle)["bundle_id"] == manifest[
        "bundle_id"
    ]

    step_metrics = validate_ort_parity(
        model,
        bundle,
        profile="deep",
        batch_size=2,
        seed=9721,
    )
    chunk_metrics = validate_ort_parity(
        model,
        bundle,
        profile="deep",
        chunk_size=8,
        batch_size=2,
        seed=9722,
    )
    assert step_metrics["max_logit_abs_error"] < 1e-3
    assert chunk_metrics["max_logit_abs_error"] < 1e-3

    torch.manual_seed(9723)
    prompt = torch.randint(
        0,
        model.config.vocab_size,
        (1, 8),
        dtype=torch.long,
    )
    next_token = torch.randint(
        0,
        model.config.vocab_size,
        (1,),
        dtype=torch.long,
    )
    initial = model.initial_state(
        1,
        device="cpu",
        dtype=torch.float32,
        profile="deep",
    )
    conv, ssm = stack_r2_state(initial)

    ort_chunk_logits, ort_conv, ort_ssm = ort_run_graph(
        bundle / "chunk-8.onnx",
        input_ids=prompt,
        conv_state=conv,
        ssm_state=ssm,
    )
    ort_step_logits, ort_next_conv, ort_next_ssm = ort_run_graph(
        bundle / "step.onnx",
        input_ids=next_token,
        conv_state=ort_conv,
        ssm_state=ort_ssm,
    )

    reference_chunk_logits, reference_state = model(
        prompt,
        initial,
        profile="deep",
    )
    reference_step_logits, reference_state = model.step(
        next_token,
        reference_state,
        profile="deep",
    )
    reference_conv, reference_ssm = stack_r2_state(
        reference_state
    )
    torch.testing.assert_close(
        ort_chunk_logits,
        reference_chunk_logits,
        rtol=5e-4,
        atol=5e-5,
    )
    torch.testing.assert_close(
        ort_step_logits,
        reference_step_logits,
        rtol=5e-4,
        atol=5e-5,
    )
    torch.testing.assert_close(
        ort_next_conv,
        reference_conv,
        rtol=5e-4,
        atol=5e-5,
    )
    torch.testing.assert_close(
        ort_next_ssm,
        reference_ssm,
        rtol=5e-4,
        atol=5e-5,
    )


def test_checkpoint_export_binds_checkpoint_identity(
    tmp_path: Path,
) -> None:
    _onnx_runtime()
    model = _model()
    checkpoint = tmp_path / "model.r2.pt"
    checkpoint_sha = save_r2_checkpoint(
        checkpoint,
        model,
        stage="dense_pretrain",
        metadata={"test": "e2"},
    )
    bundle = tmp_path / "bundle"
    manifest = export_r2_checkpoint_onnx_bundle(
        checkpoint,
        bundle,
        profile="fast",
        chunk_sizes=(8,),
    )
    assert manifest["checkpoint_sha256"] == checkpoint_sha
    assert manifest["checkpoint_stage"] == "dense_pretrain"
    assert manifest["active_layers"] == model.config.fast_layers
    validate_ort_parity(
        model,
        bundle,
        profile="fast",
        chunk_size=8,
        batch_size=1,
    )


def test_bundle_verifier_rejects_graph_tamper(
    tmp_path: Path,
) -> None:
    _onnx_runtime()
    bundle = tmp_path / "bundle"
    export_r2_onnx_bundle(
        _model(),
        bundle,
        chunk_sizes=(8,),
    )
    with (bundle / "step.onnx").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_r2_onnx_bundle(bundle)


def test_manifest_identity_rejects_contract_tamper(
    tmp_path: Path,
) -> None:
    _onnx_runtime()
    bundle = tmp_path / "bundle"
    export_r2_onnx_bundle(
        _model(),
        bundle,
        chunk_sizes=(8,),
    )
    path = bundle / "manifest.vn97onnx1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["active_layers"] -= 1
    path.write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_r2_onnx_bundle(bundle)
