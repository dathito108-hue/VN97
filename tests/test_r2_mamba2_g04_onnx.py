from pathlib import Path

import torch

from vn97.r2.mamba2_onnx import (
    Mamba2OnnxGraphRecord,
    Mamba2ReferenceConfig,
    build_mamba2_onnx_manifest,
    export_tiny_mamba2_step_graph,
    official_27b_state_contract,
)
from vn97.r2.mamba2_source_integrity import PINNED_WEIGHT_SHA256


def _tiny_tensors(config: Mamba2ReferenceConfig) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(9704)
    return {
        "in_proj.weight": torch.randn(
            config.in_proj_dim,
            config.d_model,
            generator=g,
        ) * 0.05,
        "conv1d.weight": torch.randn(
            config.conv_dim,
            1,
            config.d_conv,
            generator=g,
        ) * 0.05,
        "conv1d.bias": torch.randn(config.conv_dim, generator=g) * 0.01,
        "dt_bias": torch.randn(config.n_heads, generator=g) * 0.05,
        "A_log": torch.randn(config.n_heads, generator=g) * 0.05,
        "D": torch.randn(config.n_heads, generator=g) * 0.05,
        "norm.weight": torch.randn(config.d_inner, generator=g) * 0.05 + 1.0,
        "out_proj.weight": torch.randn(
            config.d_model,
            config.d_inner,
            generator=g,
        ) * 0.05,
    }


def test_official_mamba2_onnx_state_contract() -> None:
    contract = official_27b_state_contract().canonical_object()
    assert contract["n_layers"] == 64
    assert contract["conv_state"]["shape"] == [64, "batch", 5376, 4]
    assert contract["ssm_state"]["shape"] == [
        64,
        "batch",
        80,
        64,
        128,
    ]
    assert contract["state_is_explicit"] is True
    assert contract["same_weights_semantics"] is True


def test_manifest_binds_capsule_and_same_weights() -> None:
    graphs = (
        Mamba2OnnxGraphRecord(
            filename="step.onnx",
            kind="step",
            sequence_length=1,
            sha256="a" * 64,
            bytes=100,
        ),
        Mamba2OnnxGraphRecord(
            filename="chunk-32.onnx",
            kind="chunk",
            sequence_length=32,
            sha256="b" * 64,
            bytes=200,
        ),
    )
    manifest = build_mamba2_onnx_manifest(
        capsule_id="c" * 64,
        capsule_manifest_sha256="d" * 64,
        graphs=graphs,
        chunks=(32,),
    )
    assert manifest["schema"] == "VN97M2ONNX1"
    assert manifest["source_weight_sha256"] == PINNED_WEIGHT_SHA256
    assert manifest["same_weights_semantics"] is True
    assert manifest["quantization_used"] is False
    assert manifest["augmentation_effect"] == "exact_zero"
    assert manifest["production_activation_authorized"] is False
    assert len(manifest["bundle_id"]) == 64


def test_tiny_explicit_state_onnx_export(tmp_path: Path) -> None:
    config = Mamba2ReferenceConfig(
        d_model=4,
        d_state=3,
        d_conv=3,
        expand=2,
        head_dim=2,
        n_groups=1,
    )
    record = export_tiny_mamba2_step_graph(
        tensors=_tiny_tensors(config),
        config=config,
        output_path=tmp_path / "step.onnx",
    )
    assert record.filename == "step.onnx"
    assert record.kind == "step"
    assert record.sequence_length == 1
    assert record.bytes > 0
    assert len(record.sha256) == 64
    assert (tmp_path / "step.onnx").is_file()
