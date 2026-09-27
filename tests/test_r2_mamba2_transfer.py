import json

import pytest
import torch

from vn97.r2.mamba2_augmentation import (
    VN97Mamba2AugmentationBank,
    VN97Mamba2LayerAugmentation,
    VN97RecurrentReasoningPolicy,
)
from vn97.r2.mamba2_evolution import (
    VN97EvolutionCandidate,
    VN97EvolutionPromotionEvidence,
    VN97EvolutionStage,
    require_evolution_transition,
)
from vn97.r2.mamba2_transfer import (
    Mamba2SourceSpec,
    build_transfer_manifest,
    expected_source_shapes,
    expected_unique_parameter_count,
)


def canonical_config() -> dict[str, object]:
    return {
        "d_model": 2560,
        "d_intermediate": 0,
        "n_layer": 64,
        "vocab_size": 50277,
        "ssm_cfg": {"layer": "Mamba2"},
        "attn_layer_idx": [],
        "attn_cfg": {},
        "rms_norm": True,
        "residual_in_fp32": True,
        "fused_add_norm": True,
        "pad_vocab_size_multiple": 16,
        "tie_embeddings": True,
    }


def test_mamba2_27b_contract_without_allocating_model() -> None:
    spec = Mamba2SourceSpec.from_config(canonical_config())
    spec.require_official_27b_contract()
    assert spec.padded_vocab_size == 50288
    assert spec.d_inner == 5120
    assert spec.n_heads == 80
    assert spec.conv_dim == 5376
    assert spec.in_proj_dim == 10576
    assert spec.recurrent_ssm_state_shape == (80, 64, 128)
    assert spec.recurrent_conv_state_shape == (5376, 4)
    assert expected_unique_parameter_count(spec) == 2_702_599_680

    shapes = expected_source_shapes(spec)
    assert shapes["backbone.layers.0.mixer.in_proj.weight"] == (
        10576,
        2560,
    )
    assert shapes["backbone.layers.63.mixer.A_log"] == (80,)
    assert shapes["backbone.layers.63.mixer.norm.weight"] == (5120,)


def test_contract_rejects_attention_or_shape_drift() -> None:
    bad = canonical_config()
    bad["attn_layer_idx"] = [31]
    with pytest.raises(ValueError, match="locked Mamba-2 2.7B"):
        Mamba2SourceSpec.from_config(bad).require_official_27b_contract()


def test_transfer_manifest_declares_lossless_preservation() -> None:
    spec = Mamba2SourceSpec.from_config(canonical_config())
    manifest = build_transfer_manifest(
        spec,
        source_revision="deadbee",
        source_config_sha256="a" * 64,
        source_weight_sha256="b" * 64,
    )
    assert manifest["transfer_semantics"] == "tensor_value_identity_1to1"
    assert manifest["tokenizer_semantics"] == "source_tokenizer_preserved"
    assert manifest["quantization_used"] is False
    assert manifest["lossy_mapping_used"] is False
    assert manifest["core_reinitialized"] is False
    assert manifest["unique_core_parameters"] == 2_702_599_680
    assert len(manifest["manifest_id"]) == 64
    json.dumps(manifest, sort_keys=True)


def test_augmentation_is_exactly_zero_impact_at_g0() -> None:
    torch.manual_seed(97)
    layer = VN97Mamba2LayerAugmentation(8)
    hidden = torch.randn(2, 8)
    memory = torch.randn(2, 8)
    state = layer.initial_state(
        2,
        device=hidden.device,
        dtype=hidden.dtype,
    )
    output, next_state = layer(
        hidden,
        state,
        retrieved_memory=memory,
        enabled=True,
    )
    assert layer.zero_impact()
    assert torch.equal(output, hidden)
    assert not torch.equal(next_state.fast, next_state.slow)


def test_augmentation_bank_is_small_and_reasoning_reuses_same_model() -> None:
    bank = VN97Mamba2AugmentationBank(8, 2)
    assert bank.zero_impact()
    assert bank.trainable_parameter_count() > 0
    policy = VN97RecurrentReasoningPolicy()
    assert policy.passes_for("fast") == 1
    assert policy.passes_for("deep") == 4
    assert policy.passes_for("adaptive") == 6


def candidate(
    stage: VN97EvolutionStage,
    *,
    parent: str,
    current: str,
    unfrozen: int,
    native: int,
    inherited: int,
    fresh: bool = False,
) -> VN97EvolutionCandidate:
    return VN97EvolutionCandidate(
        stage=stage,
        parent_checkpoint_sha256=parent,
        candidate_checkpoint_sha256=current,
        core_layers_total=64,
        core_layers_unfrozen=unfrozen,
        native_blocks=native,
        inherited_source_tensors=inherited,
        fresh_initialization=fresh,
        held_out_evidence_sha256="c" * 64,
        architecture_evidence_sha256="d" * 64,
    )


def test_controlled_evolution_does_not_fake_mamba_independence() -> None:
    g0 = candidate(
        VN97EvolutionStage.G0_DIRECT_TRANSFER,
        parent="0" * 64,
        current="1" * 64,
        unfrozen=0,
        native=0,
        inherited=579,
    )
    g1 = candidate(
        VN97EvolutionStage.G1_AUGMENTED,
        parent="1" * 64,
        current="2" * 64,
        unfrozen=0,
        native=0,
        inherited=579,
    )
    g4 = candidate(
        VN97EvolutionStage.G4_NATIVE_PRETRAIN,
        parent="4" * 64,
        current="5" * 64,
        unfrozen=64,
        native=64,
        inherited=0,
        fresh=True,
    )
    g0.validate()
    require_evolution_transition(g0, g1)
    assert not g0.mamba_weight_independent
    assert not g1.mamba_weight_independent
    assert g4.mamba_weight_independent


def test_promotion_still_requires_human_and_rollback_evidence() -> None:
    with pytest.raises(ValueError, match="explicit human approval"):
        VN97EvolutionPromotionEvidence(
            held_out_passed=True,
            authority_passed=True,
            rollback_ready=True,
            explicit_human_approval=False,
        ).require_promotable()
    VN97EvolutionPromotionEvidence(
        held_out_passed=True,
        authority_passed=True,
        rollback_ready=True,
        explicit_human_approval=True,
    ).require_promotable()
