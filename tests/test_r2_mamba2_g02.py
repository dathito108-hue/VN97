import torch
import torch.nn.functional as F
import pytest

from vn97.r2.mamba2_parity import (
    Mamba2ParityReceipt,
    compare_tensor,
    hash_generated_tokens,
    hash_token_probe,
    require_production_parity,
)
from vn97.r2.mamba2_source_integrity import (
    PINNED_SOURCE_REVISION,
    PINNED_WEIGHT_SHA256,
    PINNED_WEIGHT_SIZE_BYTES,
    require_pinned_revision,
)
from vn97.r2.mamba2_ssd_reference import (
    Mamba2LayerReferenceState,
    Mamba2ReferenceConfig,
    mamba2_mixer_step_ref,
)
from vn97.r2.mamba2_transfer import (
    Mamba2SourceSpec,
    VN97_MAMBA2_SOURCE_TOKENIZER_ID,
    VN97_MAMBA2_SOURCE_TOKENIZER_REVISION,
    build_transfer_manifest,
)


def official_config() -> dict[str, object]:
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


def test_source_identity_is_pinned_to_official_release() -> None:
    assert PINNED_SOURCE_REVISION == (
        "99b226cc377d131cccc610ed4346db564f381f1e"
    )
    assert PINNED_WEIGHT_SHA256 == (
        "254d89bfef6dd9f44c8e3f54e0730af3f10c482969117cb37fb33486348b00be"
    )
    assert PINNED_WEIGHT_SIZE_BYTES == 5_405_424_282
    require_pinned_revision(PINNED_SOURCE_REVISION)
    with pytest.raises(ValueError, match="not pinned"):
        require_pinned_revision("main")


def test_source_contract_rejects_wrong_ssm_layer_or_hidden_override() -> None:
    wrong_layer = official_config()
    wrong_layer["ssm_cfg"] = {"layer": "Mamba1"}
    with pytest.raises(ValueError, match="locked Mamba-2 2.7B"):
        Mamba2SourceSpec.from_config(
            wrong_layer
        ).require_official_27b_contract()

    hidden_override = official_config()
    hidden_override["ssm_cfg"] = {
        "layer": "Mamba2",
        "dt_limit": [0.0, 1.0],
    }
    with pytest.raises(ValueError, match="locked Mamba-2 2.7B"):
        Mamba2SourceSpec.from_config(
            hidden_override
        ).require_official_27b_contract()


def test_transfer_manifest_pins_gpt_neox_tokenizer_lineage() -> None:
    spec = Mamba2SourceSpec.from_config(official_config())
    manifest = build_transfer_manifest(
        spec,
        source_revision=PINNED_SOURCE_REVISION,
        source_config_sha256="a" * 64,
        source_weight_sha256=PINNED_WEIGHT_SHA256,
    )
    assert (
        manifest["source_tokenizer_model_id"]
        == VN97_MAMBA2_SOURCE_TOKENIZER_ID
        == "EleutherAI/gpt-neox-20b"
    )
    assert (
        manifest["source_tokenizer_revision"]
        == VN97_MAMBA2_SOURCE_TOKENIZER_REVISION
        == "364ae95407723fadd1d47b023c1efb92a4d891c3"
    )


def _tiny_tensors(config: Mamba2ReferenceConfig) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(97)
    return {
        "in_proj.weight": torch.randn(
            config.in_proj_dim,
            config.d_model,
            generator=generator,
        )
        * 0.05,
        "conv1d.weight": torch.randn(
            config.conv_dim,
            1,
            config.d_conv,
            generator=generator,
        )
        * 0.05,
        "conv1d.bias": torch.randn(
            config.conv_dim,
            generator=generator,
        )
        * 0.01,
        "dt_bias": torch.randn(
            config.n_heads,
            generator=generator,
        )
        * 0.05,
        "A_log": torch.randn(
            config.n_heads,
            generator=generator,
        )
        * 0.05,
        "D": torch.randn(
            config.n_heads,
            generator=generator,
        )
        * 0.05,
        "norm.weight": torch.randn(
            config.d_inner,
            generator=generator,
        )
        * 0.05
        + 1.0,
        "out_proj.weight": torch.randn(
            config.d_model,
            config.d_inner,
            generator=generator,
        )
        * 0.05,
    }


def _official_step_oracle(
    hidden: torch.Tensor,
    state: Mamba2LayerReferenceState,
    tensors: dict[str, torch.Tensor],
    config: Mamba2ReferenceConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zxbcdt = F.linear(hidden, tensors["in_proj.weight"])
    z, xbc, dt = torch.split(
        zxbcdt,
        [
            config.d_inner,
            config.conv_dim,
            config.n_heads,
        ],
        dim=-1,
    )
    conv = torch.roll(state.conv, shifts=-1, dims=-1).clone()
    conv[:, :, -1] = xbc
    xbc = (
        conv * tensors["conv1d.weight"][:, 0, :]
    ).sum(dim=-1)
    xbc = F.silu(xbc + tensors["conv1d.bias"])
    x, b_value, c_value = torch.split(
        xbc,
        [
            config.d_inner,
            config.d_state,
            config.d_state,
        ],
        dim=-1,
    )
    a = -torch.exp(tensors["A_log"].float())
    dt_value = F.softplus(dt + tensors["dt_bias"])
    d_a = torch.exp(dt_value.float() * a[None])
    x_heads = x.reshape(
        hidden.shape[0],
        config.n_heads,
        config.head_dim,
    )
    next_ssm = (
        state.ssm.float() * d_a[:, :, None, None]
        + torch.einsum(
            "bh,bn,bhp->bhpn",
            dt_value.float(),
            b_value.float(),
            x_heads.float(),
        )
    )
    y = torch.einsum(
        "bhpn,bn->bhp",
        next_ssm.to(x_heads.dtype),
        c_value.to(x_heads.dtype),
    )
    y = (
        y
        + tensors["D"][None, :, None] * x_heads
    ).reshape(hidden.shape[0], config.d_inner)

    # Official Mamba2 default norm_before_gate=False.
    y = y.float() * F.silu(z.float())
    grouped = y.reshape(
        hidden.shape[0],
        config.n_groups,
        config.d_inner // config.n_groups,
    )
    inv = torch.rsqrt(
        grouped.square().mean(dim=-1, keepdim=True)
        + config.rms_eps
    )
    y = (
        grouped * inv
    ).reshape_as(y) * tensors["norm.weight"].float()
    out = F.linear(y.to(hidden.dtype), tensors["out_proj.weight"])
    return out, conv, next_ssm.to(state.ssm.dtype)


def test_native_ssd_step_matches_official_unfused_equations() -> None:
    config = Mamba2ReferenceConfig(
        d_model=4,
        d_state=3,
        d_conv=3,
        expand=2,
        head_dim=2,
        n_groups=1,
    )
    tensors = _tiny_tensors(config)
    generator = torch.Generator().manual_seed(971)
    hidden = torch.randn(2, 4, generator=generator)
    state = Mamba2LayerReferenceState(
        conv=torch.randn(
            2,
            config.conv_dim,
            config.d_conv,
            generator=generator,
        )
        * 0.02,
        ssm=torch.randn(
            2,
            config.n_heads,
            config.head_dim,
            config.d_state,
            generator=generator,
        )
        * 0.02,
    )
    expected_out, expected_conv, expected_ssm = _official_step_oracle(
        hidden,
        state,
        tensors,
        config,
    )
    out, next_state = mamba2_mixer_step_ref(
        hidden,
        state,
        tensors,
        config,
    )
    assert torch.allclose(out, expected_out, atol=1e-6, rtol=1e-6)
    assert torch.equal(next_state.conv, expected_conv)
    assert torch.allclose(
        next_state.ssm,
        expected_ssm,
        atol=1e-6,
        rtol=1e-6,
    )


def test_parity_receipt_is_fail_closed() -> None:
    source = torch.tensor([[1.0, 2.0]])
    same = source.clone()
    comparisons = tuple(
        compare_tensor(
            name,
            source,
            same,
            atol=1e-6,
            rtol=1e-6,
        )
        for name in (
            "layer0.hidden",
            "layer0.conv_state",
            "layer0.ssm_state",
            "final.logits",
        )
    )
    generated = hash_generated_tokens([7, 11, 13])
    receipt = Mamba2ParityReceipt(
        source_weight_sha256=PINNED_WEIGHT_SHA256,
        vn97_checkpoint_sha256="b" * 64,
        token_probe_sha256=hash_token_probe([1, 2, 3]),
        precision="fp32",
        implementation_source="test",
        comparisons=comparisons,
        source_generated_tokens_sha256=generated,
        vn97_generated_tokens_sha256=generated,
    )
    require_production_parity(receipt)
    assert receipt.passed

    bad_comparisons = list(comparisons)
    bad_comparisons[-1] = compare_tensor(
        "final.logits",
        source,
        source + 0.1,
        atol=1e-6,
        rtol=1e-6,
    )
    bad = Mamba2ParityReceipt(
        source_weight_sha256=PINNED_WEIGHT_SHA256,
        vn97_checkpoint_sha256="b" * 64,
        token_probe_sha256=hash_token_probe([1, 2, 3]),
        precision="fp32",
        implementation_source="test",
        comparisons=tuple(bad_comparisons),
        source_generated_tokens_sha256=generated,
        vn97_generated_tokens_sha256=generated,
    )
    with pytest.raises(ValueError, match="final.logits"):
        require_production_parity(bad)
