import hashlib

import pytest
import torch

from vn97 import (
    CapabilitySource,
    VN97Config,
    VN97DeploymentCheckpointFormatError,
    VN97DeploymentCheckpointIntegrityError,
    VN97LanguageCore,
    VN97TokenizerPackage,
    build_bootstrap_bundle_from_checkpoint,
    build_deployment_checkpoint,
    load_deployment_checkpoint,
    load_deployment_checkpoint_file,
    parse_capability_package,
    save_deployment_checkpoint,
)


class FakeSigner:
    key_id = "publisher.main"
    public_key = b"p" * 32

    def sign(self, message: bytes) -> bytes:
        return hashlib.sha512(self.public_key + message).digest()


def _model(*, rank=None, vocab=264):
    torch.manual_seed(97)
    model = VN97LanguageCore(
        VN97Config(
            vocab_size=vocab,
            d_model=12,
            n_layers=2,
            d_state=4,
            embedding_rank=rank,
        )
    )
    with torch.no_grad():
        # Make exact checkpoint identity independent from constructor RNG alone.
        model.layers[0].core.dt_proj.bias.add_(0.125)
    return model


def _assert_same_state(first, second):
    first_state = first.state_dict()
    second_state = second.state_dict()
    assert first_state.keys() == second_state.keys()
    for key in first_state:
        assert torch.equal(
            first_state[key].detach().cpu().float(),
            second_state[key].detach().cpu().float(),
        ), key


def test_checkpoint_roundtrip_is_deterministic_and_preserves_ties(tmp_path):
    model = _model()
    first = build_deployment_checkpoint(model)
    second = build_deployment_checkpoint(model)
    assert first == second
    assert first.startswith(b"VN97CK1\0")

    loaded = load_deployment_checkpoint(first)
    assert loaded.checkpoint_sha256 == hashlib.sha256(first).hexdigest()
    assert loaded.config == model.config
    assert not loaded.model.training
    assert loaded.model.embedding.weight is loaded.model.lm_head.weight
    _assert_same_state(model, loaded.model)

    path = tmp_path / "model.vn97ck1"
    digest = save_deployment_checkpoint(model, path)
    assert digest == loaded.checkpoint_sha256
    from_file = load_deployment_checkpoint_file(path)
    _assert_same_state(model, from_file.model)


def test_factorized_checkpoint_preserves_exact_tied_parameters():
    model = _model(rank=4)
    loaded = load_deployment_checkpoint(
        build_deployment_checkpoint(model)
    )
    assert (
        loaded.model.embedding.token_factors
        is loaded.model.lm_head.token_factors
    )
    assert (
        loaded.model.embedding.projection
        is loaded.model.lm_head.projection
    )
    _assert_same_state(model, loaded.model)


def test_checkpoint_tamper_and_nonfinite_weights_fail_closed():
    blob = bytearray(build_deployment_checkpoint(_model()))
    blob[-1] ^= 1
    with pytest.raises(
        VN97DeploymentCheckpointIntegrityError,
        match="SHA-256",
    ):
        load_deployment_checkpoint(bytes(blob))

    model = _model()
    with torch.no_grad():
        model.layers[0].core.a_log[0, 0] = float("nan")
    with pytest.raises(
        VN97DeploymentCheckpointFormatError,
        match="non-finite",
    ):
        build_deployment_checkpoint(model)


def test_checkpoint_flows_directly_into_m10k_bootstrap_bundle():
    tokenizer = VN97TokenizerPackage((b" the", b"ing"))
    model = _model(vocab=tokenizer.vocab_size)
    checkpoint = build_deployment_checkpoint(model)

    bundle = build_bootstrap_bundle_from_checkpoint(
        checkpoint,
        tokenizer=tokenizer,
        capability_version=3,
        signer=FakeSigner(),
        source_origin="vn97-training",
        source_license="proprietary",
        tile_rows=4,
        tile_cols=4,
    )
    parsed = parse_capability_package(bundle.capability_package)
    assert parsed.manifest.capability_id == "model.language"
    assert parsed.manifest.capability_version == 3
    assert parsed.manifest.source == CapabilitySource(
        "vn97-training",
        hashlib.sha256(checkpoint).hexdigest(),
        "proprietary",
    )
    assert bundle.model_image.startswith(b"VN97MI1\0")
