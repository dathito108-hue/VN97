import hashlib

import pytest
import torch

from vn97 import (
    CapabilitySource,
    VN97BootstrapBundleError,
    VN97Config,
    VN97LanguageCore,
    VN97TokenizerPackage,
    build_bootstrap_bundle,
    parse_capability_package,
    parse_signature_envelope,
    write_bootstrap_assets,
)


class FakeSigner:
    key_id = "publisher.main"
    public_key = b"p" * 32

    def sign(self, message: bytes) -> bytes:
        return hashlib.sha512(self.public_key + message).digest()


class BadSigner(FakeSigner):
    def sign(self, message: bytes) -> bytes:
        return b"bad"


def _model(tokenizer: VN97TokenizerPackage) -> VN97LanguageCore:
    torch.manual_seed(97)
    return VN97LanguageCore(
        VN97Config(
            vocab_size=tokenizer.vocab_size,
            d_model=12,
            n_layers=2,
            d_state=4,
        )
    ).eval()


def _source() -> CapabilitySource:
    return CapabilitySource(
        "trained-vn97-checkpoint",
        hashlib.sha256(b"checkpoint-v1").hexdigest(),
        "proprietary",
    )


def test_bootstrap_bundle_matches_m10j_contract(tmp_path):
    tokenizer = VN97TokenizerPackage((b" the", b"ing", b"VN97"))
    bundle = build_bootstrap_bundle(
        _model(tokenizer),
        tokenizer=tokenizer,
        source=_source(),
        capability_version=7,
        signer=FakeSigner(),
        tile_rows=4,
        tile_cols=4,
    )

    assert bundle.model_image.startswith(b"VN97MI1\0")
    assert bundle.publisher_public_key == b"p" * 32
    assert bundle.model_image_sha256 == hashlib.sha256(
        bundle.model_image
    ).hexdigest()
    assert bundle.package_sha256 == hashlib.sha256(
        bundle.capability_package
    ).hexdigest()

    parsed = parse_capability_package(bundle.capability_package)
    assert parsed.manifest.capability_id == "model.language"
    assert parsed.manifest.capability_version == 7
    assert parsed.manifest.kind == "weights"
    assert len(parsed.sections) == 1
    assert parsed.sections[0] == bundle.model_image
    assert parsed.manifest.sections[0]["role"] == "model_image"
    assert parsed.manifest.sections[0]["format"] == "VN97MI1"

    envelope = parse_signature_envelope(bundle.signature_envelope)
    assert envelope.key_id == "publisher.main"
    assert envelope.package_sha256 == bundle.package_sha256
    assert envelope.capability_id == "model.language"
    assert envelope.capability_version == 7
    assert len(envelope.signature) == 64

    output = write_bootstrap_assets(
        bundle,
        tmp_path / "vn97-bootstrap",
    )
    assert (output / "model.vn97cap1").read_bytes() == bundle.capability_package
    assert (output / "model.vn97sig1").read_bytes() == bundle.signature_envelope
    assert (output / "publisher.ed25519").read_bytes() == b"p" * 32

    # Rewriting the exact same bundle is deterministic/idempotent.
    write_bootstrap_assets(bundle, output)
    assert (output / "model.vn97cap1").read_bytes() == bundle.capability_package


def test_bootstrap_builder_rejects_invalid_signer_output():
    tokenizer = VN97TokenizerPackage()
    with pytest.raises(VN97BootstrapBundleError, match="64"):
        build_bootstrap_bundle(
            _model(tokenizer),
            tokenizer=tokenizer,
            source=_source(),
            capability_version=1,
            signer=BadSigner(),
        )


def test_bootstrap_writer_rejects_symlink_target(tmp_path):
    tokenizer = VN97TokenizerPackage()
    bundle = build_bootstrap_bundle(
        _model(tokenizer),
        tokenizer=tokenizer,
        source=_source(),
        capability_version=1,
        signer=FakeSigner(),
    )
    output = tmp_path / "vn97-bootstrap"
    output.mkdir()
    victim = tmp_path / "victim"
    victim.write_bytes(b"x")
    (output / "model.vn97cap1").symlink_to(victim)

    with pytest.raises(VN97BootstrapBundleError, match="symlink"):
        write_bootstrap_assets(bundle, output)
