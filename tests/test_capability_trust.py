import hashlib
import json
from pathlib import Path

import pytest

from vn97.capability_package import (
    CapabilityDataSection,
    CapabilitySource,
    CapabilityStager,
    build_capability_package,
    parse_capability_package,
)
from vn97.capability_trust import (
    AdapterSpec,
    CapabilityTrustStore,
    CompatibilityAmbiguityError,
    CompatibilityDisposition,
    CompatibilityError,
    CompatibilityProfile,
    Ed25519Verifier,
    FormatRule,
    LossyAdaptationDeniedError,
    SignatureEnvelope,
    SignatureEnvelopeError,
    TrustedPublisherKey,
    UntrustedCapabilityError,
    plan_capability_compatibility,
    parse_signature_envelope,
    verify_staged_capability,
)


class FakeVerifier:
    def verify(self, *, public_key: bytes, message: bytes, signature: bytes) -> bool:
        return signature == hashlib.sha512(public_key + message).digest()


PUBLIC_KEY = b"k" * 32


def make_blob(*, fmt: str = "VN97T2", version: int = 1) -> bytes:
    return build_capability_package(
        capability_id="vision.edge",
        capability_version=version,
        kind="multimodal",
        source=CapabilitySource("local-import", hashlib.sha256(b"source").hexdigest(), "test-only"),
        sections=(
            CapabilityDataSection("weights", fmt, b"weights-data"),
            CapabilityDataSection("metadata", "VN97META1", b"meta-data"),
        ),
    )


def stage(root: Path, blob: bytes):
    staged = CapabilityStager(root, max_package_bytes=1024 * 1024).stage(blob)
    return staged, parse_capability_package(blob, max_package_bytes=1024 * 1024)


def envelope(parsed) -> bytes:
    unsigned = SignatureEnvelope(
        "publisher.main",
        parsed.package_sha256,
        parsed.manifest.capability_id,
        parsed.manifest.capability_version,
        bytes(64),
    )
    signature = hashlib.sha512(PUBLIC_KEY + unsigned.signing_message()).digest()
    return SignatureEnvelope(
        "publisher.main",
        parsed.package_sha256,
        parsed.manifest.capability_id,
        parsed.manifest.capability_version,
        signature,
    ).to_bytes()


def trust_store(*, revoked=False, prefix="vision", min_version=1, max_version=5):
    return CapabilityTrustStore((
        TrustedPublisherKey(
            "publisher.main",
            PUBLIC_KEY,
            (prefix,),
            frozenset({"multimodal"}),
            min_version,
            max_version,
            revoked,
        ),
    ))


def verified(root: Path, *, fmt="VN97T2", version=1):
    staged, parsed = stage(root, make_blob(fmt=fmt, version=version))
    return verify_staged_capability(
        staged,
        envelope(parsed),
        stage_root=root,
        trust_store=trust_store(),
        verifier=FakeVerifier(),
        max_package_bytes=1024 * 1024,
    )


def profile():
    return CompatibilityProfile(
        "mobile.v1",
        1,
        frozenset({"multimodal"}),
        1,
        3,
        (
            FormatRule("weights", ("VN97T2",)),
            FormatRule("metadata", ("VN97META1",)),
        ),
    )


def test_signature_verification_and_staged_revalidation(tmp_path):
    root = tmp_path.resolve()
    staged, parsed = stage(root, make_blob())
    result = verify_staged_capability(
        staged, envelope(parsed), stage_root=root, trust_store=trust_store(),
        verifier=FakeVerifier(), max_package_bytes=1024 * 1024,
    )
    assert result.publisher_key_id == "publisher.main"
    assert result.parsed.package_sha256 == parsed.package_sha256
    staged.path.write_bytes(b"{}")
    with pytest.raises(UntrustedCapabilityError):
        verify_staged_capability(
            staged, envelope(parsed), stage_root=root, trust_store=trust_store(),
            verifier=FakeVerifier(), max_package_bytes=1024 * 1024,
        )


def test_signature_binding_and_canonical_envelope(tmp_path):
    root = tmp_path.resolve()
    staged, parsed = stage(root, make_blob())
    raw = envelope(parsed)
    assert parse_signature_envelope(raw).package_sha256 == parsed.package_sha256
    with pytest.raises(SignatureEnvelopeError):
        parse_signature_envelope(json.dumps(json.loads(raw), indent=2).encode())
    value = json.loads(raw)
    value["package_sha256"] = "f" * 64
    rebound = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(UntrustedCapabilityError):
        verify_staged_capability(
            staged, rebound, stage_root=root, trust_store=trust_store(),
            verifier=FakeVerifier(), max_package_bytes=1024 * 1024,
        )


def test_trust_scope_revocation_version_and_segment_boundary(tmp_path):
    root = tmp_path.resolve()
    staged, parsed = stage(root, make_blob(version=2))
    for store in (
        trust_store(revoked=True),
        trust_store(prefix="audio"),
        trust_store(min_version=3, max_version=4),
        trust_store(prefix="vis"),
    ):
        with pytest.raises(UntrustedCapabilityError):
            verify_staged_capability(
                staged, envelope(parsed), stage_root=root, trust_store=store,
                verifier=FakeVerifier(), max_package_bytes=1024 * 1024,
            )


def test_direct_and_adapt_plans_are_deterministic_and_bound_to_trust(tmp_path):
    root = tmp_path.resolve()
    direct = verified(root)
    direct_plan = plan_capability_compatibility(direct, profile())
    assert direct_plan.disposition is CompatibilityDisposition.DIRECT
    assert direct_plan.publisher_key_id == direct.publisher_key_id
    assert direct_plan.signature_sha256 == direct.signature_sha256
    assert direct_plan.profile_sha256 == profile().fingerprint()

    root2 = (tmp_path / "other").resolve()
    root2.mkdir()
    adapted = verified(root2, fmt="FP16")
    adapters = (AdapterSpec("weights.fp16-to-vn97t2", "weights", "FP16", "VN97T2"),)
    first = plan_capability_compatibility(adapted, profile(), adapters=adapters)
    second = plan_capability_compatibility(adapted, profile(), adapters=adapters)
    assert first == second
    assert first.disposition is CompatibilityDisposition.ADAPT_REQUIRED
    assert first.sections[0].adapter_id == "weights.fp16-to-vn97t2"


def test_lossy_adaptation_denied_by_default(tmp_path):
    root = tmp_path.resolve()
    value = verified(root, fmt="FP16")
    lossy = (AdapterSpec("weights.lossy", "weights", "FP16", "VN97T2", True),)
    with pytest.raises(LossyAdaptationDeniedError):
        plan_capability_compatibility(value, profile(), adapters=lossy)
    allowed = plan_capability_compatibility(value, profile(), adapters=lossy, allow_lossy=True)
    assert allowed.sections[0].lossy is True


def test_ambiguous_and_profile_incompatible_paths_fail_closed(tmp_path):
    root = tmp_path.resolve()
    value = verified(root, fmt="FP16")
    adapters = (
        AdapterSpec("weights.a", "weights", "FP16", "VN97T2"),
        AdapterSpec("weights.b", "weights", "FP16", "VN97T2"),
    )
    with pytest.raises(CompatibilityAmbiguityError):
        plan_capability_compatibility(value, profile(), adapters=adapters)
    incompatible = CompatibilityProfile(
        "mobile.v2", 1, frozenset({"multimodal"}), 2, 3,
        (FormatRule("weights", ("VN97T2",)), FormatRule("metadata", ("VN97META1",))),
    )
    with pytest.raises(CompatibilityError):
        plan_capability_compatibility(value, incompatible, adapters=adapters)


def test_envelope_trust_and_profile_inputs_are_bounded():
    with pytest.raises(ValueError):
        SignatureEnvelope("Bad Key", "0" * 64, "vision.edge", 1, bytes(64))
    with pytest.raises(ValueError):
        TrustedPublisherKey("key", b"x", ("vision",), frozenset({"multimodal"}))
    with pytest.raises(ValueError):
        CapabilityTrustStore((
            TrustedPublisherKey("same", PUBLIC_KEY, ("vision",), frozenset({"multimodal"})),
            TrustedPublisherKey("same", PUBLIC_KEY, ("vision",), frozenset({"multimodal"})),
        ))
    with pytest.raises(ValueError):
        CompatibilityProfile("p", 1, frozenset({"multimodal"}), 1, 1, ())


def test_ed25519_adapter_when_backend_available():
    crypto = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    serialization = pytest.importorskip("cryptography.hazmat.primitives.serialization")
    private = crypto.Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    unsigned = SignatureEnvelope("publisher.main", "0" * 64, "vision.edge", 1, bytes(64))
    message = unsigned.signing_message()
    signature = private.sign(message)
    verifier = Ed25519Verifier()
    assert verifier.verify(public_key=public, message=message, signature=signature)
    assert not verifier.verify(public_key=public, message=message + b"x", signature=signature)
