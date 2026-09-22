from dataclasses import dataclass, replace
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
    CapabilityTrustStore,
    CompatibilityProfile,
    FormatRule,
    SignatureEnvelope,
    TrustedPublisherKey,
    UntrustedCapabilityError,
    plan_capability_compatibility,
    verify_staged_capability,
)
from vn97.capability_activation import (
    ActivationConflictError,
    ActivationRecoveryRequired,
    BackendStatus,
    BackendTransactionState,
    CapabilityActivationCoordinator,
    CapabilityInventoryStore,
    InventoryCorruptionError,
    PreparedActivation,
    VersionTransitionError,
)

PUBLIC_KEY = b"k" * 32


class FakeVerifier:
    def verify(self, *, public_key: bytes, message: bytes, signature: bytes) -> bool:
        return signature == hashlib.sha512(public_key + message).digest()


class Crash(BaseException):
    pass


class FakeBackend:
    backend_id = "model.data"

    def __init__(self):
        self.records = {}
        self.live_revision = None
        self.prepare_calls = 0
        self.commit_calls = 0
        self.rollback_calls = 0
        self.crash_prepare = False
        self.crash_commit_after = False
        self.crash_rollback_after = False

    def prepare(self, verified, plan):
        self.prepare_calls += 1
        if self.crash_prepare:
            raise Crash("prepare crash")
        token = f"tx.{self.prepare_calls}"
        artifact = hashlib.sha256(
            (verified.parsed.package_sha256 + plan.profile_sha256).encode()
        ).hexdigest()
        self.records[token] = {
            "state": BackendTransactionState.PREPARED,
            "prior": self.live_revision,
            "revision": None,
            "artifact": artifact,
        }
        return PreparedActivation(self.backend_id, token, artifact)

    def commit(self, token):
        self.commit_calls += 1
        record = self.records[token]
        revision = f"rev.{self.commit_calls}"
        record["state"] = BackendTransactionState.COMMITTED
        record["revision"] = revision
        self.live_revision = revision
        if self.crash_commit_after:
            raise Crash("commit crash after durable backend commit")
        return revision

    def inspect(self, token):
        record = self.records[token]
        revision = (
            record["revision"]
            if record["state"] is BackendTransactionState.COMMITTED
            else None
        )
        return BackendStatus(record["state"], revision)

    def rollback(self, token):
        self.rollback_calls += 1
        record = self.records[token]
        record["state"] = BackendTransactionState.ROLLED_BACK
        record["revision"] = None
        self.live_revision = record["prior"]
        if self.crash_rollback_after:
            raise Crash("rollback crash after backend rollback")


@dataclass(frozen=True)
class Evidence:
    verified: object
    envelope: bytes


def profile():
    return CompatibilityProfile(
        "mobile.v1",
        1,
        frozenset({"multimodal"}),
        1,
        10,
        (
            FormatRule("weights", ("VN97T2",)),
            FormatRule("metadata", ("VN97META1",)),
        ),
    )


def trust_store(*, revoked=False):
    return CapabilityTrustStore(
        (
            TrustedPublisherKey(
                "publisher.main",
                PUBLIC_KEY,
                ("vision",),
                frozenset({"multimodal"}),
                1,
                10,
                revoked,
            ),
        )
    )


def make_evidence(stage_root: Path, *, version: int, payload: bytes):
    blob = build_capability_package(
        capability_id="vision.edge",
        capability_version=version,
        kind="multimodal",
        source=CapabilitySource(
            "local-import",
            hashlib.sha256(b"source" + bytes([version])).hexdigest(),
            "test-only",
        ),
        sections=(
            CapabilityDataSection("weights", "VN97T2", payload),
            CapabilityDataSection("metadata", "VN97META1", b"meta" + bytes([version])),
        ),
    )
    staged = CapabilityStager(
        stage_root,
        max_package_bytes=1024 * 1024,
    ).stage(blob)
    parsed = parse_capability_package(
        blob,
        max_package_bytes=1024 * 1024,
    )
    unsigned = SignatureEnvelope(
        "publisher.main",
        parsed.package_sha256,
        "vision.edge",
        version,
        bytes(64),
    )
    signed = SignatureEnvelope(
        "publisher.main",
        parsed.package_sha256,
        "vision.edge",
        version,
        hashlib.sha512(PUBLIC_KEY + unsigned.signing_message()).digest(),
    ).to_bytes()
    verified = verify_staged_capability(
        staged,
        signed,
        stage_root=stage_root,
        trust_store=trust_store(),
        verifier=FakeVerifier(),
        max_package_bytes=1024 * 1024,
    )
    return Evidence(verified, signed)


def make_plan(evidence):
    return plan_capability_compatibility(evidence.verified, profile())


def activate(coordinator, evidence, selected_plan, backend, *, store=None, **kwargs):
    return coordinator.activate(
        evidence.verified,
        evidence.envelope,
        selected_plan,
        profile(),
        backend,
        stage_root=evidence.verified.staged.path.parent,
        trust_store=trust_store() if store is None else store,
        verifier=FakeVerifier(),
        **kwargs,
    )


def roots(tmp_path):
    stage = (tmp_path / "stage").resolve()
    stage.mkdir()
    inventory = (tmp_path / "inventory").resolve()
    inventory.mkdir()
    return stage, inventory


def test_activate_idempotent_upgrade_and_rollback_restores_previous(tmp_path):
    stage_root, inventory_root = roots(tmp_path)
    coordinator = CapabilityActivationCoordinator(CapabilityInventoryStore(inventory_root))
    backend = FakeBackend()

    v1 = make_evidence(stage_root, version=1, payload=b"v1")
    p1 = make_plan(v1)
    a1 = activate(coordinator, v1, p1, backend)
    assert a1.capability_version == 1
    assert backend.live_revision == "rev.1"

    again = activate(coordinator, v1, p1, backend)
    assert again == a1
    assert backend.prepare_calls == 1

    v2 = make_evidence(stage_root, version=2, payload=b"v2")
    a2 = activate(coordinator, v2, make_plan(v2), backend)
    assert a2.capability_version == 2
    assert backend.live_revision == "rev.2"
    assert coordinator.inventory().current("vision.edge") == a2

    restored = coordinator.rollback("vision.edge", backend)
    assert restored == a1
    assert coordinator.inventory().current("vision.edge") == a1
    assert backend.live_revision == "rev.1"

    empty = coordinator.rollback("vision.edge", backend)
    assert empty is None
    assert coordinator.inventory().current("vision.edge") is None
    assert backend.live_revision is None


def test_stale_plan_same_version_and_downgrade_fail_before_prepare(tmp_path):
    stage_root, inventory_root = roots(tmp_path)
    coordinator = CapabilityActivationCoordinator(CapabilityInventoryStore(inventory_root))
    backend = FakeBackend()

    v1 = make_evidence(stage_root, version=1, payload=b"v1")
    p1 = make_plan(v1)
    stale = replace(p1, profile_sha256="f" * 64)
    with pytest.raises(ActivationConflictError):
        activate(coordinator, v1, stale, backend)
    assert backend.prepare_calls == 0

    activate(coordinator, v1, p1, backend)
    same = make_evidence(stage_root, version=1, payload=b"same-version-new-bytes")
    with pytest.raises(VersionTransitionError):
        activate(coordinator, same, make_plan(same), backend)
    assert backend.prepare_calls == 1

    v2 = make_evidence(stage_root, version=2, payload=b"v2")
    activate(coordinator, v2, make_plan(v2), backend)
    with pytest.raises(VersionTransitionError):
        activate(coordinator, v1, p1, backend)
    assert backend.prepare_calls == 2


def test_activation_revalidates_current_trust_and_staged_bytes(tmp_path):
    stage_root, inventory_root = roots(tmp_path)
    coordinator = CapabilityActivationCoordinator(CapabilityInventoryStore(inventory_root))
    backend = FakeBackend()
    value = make_evidence(stage_root, version=1, payload=b"v1")
    selected = make_plan(value)

    with pytest.raises(UntrustedCapabilityError):
        activate(
            coordinator,
            value,
            selected,
            backend,
            store=trust_store(revoked=True),
        )
    assert backend.prepare_calls == 0

    value.verified.staged.path.write_bytes(b"tampered")
    with pytest.raises(UntrustedCapabilityError):
        activate(coordinator, value, selected, backend)
    assert backend.prepare_calls == 0


def test_reserved_crash_recovery_clears_without_activation(tmp_path):
    stage_root, inventory_root = roots(tmp_path)
    coordinator = CapabilityActivationCoordinator(CapabilityInventoryStore(inventory_root))
    backend = FakeBackend()
    value = make_evidence(stage_root, version=1, payload=b"v1")

    backend.crash_prepare = True
    with pytest.raises(Crash):
        activate(coordinator, value, make_plan(value), backend)

    snapshot = coordinator.inventory()
    assert snapshot.pending_operation == "activate"
    assert snapshot.current("vision.edge") is None

    recovered = coordinator.recover({backend.backend_id: backend})
    assert recovered.pending_operation is None
    assert recovered.current("vision.edge") is None


def test_committed_activation_crash_recovery_finalizes_full_provenance(tmp_path):
    stage_root, inventory_root = roots(tmp_path)
    coordinator = CapabilityActivationCoordinator(CapabilityInventoryStore(inventory_root))
    backend = FakeBackend()
    value = make_evidence(stage_root, version=1, payload=b"v1")
    expected_plan = make_plan(value)

    backend.crash_commit_after = True
    with pytest.raises(Crash):
        activate(coordinator, value, expected_plan, backend)
    assert coordinator.inventory().pending_operation == "activate"
    assert backend.live_revision == "rev.1"

    backend.crash_commit_after = False
    recovered = coordinator.recover({backend.backend_id: backend})
    item = recovered.current("vision.edge")
    assert item is not None
    assert item.package_sha256 == value.verified.parsed.package_sha256
    assert item.publisher_key_id == value.verified.publisher_key_id
    assert item.signature_sha256 == value.verified.signature_sha256
    assert item.profile_sha256 == expected_plan.profile_sha256
    assert item.source_origin == value.verified.parsed.manifest.source.origin
    assert item.source_sha256 == value.verified.parsed.manifest.source.source_sha256
    assert item.source_license == value.verified.parsed.manifest.source.license
    assert recovered.history[-1].action == "recover_activate"


def test_rollback_crash_recovery_pops_and_restores_previous(tmp_path):
    stage_root, inventory_root = roots(tmp_path)
    coordinator = CapabilityActivationCoordinator(CapabilityInventoryStore(inventory_root))
    backend = FakeBackend()

    v1 = make_evidence(stage_root, version=1, payload=b"v1")
    a1 = activate(coordinator, v1, make_plan(v1), backend)
    v2 = make_evidence(stage_root, version=2, payload=b"v2")
    activate(coordinator, v2, make_plan(v2), backend)

    backend.crash_rollback_after = True
    with pytest.raises(Crash):
        coordinator.rollback("vision.edge", backend)
    assert coordinator.inventory().pending_operation == "rollback"
    assert backend.live_revision == "rev.1"

    backend.crash_rollback_after = False
    recovered = coordinator.recover({backend.backend_id: backend})
    assert recovered.current("vision.edge") == a1
    assert recovered.history[-1].action == "recover_rollback"


def test_pending_requires_backend_and_public_snapshot_hides_token(tmp_path):
    stage_root, inventory_root = roots(tmp_path)
    coordinator = CapabilityActivationCoordinator(CapabilityInventoryStore(inventory_root))
    backend = FakeBackend()
    value = make_evidence(stage_root, version=1, payload=b"v1")

    backend.crash_commit_after = True
    with pytest.raises(Crash):
        activate(coordinator, value, make_plan(value), backend)

    with pytest.raises(ActivationRecoveryRequired):
        coordinator.recover({})
    snapshot = coordinator.inventory()
    assert snapshot.pending_operation == "activate"
    assert "tx." not in repr(snapshot)


def test_inventory_tamper_noncanonical_and_symlink_fail_closed(tmp_path):
    stage_root, inventory_root = roots(tmp_path)
    store = CapabilityInventoryStore(inventory_root)
    coordinator = CapabilityActivationCoordinator(store)
    backend = FakeBackend()
    value = make_evidence(stage_root, version=1, payload=b"v1")
    activate(coordinator, value, make_plan(value), backend)

    inventory_path = inventory_root / "inventory.vn97inv1.json"
    obj = json.loads(inventory_path.read_text())
    inventory_path.write_text(json.dumps(obj, indent=2))
    with pytest.raises(InventoryCorruptionError):
        store.load()

    inventory_path.unlink()
    target = inventory_root / "target.json"
    target.write_text("{}")
    inventory_path.symlink_to(target)
    with pytest.raises(InventoryCorruptionError):
        store.load()


def test_pending_identity_tamper_is_detected_on_load(tmp_path):
    stage_root, inventory_root = roots(tmp_path)
    coordinator = CapabilityActivationCoordinator(CapabilityInventoryStore(inventory_root))
    backend = FakeBackend()
    value = make_evidence(stage_root, version=1, payload=b"v1")

    backend.crash_commit_after = True
    with pytest.raises(Crash):
        activate(coordinator, value, make_plan(value), backend)

    path = inventory_root / "inventory.vn97inv1.json"
    obj = json.loads(path.read_text())
    obj["pending"]["identity"]["package_sha256"] = "not-a-sha"
    path.write_text(json.dumps(obj, sort_keys=True, separators=(",", ":")))
    with pytest.raises(InventoryCorruptionError):
        coordinator.inventory()
