from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path
import secrets
from typing import Protocol

from .capability_inventory import (
    MAX_HISTORY,
    MAX_STACK_DEPTH,
    CapabilityActivationError,
    CapabilityInventoryItem,
    CapabilityInventoryStore,
    InventoryCorruptionError,
    InventoryEvent,
    InventorySnapshot,
    _ID_RE,
    _REV_RE,
    _StoredActivation,
    _canon,
    _require_id,
    _require_sha,
    _validate_item,
    _validate_plan_identity,
)
from .capability_trust import (
    AdapterSpec,
    CapabilityTrustStore,
    CompatibilityPlan,
    CompatibilityProfile,
    SignatureVerifier,
    VerifiedCapability,
    plan_capability_compatibility,
    verify_staged_capability,
)


class ActivationConflictError(CapabilityActivationError):
    pass


class ActivationRecoveryRequired(CapabilityActivationError):
    pass


class VersionTransitionError(CapabilityActivationError):
    pass


class BackendTransactionState(str, Enum):
    PREPARED = "prepared"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True)
class BackendStatus:
    state: BackendTransactionState
    runtime_revision: str | None = None

    def __post_init__(self) -> None:
        if self.state is BackendTransactionState.COMMITTED:
            if self.runtime_revision is None or not _REV_RE.fullmatch(self.runtime_revision):
                raise ValueError("committed status requires valid runtime_revision")
        elif self.runtime_revision is not None:
            raise ValueError("non-committed status must not contain runtime_revision")


@dataclass(frozen=True)
class PreparedActivation:
    backend_id: str
    token: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        from .capability_inventory import _SHA_RE, _TOKEN_RE

        if not _ID_RE.fullmatch(self.backend_id):
            raise ValueError("backend_id is invalid")
        if not _TOKEN_RE.fullmatch(self.token):
            raise ValueError("backend token is invalid")
        if not _SHA_RE.fullmatch(self.artifact_sha256):
            raise ValueError("artifact_sha256 is invalid")


class CapabilityActivationBackend(Protocol):
    @property
    def backend_id(self) -> str: ...

    def prepare(
        self,
        verified: VerifiedCapability,
        plan: CompatibilityPlan,
    ) -> PreparedActivation: ...

    def commit(self, token: str) -> str: ...

    def inspect(self, token: str) -> BackendStatus: ...

    def rollback(self, token: str) -> None: ...


def _plan_obj(plan: CompatibilityPlan) -> dict[str, object]:
    return {
        "capability_id": plan.capability_id,
        "capability_version": plan.capability_version,
        "disposition": plan.disposition.value,
        "package_sha256": plan.package_sha256,
        "profile_id": plan.profile_id,
        "profile_sha256": plan.profile_sha256,
        "publisher_key_id": plan.publisher_key_id,
        "runtime_api_version": plan.runtime_api_version,
        "sections": [
            {
                "adapter_id": section.adapter_id,
                "lossy": section.lossy,
                "role": section.role,
                "section_index": section.section_index,
                "source_format": section.source_format,
                "target_format": section.target_format,
            }
            for section in plan.sections
        ],
        "signature_sha256": plan.signature_sha256,
    }


def compatibility_plan_sha256(plan: CompatibilityPlan) -> str:
    return hashlib.sha256(_canon(_plan_obj(plan))).hexdigest()


def _pending_plan_identity(
    plan: CompatibilityPlan,
    verified: VerifiedCapability,
) -> dict[str, object]:
    source = verified.parsed.manifest.source
    return {
        "capability_id": plan.capability_id,
        "capability_version": plan.capability_version,
        "package_sha256": plan.package_sha256,
        "plan_sha256": compatibility_plan_sha256(plan),
        "profile_id": plan.profile_id,
        "profile_sha256": plan.profile_sha256,
        "publisher_key_id": plan.publisher_key_id,
        "runtime_api_version": plan.runtime_api_version,
        "signature_sha256": plan.signature_sha256,
        "source_license": source.license,
        "source_origin": source.origin,
        "source_sha256": source.source_sha256,
    }


def _activation_id(
    plan_identity: dict[str, object],
    backend_id: str,
    artifact_sha256: str,
    revision: str,
) -> str:
    return hashlib.sha256(
        _canon(
            {
                **plan_identity,
                "artifact_sha256": artifact_sha256,
                "backend_id": backend_id,
                "runtime_revision": revision,
            }
        )
    ).hexdigest()


class CapabilityActivationCoordinator:
    def __init__(self, store: CapabilityInventoryStore) -> None:
        self.store = store

    def inventory(self) -> InventorySnapshot:
        return self.store.load()

    def activate(
        self,
        verified: VerifiedCapability,
        signature_envelope: bytes,
        plan: CompatibilityPlan,
        profile: CompatibilityProfile,
        backend: CapabilityActivationBackend,
        *,
        stage_root: Path,
        trust_store: CapabilityTrustStore,
        verifier: SignatureVerifier,
        adapters: tuple[AdapterSpec, ...] = (),
        allow_lossy: bool = False,
        allow_same_version_replace: bool = False,
        allow_downgrade: bool = False,
    ) -> CapabilityInventoryItem:
        fresh = verify_staged_capability(
            verified.staged,
            signature_envelope,
            stage_root=stage_root,
            trust_store=trust_store,
            verifier=verifier,
        )
        if fresh != verified:
            raise ActivationConflictError("verified capability changed before activation")
        expected = plan_capability_compatibility(
            fresh,
            profile,
            adapters=adapters,
            allow_lossy=allow_lossy,
        )
        if plan != expected:
            raise ActivationConflictError("compatibility plan is stale or mismatched")
        if not _ID_RE.fullmatch(backend.backend_id):
            raise ValueError("backend_id is invalid")
        identity = _pending_plan_identity(plan, fresh)

        with self.store.locked() as root_fd:
            state = self.store._load(root_fd)
            if state.pending is not None:
                raise ActivationRecoveryRequired("inventory has unresolved transaction")
            stack = state.stacks.get(plan.capability_id, [])
            current = stack[-1].item if stack else None
            if current is not None:
                exact = (
                    current.package_sha256 == plan.package_sha256
                    and current.signature_sha256 == plan.signature_sha256
                    and current.profile_sha256 == plan.profile_sha256
                    and current.plan_sha256 == identity["plan_sha256"]
                    and current.backend_id == backend.backend_id
                )
                if exact:
                    return current
                if len(stack) >= MAX_STACK_DEPTH:
                    raise VersionTransitionError("activation stack depth limit reached")
                if plan.capability_version < current.capability_version and not allow_downgrade:
                    raise VersionTransitionError("capability downgrade is denied")
                if (
                    plan.capability_version == current.capability_version
                    and not allow_same_version_replace
                ):
                    raise VersionTransitionError("same-version replacement is denied")

            tx_id = hashlib.sha256(
                _canon(
                    {
                        "generation": state.generation,
                        "identity": identity,
                        "nonce": secrets.token_hex(16),
                        "operation": "activate",
                    }
                )
            ).hexdigest()
            state.pending = {
                "activation_id": None,
                "artifact_sha256": None,
                "backend_id": backend.backend_id,
                "backend_token": None,
                "capability_id": plan.capability_id,
                "generation_base": state.generation,
                "identity": identity,
                "operation": "activate",
                "phase": "reserved",
                "tx_id": tx_id,
            }
            self.store._write(root_fd, state)

        try:
            prepared = backend.prepare(fresh, plan)
        except Exception:
            self._clear_reserved(tx_id)
            raise
        if prepared.backend_id != backend.backend_id:
            self._clear_reserved(tx_id)
            raise ActivationConflictError("backend prepare returned mismatched backend_id")

        with self.store.locked() as root_fd:
            state = self.store._load(root_fd)
            pending = state.pending
            if not pending or pending["tx_id"] != tx_id or pending["phase"] != "reserved":
                try:
                    backend.rollback(prepared.token)
                finally:
                    raise ActivationRecoveryRequired(
                        "activation reservation changed before commit"
                    )
            pending = dict(pending)
            pending.update(
                {
                    "phase": "prepared",
                    "backend_token": prepared.token,
                    "artifact_sha256": prepared.artifact_sha256,
                }
            )
            state.pending = pending
            self.store._write(root_fd, state)

        try:
            revision = backend.commit(prepared.token)
            if not isinstance(revision, str) or not _REV_RE.fullmatch(revision):
                raise CapabilityActivationError("backend returned invalid runtime_revision")
        except Exception:
            try:
                backend.rollback(prepared.token)
            except Exception as rollback_exc:
                raise ActivationRecoveryRequired(
                    "activation needs recovery after commit failure"
                ) from rollback_exc
            self._clear_prepared(tx_id, prepared.token)
            raise
        return self._finalize_activation(tx_id, revision, "activate")

    def rollback(
        self,
        capability_id: str,
        backend: CapabilityActivationBackend,
    ) -> CapabilityInventoryItem | None:
        _require_id(capability_id, "capability_id")
        with self.store.locked() as root_fd:
            state = self.store._load(root_fd)
            if state.pending is not None:
                raise ActivationRecoveryRequired("inventory has unresolved transaction")
            stack = state.stacks.get(capability_id)
            if not stack:
                raise CapabilityActivationError("capability is not active")
            current = stack[-1]
            if current.item.backend_id != backend.backend_id:
                raise ActivationConflictError("rollback backend mismatch")
            tx_id = hashlib.sha256(
                _canon(
                    {
                        "activation_id": current.item.activation_id,
                        "generation": state.generation,
                        "nonce": secrets.token_hex(16),
                        "operation": "rollback",
                    }
                )
            ).hexdigest()
            state.pending = {
                "activation_id": current.item.activation_id,
                "artifact_sha256": current.item.artifact_sha256,
                "backend_id": backend.backend_id,
                "backend_token": current.backend_token,
                "capability_id": capability_id,
                "generation_base": state.generation,
                "identity": {},
                "operation": "rollback",
                "phase": "prepared",
                "tx_id": tx_id,
            }
            self.store._write(root_fd, state)
        try:
            backend.rollback(current.backend_token)
        except Exception as exc:
            raise ActivationRecoveryRequired("rollback requires recovery") from exc
        return self._finalize_rollback(tx_id, "rollback")

    def recover(
        self,
        backends: dict[str, CapabilityActivationBackend],
    ) -> InventorySnapshot:
        with self.store.locked() as root_fd:
            state = self.store._load(root_fd)
            pending = state.pending
            if pending is None:
                return state.snapshot()
            if pending["phase"] == "reserved":
                state.pending = None
                self.store._write(root_fd, state)
                return state.snapshot()
            backend = backends.get(pending["backend_id"])
            if backend is None or backend.backend_id != pending["backend_id"]:
                raise ActivationRecoveryRequired("required backend is unavailable")
            token = pending["backend_token"]
            tx_id = pending["tx_id"]
            operation = pending["operation"]

        status = backend.inspect(token)
        if operation == "activate":
            if status.state is BackendTransactionState.COMMITTED:
                self._finalize_activation(
                    tx_id,
                    status.runtime_revision,
                    "recover_activate",
                )
            elif status.state in {
                BackendTransactionState.PREPARED,
                BackendTransactionState.ROLLED_BACK,
            }:
                if status.state is BackendTransactionState.PREPARED:
                    try:
                        backend.rollback(token)
                    except Exception as exc:
                        raise ActivationRecoveryRequired(
                            "prepared activation rollback failed"
                        ) from exc
                self._clear_prepared(tx_id, token)
            else:
                raise ActivationRecoveryRequired("unsupported activation recovery state")
        else:
            if status.state is not BackendTransactionState.ROLLED_BACK:
                try:
                    backend.rollback(token)
                except Exception as exc:
                    raise ActivationRecoveryRequired("rollback recovery failed") from exc
            self._finalize_rollback(tx_id, "recover_rollback")
        return self.inventory()

    def _clear_reserved(self, tx_id: str) -> None:
        with self.store.locked() as root_fd:
            state = self.store._load(root_fd)
            if (
                state.pending
                and state.pending["tx_id"] == tx_id
                and state.pending["phase"] == "reserved"
            ):
                state.pending = None
                self.store._write(root_fd, state)

    def _clear_prepared(self, tx_id: str, token: str) -> None:
        with self.store.locked() as root_fd:
            state = self.store._load(root_fd)
            if (
                state.pending
                and state.pending["tx_id"] == tx_id
                and state.pending["backend_token"] == token
            ):
                state.pending = None
                self.store._write(root_fd, state)

    def _finalize_activation(
        self,
        tx_id: str,
        revision: str | None,
        action: str,
    ) -> CapabilityInventoryItem:
        if revision is None or not _REV_RE.fullmatch(revision):
            raise ActivationRecoveryRequired(
                "committed activation lacks valid runtime_revision"
            )
        with self.store.locked() as root_fd:
            state = self.store._load(root_fd)
            pending = state.pending
            if (
                not pending
                or pending["tx_id"] != tx_id
                or pending["operation"] != "activate"
                or pending["phase"] != "prepared"
            ):
                raise ActivationRecoveryRequired(
                    "activation transaction changed before finalization"
                )
            identity = _validate_plan_identity(
                pending["identity"],
                expected_capability_id=pending["capability_id"],
            )
            artifact = _require_sha(
                pending["artifact_sha256"],
                "pending artifact_sha256",
            )
            token = pending["backend_token"]
            activation_id = _activation_id(
                identity,
                pending["backend_id"],
                artifact,
                revision,
            )
            item = CapabilityInventoryItem(
                activation_id=activation_id,
                capability_id=identity["capability_id"],
                capability_version=identity["capability_version"],
                package_sha256=identity["package_sha256"],
                publisher_key_id=identity["publisher_key_id"],
                signature_sha256=identity["signature_sha256"],
                profile_id=identity["profile_id"],
                profile_sha256=identity["profile_sha256"],
                plan_sha256=identity["plan_sha256"],
                runtime_api_version=identity["runtime_api_version"],
                backend_id=pending["backend_id"],
                artifact_sha256=artifact,
                runtime_revision=revision,
                source_origin=identity["source_origin"],
                source_sha256=identity["source_sha256"],
                source_license=identity["source_license"],
            )
            _validate_item(item)
            stack = state.stacks.setdefault(item.capability_id, [])
            if len(stack) >= MAX_STACK_DEPTH:
                raise ActivationRecoveryRequired("activation stack became full")
            stack.append(_StoredActivation(item, token))
            state.generation += 1
            state.history.append(
                InventoryEvent(
                    generation=state.generation,
                    action=action,
                    capability_id=item.capability_id,
                    activation_id=item.activation_id,
                    package_sha256=item.package_sha256,
                )
            )
            state.history = state.history[-MAX_HISTORY:]
            state.pending = None
            self.store._write(root_fd, state)
            return item

    def _finalize_rollback(
        self,
        tx_id: str,
        action: str,
    ) -> CapabilityInventoryItem | None:
        with self.store.locked() as root_fd:
            state = self.store._load(root_fd)
            pending = state.pending
            if (
                not pending
                or pending["tx_id"] != tx_id
                or pending["operation"] != "rollback"
            ):
                raise ActivationRecoveryRequired(
                    "rollback transaction changed before finalization"
                )
            stack = state.stacks.get(pending["capability_id"])
            if not stack or stack[-1].item.activation_id != pending["activation_id"]:
                raise ActivationConflictError("active capability changed during rollback")
            removed = stack.pop()
            if not stack:
                state.stacks.pop(removed.item.capability_id, None)
            state.generation += 1
            state.history.append(
                InventoryEvent(
                    generation=state.generation,
                    action=action,
                    capability_id=removed.item.capability_id,
                    activation_id=removed.item.activation_id,
                    package_sha256=removed.item.package_sha256,
                )
            )
            state.history = state.history[-MAX_HISTORY:]
            state.pending = None
            self.store._write(root_fd, state)
            return stack[-1].item if stack else None
