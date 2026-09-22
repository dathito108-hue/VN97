from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import secrets
import time
from typing import Callable, Iterable, Mapping, Protocol


class AuthorityError(RuntimeError):
    pass


class CapabilityNotFoundError(AuthorityError):
    pass


class AuthorizationDeniedError(AuthorityError):
    pass


class ApprovalRequiredError(AuthorityError):
    pass


class InvalidApprovalError(AuthorityError):
    pass


class LeaseError(AuthorityError):
    pass


class AuditIntegrityError(AuthorityError):
    pass


class ExternalExecutionError(AuthorityError):
    pass


class ReceiptStatus(IntEnum):
    DENIED = 1
    SUCCEEDED = 2
    FAILED = 3


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _strict_json_object(text: str) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise ValueError(f"invalid JSON constant: {value}")

    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    value = json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=pairs_hook,
    )
    if not isinstance(value, dict):
        raise ValueError("payload JSON root must be an object")
    return value


def _sha256_hex(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _valid_identifier(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    if len(value.encode("utf-8")) > 256:
        raise ValueError(f"{label} is too long")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-/")
    if any(char not in allowed for char in value):
        raise ValueError(f"{label} contains unsupported characters")
    return value


@dataclass(frozen=True)
class CapabilityScope:
    entries: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        normalized = tuple((str(key), str(value)) for key, value in self.entries)
        if not normalized:
            raise ValueError("capability scope must not be empty")
        keys = [key for key, _ in normalized]
        if len(set(keys)) != len(keys):
            raise ValueError("capability scope keys must be unique")
        for key, value in normalized:
            _valid_identifier(key, label="scope key")
            if not value or len(value.encode("utf-8")) > 4096:
                raise ValueError("scope values must be non-empty and bounded")
        object.__setattr__(self, "entries", tuple(sorted(normalized)))

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "CapabilityScope":
        return cls(tuple((str(key), str(value)) for key, value in values.items()))

    @property
    def keys(self) -> frozenset[str]:
        return frozenset(key for key, _ in self.entries)

    @property
    def digest(self) -> str:
        return _sha256_hex({"scope": list(self.entries)})

    def as_dict(self) -> dict[str, str]:
        return dict(self.entries)


@dataclass(frozen=True)
class ExternalActionRequest:
    plan_id: str
    step_id: int
    objective: str
    capability_id: str
    scope: CapabilityScope
    payload_json: str = "{}"

    def __post_init__(self) -> None:
        if not self.plan_id:
            raise ValueError("plan_id must not be empty")
        if self.step_id <= 0:
            raise ValueError("step_id must be positive")
        if not self.objective:
            raise ValueError("objective must not be empty")
        _valid_identifier(self.capability_id, label="capability_id")
        payload = _strict_json_object(self.payload_json)
        canonical = _canonical_json_bytes(payload).decode("utf-8")
        object.__setattr__(self, "payload_json", canonical)

    @classmethod
    def create(
        cls,
        *,
        plan_id: str,
        step_id: int,
        objective: str,
        capability_id: str,
        scope: CapabilityScope,
        payload: Mapping[str, object] | None = None,
    ) -> "ExternalActionRequest":
        payload_json = _canonical_json_bytes(dict(payload or {})).decode("utf-8")
        return cls(
            plan_id=plan_id,
            step_id=step_id,
            objective=objective,
            capability_id=capability_id,
            scope=scope,
            payload_json=payload_json,
        )

    @property
    def request_digest(self) -> str:
        return _sha256_hex(
            {
                "plan_id": self.plan_id,
                "step_id": self.step_id,
                "objective": self.objective,
                "capability_id": self.capability_id,
                "scope": list(self.scope.entries),
                "payload": _strict_json_object(self.payload_json),
            }
        )

    def payload_object(self) -> dict[str, object]:
        return _strict_json_object(self.payload_json)


PayloadValidator = Callable[[ExternalActionRequest], None]


@dataclass(frozen=True)
class CapabilityDescriptor:
    capability_id: str
    required_scope_keys: frozenset[str]
    optional_scope_keys: frozenset[str] = frozenset()
    approval_required: bool = True
    max_payload_utf8_bytes: int = 16 * 1024
    max_lease_ns: int = 300_000_000_000
    max_lease_uses: int = 1

    def __post_init__(self) -> None:
        _valid_identifier(self.capability_id, label="capability_id")
        required = frozenset(self.required_scope_keys)
        optional = frozenset(self.optional_scope_keys)
        if not required:
            raise ValueError("capability descriptor requires at least one scope key")
        if required & optional:
            raise ValueError("required and optional scope keys must be disjoint")
        for key in required | optional:
            _valid_identifier(key, label="scope key")
        if self.max_payload_utf8_bytes <= 0:
            raise ValueError("max_payload_utf8_bytes must be positive")
        if self.max_lease_ns <= 0:
            raise ValueError("max_lease_ns must be positive")
        if self.max_lease_uses <= 0:
            raise ValueError("max_lease_uses must be positive")
        object.__setattr__(self, "required_scope_keys", required)
        object.__setattr__(self, "optional_scope_keys", optional)

    def validate_request(self, request: ExternalActionRequest) -> None:
        if request.capability_id != self.capability_id:
            raise AuthorizationDeniedError("request capability does not match descriptor")
        keys = request.scope.keys
        if not self.required_scope_keys.issubset(keys):
            raise AuthorizationDeniedError("request scope is missing required keys")
        allowed = self.required_scope_keys | self.optional_scope_keys
        if not keys.issubset(allowed):
            raise AuthorizationDeniedError("request scope contains unsupported keys")
        if len(request.payload_json.encode("utf-8")) > self.max_payload_utf8_bytes:
            raise AuthorizationDeniedError("request payload exceeds capability limit")


@dataclass(frozen=True)
class ActionOutcome:
    success: bool
    result: str
    confidence: float = 1.0
    evidence_record_ids: tuple[int, ...] = ()
    retryable: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise TypeError("success must be bool")
        if not self.result:
            raise ValueError("action outcome result must not be empty")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("action outcome confidence must be finite and in [0, 1]")
        evidence = tuple(int(value) for value in self.evidence_record_ids)
        if any(value <= 0 for value in evidence):
            raise ValueError("evidence_record_ids must be positive")
        if len(set(evidence)) != len(evidence):
            raise ValueError("evidence_record_ids must be unique")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "evidence_record_ids", evidence)


@dataclass(frozen=True)
class ApprovalToken:
    approval_id: str
    issuer: str
    principal: str
    request_digest: str
    issued_ns: int
    expires_ns: int
    signature: str


class ApprovalAuthority:
    def __init__(self, secret: bytes, *, issuer: str = "local-user") -> None:
        if len(secret) < 16:
            raise ValueError("approval secret must be at least 16 bytes")
        self._secret = bytes(secret)
        self.issuer = _valid_identifier(issuer, label="approval issuer")

    def _signature_payload(
        self,
        *,
        approval_id: str,
        principal: str,
        request_digest: str,
        issued_ns: int,
        expires_ns: int,
    ) -> dict[str, object]:
        return {
            "approval_id": approval_id,
            "issuer": self.issuer,
            "principal": principal,
            "request_digest": request_digest,
            "issued_ns": issued_ns,
            "expires_ns": expires_ns,
        }

    def approve(
        self,
        request: ExternalActionRequest,
        *,
        principal: str,
        ttl_ns: int,
        now_ns: int | None = None,
    ) -> ApprovalToken:
        _valid_identifier(principal, label="principal")
        if ttl_ns <= 0:
            raise ValueError("approval ttl_ns must be positive")
        issued = time.time_ns() if now_ns is None else int(now_ns)
        if issued < 0:
            raise ValueError("now_ns must be non-negative")
        expires = issued + int(ttl_ns)
        approval_id = secrets.token_hex(16)
        payload = self._signature_payload(
            approval_id=approval_id,
            principal=principal,
            request_digest=request.request_digest,
            issued_ns=issued,
            expires_ns=expires,
        )
        signature = hmac.new(
            self._secret,
            _canonical_json_bytes(payload),
            hashlib.sha256,
        ).hexdigest()
        return ApprovalToken(signature=signature, **payload)  # type: ignore[arg-type]

    def verify(
        self,
        token: ApprovalToken,
        request: ExternalActionRequest,
        *,
        principal: str,
        now_ns: int,
    ) -> None:
        if token.issuer != self.issuer:
            raise InvalidApprovalError("approval issuer mismatch")
        if token.principal != principal:
            raise InvalidApprovalError("approval principal mismatch")
        if token.request_digest != request.request_digest:
            raise InvalidApprovalError("approval request binding mismatch")
        if now_ns < token.issued_ns or now_ns >= token.expires_ns:
            raise InvalidApprovalError("approval is outside its validity window")
        payload = self._signature_payload(
            approval_id=token.approval_id,
            principal=token.principal,
            request_digest=token.request_digest,
            issued_ns=token.issued_ns,
            expires_ns=token.expires_ns,
        )
        expected = hmac.new(
            self._secret,
            _canonical_json_bytes(payload),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(token.signature, expected):
            raise InvalidApprovalError("approval signature mismatch")


@dataclass(frozen=True)
class PolicyGrant:
    principal: str
    capability_id: str
    scope: CapabilityScope
    approval_required: bool | None = None
    max_lease_ns: int = 60_000_000_000
    max_lease_uses: int = 1

    def __post_init__(self) -> None:
        _valid_identifier(self.principal, label="principal")
        _valid_identifier(self.capability_id, label="capability_id")
        if self.max_lease_ns <= 0:
            raise ValueError("max_lease_ns must be positive")
        if self.max_lease_uses <= 0:
            raise ValueError("max_lease_uses must be positive")


@dataclass(frozen=True)
class CapabilityLease:
    lease_id: str
    principal: str
    capability_id: str
    scope_digest: str
    issued_ns: int
    expires_ns: int
    max_uses: int


@dataclass(frozen=True)
class AuthorizedAction:
    request: ExternalActionRequest
    principal: str
    lease: CapabilityLease
    approval_id: str = ""


CapabilityHandler = Callable[[AuthorizedAction], ActionOutcome]


@dataclass(frozen=True)
class _RegisteredCapability:
    descriptor: CapabilityDescriptor
    handler: CapabilityHandler
    validator: PayloadValidator | None


class TypedCapabilityRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, _RegisteredCapability] = {}
        self._sealed = False

    def register(
        self,
        descriptor: CapabilityDescriptor,
        handler: CapabilityHandler,
        *,
        validator: PayloadValidator | None = None,
    ) -> None:
        if self._sealed:
            raise AuthorityError("capability registry is sealed")
        if descriptor.capability_id in self._entries:
            raise AuthorityError("capability is already registered")
        self._entries[descriptor.capability_id] = _RegisteredCapability(
            descriptor,
            handler,
            validator,
        )

    def seal(self) -> None:
        self._sealed = True

    @property
    def sealed(self) -> bool:
        return self._sealed

    def descriptor(self, capability_id: str) -> CapabilityDescriptor:
        entry = self._entries.get(capability_id)
        if entry is None:
            raise CapabilityNotFoundError(f"unknown capability: {capability_id}")
        return entry.descriptor

    def validate(self, request: ExternalActionRequest) -> CapabilityDescriptor:
        entry = self._entries.get(request.capability_id)
        if entry is None:
            raise CapabilityNotFoundError(f"unknown capability: {request.capability_id}")
        entry.descriptor.validate_request(request)
        if entry.validator is not None:
            entry.validator(request)
        return entry.descriptor

    def _execute_authorized(self, action: AuthorizedAction) -> ActionOutcome:
        entry = self._entries.get(action.request.capability_id)
        if entry is None:
            raise CapabilityNotFoundError(
                f"unknown capability: {action.request.capability_id}"
            )
        outcome = entry.handler(action)
        if not isinstance(outcome, ActionOutcome):
            raise ExternalExecutionError("capability handler must return ActionOutcome")
        return outcome


class DenyByDefaultAuthorityGate:
    def __init__(
        self,
        grants: Iterable[PolicyGrant],
        *,
        approval_authority: ApprovalAuthority | None = None,
    ) -> None:
        self._grants: dict[tuple[str, str, str], PolicyGrant] = {}
        for grant in grants:
            key = (grant.principal, grant.capability_id, grant.scope.digest)
            if key in self._grants:
                raise ValueError("duplicate authority policy grant")
            self._grants[key] = grant
        self._approval_authority = approval_authority
        self._leases: dict[str, CapabilityLease] = {}
        self._uses: dict[str, int] = {}

    def issue_lease(
        self,
        descriptor: CapabilityDescriptor,
        request: ExternalActionRequest,
        *,
        principal: str,
        approval: ApprovalToken | None = None,
        now_ns: int | None = None,
    ) -> CapabilityLease:
        _valid_identifier(principal, label="principal")
        current = time.time_ns() if now_ns is None else int(now_ns)
        if current < 0:
            raise ValueError("now_ns must be non-negative")
        descriptor.validate_request(request)
        key = (principal, request.capability_id, request.scope.digest)
        grant = self._grants.get(key)
        if grant is None:
            raise AuthorizationDeniedError("no matching authority policy grant")
        requires_approval = (
            descriptor.approval_required
            if grant.approval_required is None
            else grant.approval_required
        )
        if requires_approval:
            if approval is None:
                raise ApprovalRequiredError("capability requires explicit approval")
            if self._approval_authority is None:
                raise InvalidApprovalError("approval authority is not configured")
            self._approval_authority.verify(
                approval,
                request,
                principal=principal,
                now_ns=current,
            )
        ttl_ns = min(descriptor.max_lease_ns, grant.max_lease_ns)
        max_uses = min(descriptor.max_lease_uses, grant.max_lease_uses)
        lease = CapabilityLease(
            lease_id=secrets.token_hex(16),
            principal=principal,
            capability_id=request.capability_id,
            scope_digest=request.scope.digest,
            issued_ns=current,
            expires_ns=current + ttl_ns,
            max_uses=max_uses,
        )
        self._leases[lease.lease_id] = lease
        self._uses[lease.lease_id] = 0
        return lease

    def consume(
        self,
        lease: CapabilityLease,
        request: ExternalActionRequest,
        *,
        principal: str,
        now_ns: int | None = None,
    ) -> None:
        current = time.time_ns() if now_ns is None else int(now_ns)
        issued = self._leases.get(lease.lease_id)
        if issued is None or issued != lease:
            raise LeaseError("lease was not issued by this authority gate")
        if lease.principal != principal:
            raise LeaseError("lease principal mismatch")
        if lease.capability_id != request.capability_id:
            raise LeaseError("lease capability mismatch")
        if lease.scope_digest != request.scope.digest:
            raise LeaseError("lease scope mismatch")
        if current < lease.issued_ns or current >= lease.expires_ns:
            raise LeaseError("lease is outside its validity window")
        used = self._uses[lease.lease_id]
        if used >= lease.max_uses:
            raise LeaseError("lease use budget exhausted")
        self._uses[lease.lease_id] = used + 1


@dataclass(frozen=True)
class ActionReceipt:
    sequence: int
    timestamp_ns: int
    previous_receipt_id: str
    receipt_id: str
    status: ReceiptStatus
    request_digest: str
    plan_id: str
    step_id: int
    capability_id: str
    scope_digest: str
    principal: str
    lease_id: str
    approval_id: str
    result: str
    confidence: float
    evidence_record_ids: tuple[int, ...]
    retryable: bool
    error_type: str

    def payload_without_id(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "timestamp_ns": self.timestamp_ns,
            "previous_receipt_id": self.previous_receipt_id,
            "status": int(self.status),
            "request_digest": self.request_digest,
            "plan_id": self.plan_id,
            "step_id": self.step_id,
            "capability_id": self.capability_id,
            "scope_digest": self.scope_digest,
            "principal": self.principal,
            "lease_id": self.lease_id,
            "approval_id": self.approval_id,
            "result": self.result,
            "confidence": self.confidence,
            "evidence_record_ids": list(self.evidence_record_ids),
            "retryable": self.retryable,
            "error_type": self.error_type,
        }


class ImmutableActionAudit:
    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self._path = None if path is None else Path(path)
        self._receipts: list[ActionReceipt] = []
        if self._path is not None and self._path.exists():
            self._load()

    @property
    def receipts(self) -> tuple[ActionReceipt, ...]:
        return tuple(self._receipts)

    def _parse_receipt(self, value: object) -> ActionReceipt:
        if not isinstance(value, dict):
            raise AuditIntegrityError("audit record must be a JSON object")
        expected = {
            "sequence", "timestamp_ns", "previous_receipt_id", "receipt_id",
            "status", "request_digest", "plan_id", "step_id", "capability_id",
            "scope_digest", "principal", "lease_id", "approval_id", "result",
            "confidence", "evidence_record_ids", "retryable", "error_type",
        }
        if set(value) != expected:
            raise AuditIntegrityError("audit record has unexpected fields")
        try:
            receipt = ActionReceipt(
                sequence=int(value["sequence"]),
                timestamp_ns=int(value["timestamp_ns"]),
                previous_receipt_id=str(value["previous_receipt_id"]),
                receipt_id=str(value["receipt_id"]),
                status=ReceiptStatus(int(value["status"])),
                request_digest=str(value["request_digest"]),
                plan_id=str(value["plan_id"]),
                step_id=int(value["step_id"]),
                capability_id=str(value["capability_id"]),
                scope_digest=str(value["scope_digest"]),
                principal=str(value["principal"]),
                lease_id=str(value["lease_id"]),
                approval_id=str(value["approval_id"]),
                result=str(value["result"]),
                confidence=float(value["confidence"]),
                evidence_record_ids=tuple(int(v) for v in value["evidence_record_ids"]),
                retryable=bool(value["retryable"]),
                error_type=str(value["error_type"]),
            )
        except (TypeError, ValueError) as exc:
            raise AuditIntegrityError("audit record has invalid field types") from exc
        expected_id = _sha256_hex(receipt.payload_without_id())
        if not hmac.compare_digest(receipt.receipt_id, expected_id):
            raise AuditIntegrityError("audit receipt digest mismatch")
        return receipt

    def _load(self) -> None:
        assert self._path is not None
        previous = ""
        expected_sequence = 1
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.endswith("\n"):
                    raise AuditIntegrityError("audit contains a torn final record")
                try:
                    value = _strict_json_object(line[:-1])
                except (json.JSONDecodeError, ValueError) as exc:
                    raise AuditIntegrityError("audit JSON is invalid") from exc
                receipt = self._parse_receipt(value)
                if receipt.sequence != expected_sequence:
                    raise AuditIntegrityError("audit sequence is not contiguous")
                if receipt.previous_receipt_id != previous:
                    raise AuditIntegrityError("audit hash chain is broken")
                self._receipts.append(receipt)
                previous = receipt.receipt_id
                expected_sequence += 1

    def append(
        self,
        *,
        status: ReceiptStatus,
        request: ExternalActionRequest,
        principal: str,
        lease_id: str = "",
        approval_id: str = "",
        outcome: ActionOutcome | None = None,
        error_type: str = "",
        timestamp_ns: int | None = None,
    ) -> ActionReceipt:
        current = time.time_ns() if timestamp_ns is None else int(timestamp_ns)
        previous = self._receipts[-1].receipt_id if self._receipts else ""
        result = "" if outcome is None else outcome.result
        confidence = 0.0 if outcome is None else outcome.confidence
        evidence = () if outcome is None else outcome.evidence_record_ids
        retryable = False if outcome is None else outcome.retryable
        base = {
            "sequence": len(self._receipts) + 1,
            "timestamp_ns": current,
            "previous_receipt_id": previous,
            "status": int(status),
            "request_digest": request.request_digest,
            "plan_id": request.plan_id,
            "step_id": request.step_id,
            "capability_id": request.capability_id,
            "scope_digest": request.scope.digest,
            "principal": principal,
            "lease_id": lease_id,
            "approval_id": approval_id,
            "result": result,
            "confidence": confidence,
            "evidence_record_ids": list(evidence),
            "retryable": retryable,
            "error_type": error_type,
        }
        receipt_id = _sha256_hex(base)
        receipt = ActionReceipt(
            receipt_id=receipt_id,
            evidence_record_ids=tuple(evidence),
            status=ReceiptStatus(status),
            **{k: v for k, v in base.items() if k not in {"status", "evidence_record_ids"}},
        )
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            record = dict(base)
            record["receipt_id"] = receipt_id
            encoded = _canonical_json_bytes(record) + b"\n"
            fd = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                written = os.write(fd, encoded)
                if written != len(encoded):
                    raise OSError("short audit write")
                os.fsync(fd)
            finally:
                os.close(fd)
        self._receipts.append(receipt)
        return receipt

    def successful_receipt(self, request_digest: str) -> ActionReceipt | None:
        for receipt in reversed(self._receipts):
            if (
                receipt.request_digest == request_digest
                and receipt.status == ReceiptStatus.SUCCEEDED
            ):
                return receipt
        return None


@dataclass(frozen=True)
class ExecutionResult:
    receipt: ActionReceipt
    replayed: bool


class _PlanControllerLike(Protocol):
    plan: object

    def record_external_result(
        self,
        step_id: int,
        *,
        result: str,
        confidence: float,
        evidence_record_ids: Iterable[int] = (),
    ) -> None:
        ...

    def fail_step(
        self,
        step_id: int,
        *,
        reason: str,
        retryable: bool = True,
    ) -> None:
        ...


class ExternalExecutionFabric:
    """The only M6A path from WAITING_EXTERNAL to a registered side effect."""

    def __init__(
        self,
        registry: TypedCapabilityRegistry,
        authority: DenyByDefaultAuthorityGate,
        audit: ImmutableActionAudit,
    ) -> None:
        if not registry.sealed:
            raise ValueError("capability registry must be sealed before execution")
        self._registry = registry
        self._authority = authority
        self._audit = audit

    def _validate_waiting_step(
        self,
        controller: _PlanControllerLike,
        request: ExternalActionRequest,
    ) -> None:
        plan = controller.plan
        if getattr(plan, "plan_id", None) != request.plan_id:
            raise ExternalExecutionError("request plan_id does not match controller")
        plan_status = getattr(getattr(plan, "status", None), "name", "")
        if plan_status != "WAITING_EXTERNAL":
            raise ExternalExecutionError("plan is not WAITING_EXTERNAL")
        try:
            step = plan.step(request.step_id)
        except Exception as exc:
            raise ExternalExecutionError("request step_id is invalid") from exc
        if getattr(getattr(step, "status", None), "name", "") != "WAITING_EXTERNAL":
            raise ExternalExecutionError("step is not WAITING_EXTERNAL")
        spec = getattr(step, "spec", None)
        if getattr(getattr(spec, "kind", None), "name", "") != "EXTERNAL":
            raise ExternalExecutionError("waiting step is not EXTERNAL")
        if getattr(spec, "objective", None) != request.objective:
            raise ExternalExecutionError("request objective does not match immutable plan step")

    def execute_waiting(
        self,
        controller: _PlanControllerLike,
        request: ExternalActionRequest,
        *,
        principal: str,
        approval: ApprovalToken | None = None,
        lease: CapabilityLease | None = None,
        now_ns: int | None = None,
    ) -> ExecutionResult:
        self._validate_waiting_step(controller, request)
        descriptor = self._registry.validate(request)

        prior = self._audit.successful_receipt(request.request_digest)
        if prior is not None:
            controller.record_external_result(
                request.step_id,
                result=prior.result,
                confidence=prior.confidence,
                evidence_record_ids=prior.evidence_record_ids,
            )
            return ExecutionResult(prior, replayed=True)

        current = time.time_ns() if now_ns is None else int(now_ns)
        active_lease = lease
        try:
            if active_lease is None:
                active_lease = self._authority.issue_lease(
                    descriptor,
                    request,
                    principal=principal,
                    approval=approval,
                    now_ns=current,
                )
            self._authority.consume(
                active_lease,
                request,
                principal=principal,
                now_ns=current,
            )
        except AuthorityError as exc:
            self._audit.append(
                status=ReceiptStatus.DENIED,
                request=request,
                principal=principal,
                lease_id="" if active_lease is None else active_lease.lease_id,
                approval_id="" if approval is None else approval.approval_id,
                error_type=type(exc).__name__,
                timestamp_ns=current,
            )
            raise

        action = AuthorizedAction(
            request=request,
            principal=principal,
            lease=active_lease,
            approval_id="" if approval is None else approval.approval_id,
        )
        try:
            outcome = self._registry._execute_authorized(action)
        except Exception as exc:
            receipt = self._audit.append(
                status=ReceiptStatus.FAILED,
                request=request,
                principal=principal,
                lease_id=active_lease.lease_id,
                approval_id=action.approval_id,
                error_type=type(exc).__name__,
                timestamp_ns=current,
            )
            controller.fail_step(
                request.step_id,
                reason=f"external capability failure: {type(exc).__name__}",
                retryable=False,
            )
            raise ExternalExecutionError(
                "external capability raised before producing a typed outcome"
            ) from exc

        if not outcome.success:
            receipt = self._audit.append(
                status=ReceiptStatus.FAILED,
                request=request,
                principal=principal,
                lease_id=active_lease.lease_id,
                approval_id=action.approval_id,
                outcome=outcome,
                timestamp_ns=current,
            )
            controller.fail_step(
                request.step_id,
                reason=outcome.result,
                retryable=outcome.retryable,
            )
            return ExecutionResult(receipt, replayed=False)

        receipt = self._audit.append(
            status=ReceiptStatus.SUCCEEDED,
            request=request,
            principal=principal,
            lease_id=active_lease.lease_id,
            approval_id=action.approval_id,
            outcome=outcome,
            timestamp_ns=current,
        )
        controller.record_external_result(
            request.step_id,
            result=outcome.result,
            confidence=outcome.confidence,
            evidence_record_ids=outcome.evidence_record_ids,
        )
        return ExecutionResult(receipt, replayed=False)
