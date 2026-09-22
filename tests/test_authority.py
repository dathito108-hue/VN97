from pathlib import Path

import pytest

from vn97.authority import (
    ActionOutcome,
    ApprovalAuthority,
    ApprovalRequiredError,
    AuditIntegrityError,
    AuthorizationDeniedError,
    CapabilityDescriptor,
    CapabilityScope,
    DenyByDefaultAuthorityGate,
    ExternalActionRequest,
    ExternalExecutionError,
    ExternalExecutionFabric,
    ImmutableActionAudit,
    InvalidApprovalError,
    LeaseError,
    PolicyGrant,
    ReceiptStatus,
    TypedCapabilityRegistry,
)
from vn97.planner import (
    PlanController,
    PlanStatus,
    PlanStepSpec,
    StepKind,
    StepStatus,
)


_PRINCIPAL = "runtime.user"
_CAPABILITY = "file.write"
_OBJECTIVE = "write local note"


def _scope(path: str = "/data/user/0/vn97/files/note.txt") -> CapabilityScope:
    return CapabilityScope.from_mapping({"path": path})


def _waiting_controller() -> PlanController:
    controller = PlanController.create(
        "persist a local note",
        (PlanStepSpec(StepKind.EXTERNAL, _OBJECTIVE),),
        created_ns=1,
    )
    controller.begin_step(1)
    assert controller.plan.status == PlanStatus.WAITING_EXTERNAL
    assert controller.plan.step(1).status == StepStatus.WAITING_EXTERNAL
    return controller


def _request(
    controller: PlanController,
    *,
    scope: CapabilityScope | None = None,
    objective: str = _OBJECTIVE,
) -> ExternalActionRequest:
    return ExternalActionRequest.create(
        plan_id=controller.plan.plan_id,
        step_id=1,
        objective=objective,
        capability_id=_CAPABILITY,
        scope=scope or _scope(),
        payload={"text": "hello from VN97"},
    )


def _stack(
    tmp_path: Path,
    *,
    approval_required: bool = True,
    handler=None,
):
    calls = []

    if handler is None:
        def handler(action):
            calls.append(action)
            return ActionOutcome(True, "local note written", 0.99)

    registry = TypedCapabilityRegistry()
    descriptor = CapabilityDescriptor(
        _CAPABILITY,
        frozenset({"path"}),
        approval_required=approval_required,
        max_lease_ns=100,
        max_lease_uses=1,
    )
    registry.register(descriptor, handler)
    registry.seal()

    approver = ApprovalAuthority(b"m6a-test-secret-material-000000")
    grant = PolicyGrant(
        _PRINCIPAL,
        _CAPABILITY,
        _scope(),
        approval_required=approval_required,
        max_lease_ns=100,
        max_lease_uses=1,
    )
    gate = DenyByDefaultAuthorityGate(
        [grant],
        approval_authority=approver,
    )
    audit = ImmutableActionAudit(tmp_path / "m6a-audit.jsonl")
    fabric = ExternalExecutionFabric(registry, gate, audit)
    return calls, registry, approver, gate, audit, fabric


def test_m6a_denies_without_approval_and_keeps_planner_waiting(tmp_path: Path):
    calls, _, _, _, audit, fabric = _stack(tmp_path)
    controller = _waiting_controller()
    request = _request(controller)

    with pytest.raises(ApprovalRequiredError):
        fabric.execute_waiting(
            controller,
            request,
            principal=_PRINCIPAL,
            now_ns=10,
        )

    assert calls == []
    assert controller.plan.status == PlanStatus.WAITING_EXTERNAL
    assert controller.plan.step(1).status == StepStatus.WAITING_EXTERNAL
    assert audit.receipts[-1].status == ReceiptStatus.DENIED


def test_authorized_waiting_external_executes_and_resumes_real_planner(tmp_path: Path):
    calls, _, approver, _, audit, fabric = _stack(tmp_path)
    controller = _waiting_controller()
    request = _request(controller)
    approval = approver.approve(
        request,
        principal=_PRINCIPAL,
        ttl_ns=100,
        now_ns=10,
    )

    result = fabric.execute_waiting(
        controller,
        request,
        principal=_PRINCIPAL,
        approval=approval,
        now_ns=20,
    )

    assert not result.replayed
    assert len(calls) == 1
    assert controller.plan.status == PlanStatus.COMPLETED
    step = controller.plan.step(1)
    assert step.status == StepStatus.SUCCEEDED
    assert step.result == "local note written"
    assert step.confidence == 0.99
    assert result.receipt.status == ReceiptStatus.SUCCEEDED
    assert result.receipt.approval_id == approval.approval_id
    assert ImmutableActionAudit(tmp_path / "m6a-audit.jsonl").receipts[-1] == result.receipt


def test_success_receipt_replays_result_without_second_side_effect(tmp_path: Path):
    calls, _, approver, _, _, fabric = _stack(tmp_path)
    first = _waiting_controller()
    request = _request(first)
    approval = approver.approve(
        request,
        principal=_PRINCIPAL,
        ttl_ns=100,
        now_ns=10,
    )
    first_result = fabric.execute_waiting(
        first,
        request,
        principal=_PRINCIPAL,
        approval=approval,
        now_ns=20,
    )
    assert not first_result.replayed
    assert len(calls) == 1

    restored = _waiting_controller()
    replay = fabric.execute_waiting(
        restored,
        request,
        principal=_PRINCIPAL,
        now_ns=30,
    )
    assert replay.replayed
    assert replay.receipt.receipt_id == first_result.receipt.receipt_id
    assert len(calls) == 1
    assert restored.plan.status == PlanStatus.COMPLETED


def test_policy_scope_is_exact_and_denies_ungranted_target(tmp_path: Path):
    calls, _, approver, _, audit, fabric = _stack(tmp_path)
    controller = _waiting_controller()
    request = _request(controller, scope=_scope("/data/user/0/vn97/files/other.txt"))
    approval = approver.approve(
        request,
        principal=_PRINCIPAL,
        ttl_ns=100,
        now_ns=10,
    )

    with pytest.raises(AuthorizationDeniedError):
        fabric.execute_waiting(
            controller,
            request,
            principal=_PRINCIPAL,
            approval=approval,
            now_ns=20,
        )

    assert calls == []
    assert audit.receipts[-1].status == ReceiptStatus.DENIED
    assert controller.plan.status == PlanStatus.WAITING_EXTERNAL


def test_approval_is_bound_to_exact_request_digest(tmp_path: Path):
    calls, _, approver, _, audit, fabric = _stack(tmp_path)
    controller = _waiting_controller()
    original = _request(controller)
    approval = approver.approve(
        original,
        principal=_PRINCIPAL,
        ttl_ns=100,
        now_ns=10,
    )
    substituted = ExternalActionRequest.create(
        plan_id=original.plan_id,
        step_id=original.step_id,
        objective=original.objective,
        capability_id=original.capability_id,
        scope=original.scope,
        payload={"text": "different content"},
    )

    with pytest.raises(InvalidApprovalError):
        fabric.execute_waiting(
            controller,
            substituted,
            principal=_PRINCIPAL,
            approval=approval,
            now_ns=20,
        )

    assert calls == []
    assert audit.receipts[-1].status == ReceiptStatus.DENIED


def test_lease_is_issued_by_gate_and_bounded_by_time_and_use(tmp_path: Path):
    _, registry, _, gate, _, _ = _stack(tmp_path, approval_required=False)
    controller = _waiting_controller()
    request = _request(controller)
    descriptor = registry.descriptor(_CAPABILITY)

    lease = gate.issue_lease(
        descriptor,
        request,
        principal=_PRINCIPAL,
        now_ns=10,
    )
    gate.consume(lease, request, principal=_PRINCIPAL, now_ns=20)
    with pytest.raises(LeaseError, match="budget"):
        gate.consume(lease, request, principal=_PRINCIPAL, now_ns=21)

    expired = gate.issue_lease(
        descriptor,
        request,
        principal=_PRINCIPAL,
        now_ns=10,
    )
    with pytest.raises(LeaseError, match="validity"):
        gate.consume(expired, request, principal=_PRINCIPAL, now_ns=110)


def test_request_must_match_immutable_waiting_plan_step_before_execution(tmp_path: Path):
    calls, _, approver, _, _, fabric = _stack(tmp_path)
    controller = _waiting_controller()
    request = _request(controller, objective="replace a different file")
    approval = approver.approve(
        request,
        principal=_PRINCIPAL,
        ttl_ns=100,
        now_ns=10,
    )

    with pytest.raises(ExternalExecutionError, match="objective"):
        fabric.execute_waiting(
            controller,
            request,
            principal=_PRINCIPAL,
            approval=approval,
            now_ns=20,
        )
    assert calls == []
    assert controller.plan.status == PlanStatus.WAITING_EXTERNAL


def test_handler_exception_fails_plan_closed_and_is_receipted(tmp_path: Path):
    def broken_handler(_action):
        raise RuntimeError("device failure")

    _, _, _, _, audit, fabric = _stack(
        tmp_path,
        approval_required=False,
        handler=broken_handler,
    )
    controller = _waiting_controller()

    with pytest.raises(ExternalExecutionError, match="typed outcome"):
        fabric.execute_waiting(
            controller,
            _request(controller),
            principal=_PRINCIPAL,
            now_ns=20,
        )

    assert controller.plan.status == PlanStatus.FAILED
    assert controller.plan.step(1).status == StepStatus.FAILED
    assert audit.receipts[-1].status == ReceiptStatus.FAILED
    assert audit.receipts[-1].error_type == "RuntimeError"


def test_explicit_failed_outcome_uses_handler_retry_policy(tmp_path: Path):
    def denied_handler(_action):
        return ActionOutcome(
            False,
            "platform permission revoked",
            confidence=0.0,
            retryable=False,
        )

    _, _, _, _, audit, fabric = _stack(
        tmp_path,
        approval_required=False,
        handler=denied_handler,
    )
    controller = _waiting_controller()
    result = fabric.execute_waiting(
        controller,
        _request(controller),
        principal=_PRINCIPAL,
        now_ns=20,
    )

    assert result.receipt.status == ReceiptStatus.FAILED
    assert controller.plan.status == PlanStatus.FAILED
    assert audit.receipts[-1].result == "platform permission revoked"


def test_persisted_hash_chain_detects_receipt_tampering(tmp_path: Path):
    _, _, _, _, audit, fabric = _stack(tmp_path, approval_required=False)
    controller = _waiting_controller()
    fabric.execute_waiting(
        controller,
        _request(controller),
        principal=_PRINCIPAL,
        now_ns=20,
    )

    audit_path = tmp_path / "m6a-audit.jsonl"
    original = audit_path.read_text(encoding="utf-8")
    audit_path.write_text(
        original.replace("local note written", "forged result"),
        encoding="utf-8",
    )
    with pytest.raises(AuditIntegrityError, match="digest"):
        ImmutableActionAudit(audit_path)
