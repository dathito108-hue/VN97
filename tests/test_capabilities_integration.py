from pathlib import Path

from vn97.authority import (
    ApprovalAuthority,
    CapabilityScope,
    DenyByDefaultAuthorityGate,
    ExternalActionRequest,
    ExternalExecutionFabric,
    ImmutableActionAudit,
    PolicyGrant,
    TypedCapabilityRegistry,
)
from vn97.capabilities import (
    CAP_FILE_WRITE,
    CapabilityPackConfig,
    ConfinedFileStore,
    FileRoot,
    register_m6b_capabilities,
)
from vn97.planner import PlanController, PlanStatus, PlanStepSpec, StepKind


def _waiting_controller() -> PlanController:
    controller = PlanController.create(
        "persist a local note",
        (PlanStepSpec(StepKind.EXTERNAL, "write note safely"),),
        created_ns=1,
    )
    controller.begin_step(1)
    assert controller.plan.status == PlanStatus.WAITING_EXTERNAL
    return controller


def test_m6b_file_write_runs_through_m6a_and_replays_without_duplicate_write(
    tmp_path: Path,
):
    root = tmp_path / "root"
    root.mkdir()
    audit_path = tmp_path / "actions.jsonl"

    registry = TypedCapabilityRegistry()
    registered = register_m6b_capabilities(
        registry,
        CapabilityPackConfig(
            file_store=ConfinedFileStore([FileRoot("app", root)])
        ),
    )
    assert registered == ("file.read", "file.write")
    registry.seal()

    first = _waiting_controller()
    scope = CapabilityScope.from_mapping({"root": "app", "path": "note.txt"})
    request = ExternalActionRequest.create(
        plan_id=first.plan.plan_id,
        step_id=1,
        objective="write note safely",
        capability_id=CAP_FILE_WRITE,
        scope=scope,
        payload={"text": "hello VN97"},
    )

    principal = "runtime.user"
    approver = ApprovalAuthority(b"m6b-integration-secret-material")
    gate = DenyByDefaultAuthorityGate(
        [PolicyGrant(principal, CAP_FILE_WRITE, scope)],
        approval_authority=approver,
    )
    audit = ImmutableActionAudit(audit_path)
    fabric = ExternalExecutionFabric(registry, gate, audit)
    approval = approver.approve(
        request,
        principal=principal,
        ttl_ns=1_000_000,
        now_ns=10,
    )
    executed = fabric.execute_waiting(
        first,
        request,
        principal=principal,
        approval=approval,
        now_ns=20,
    )

    assert not executed.replayed
    assert first.plan.status == PlanStatus.COMPLETED
    assert (root / "note.txt").read_text(encoding="utf-8") == "hello VN97"

    restored = _waiting_controller()
    replay_request = ExternalActionRequest.create(
        plan_id=restored.plan.plan_id,
        step_id=1,
        objective="write note safely",
        capability_id=CAP_FILE_WRITE,
        scope=scope,
        payload={"text": "hello VN97"},
    )
    reloaded_audit = ImmutableActionAudit(audit_path)
    replay_fabric = ExternalExecutionFabric(registry, gate, reloaded_audit)
    replayed = replay_fabric.execute_waiting(
        restored,
        replay_request,
        principal=principal,
        now_ns=30,
    )

    assert replayed.replayed
    assert restored.plan.status == PlanStatus.COMPLETED
    assert len(reloaded_audit.receipts) == 1
    assert (root / "note.txt").read_text(encoding="utf-8") == "hello VN97"
