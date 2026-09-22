from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json

import pytest

from vn97.authority import (
    ActionOutcome,
    ApprovalAuthority,
    CapabilityDescriptor,
    DenyByDefaultAuthorityGate,
    ExternalExecutionFabric,
    ImmutableActionAudit,
    PolicyGrant,
    TypedCapabilityRegistry,
)
from vn97.cognition_adapter import VN97CognitionAdapter, VN97CognitionOutputError
from vn97.external_intent import (
    ApprovalSessionError,
    ExternalApprovalCoordinator,
    ExternalApprovalLimits,
    ExternalCapabilityView,
    ExternalIntent,
    ExternalIntentBinder,
    ExternalIntentContractError,
    ExternalIntentError,
    ExternalIntentRequest,
)
from vn97.planner import PlanController, PlanStatus, PlanStepSpec, StepKind


CAPABILITY = "test.echo"


def _registry(calls=None):
    registry = TypedCapabilityRegistry()

    def handler(action):
        if calls is not None:
            calls.append(action.request.request_digest)
        return ActionOutcome(True, "external-ok")

    registry.register(
        CapabilityDescriptor(
            CAPABILITY,
            frozenset({"target"}),
            optional_scope_keys=frozenset({"mode"}),
            approval_required=True,
            max_payload_utf8_bytes=4096,
            max_lease_uses=1,
        ),
        handler,
    )
    registry.seal()
    return registry


def _waiting_controller():
    controller = PlanController.create(
        "perform an external action",
        (PlanStepSpec(StepKind.EXTERNAL, "send exact external action"),),
        created_ns=1,
    )
    controller.begin_step(1)
    assert controller.plan.status == PlanStatus.WAITING_EXTERNAL
    return controller


def _intent(payload=None):
    return ExternalIntent.create(
        capability_id=CAPABILITY,
        scope={"target": "alpha"},
        payload=payload or {"message": "hello"},
    )


def test_binder_requires_sealed_registry():
    registry = TypedCapabilityRegistry()
    registry.register(
        CapabilityDescriptor(CAPABILITY, frozenset({"target"})),
        lambda action: ActionOutcome(True, "ok"),
    )
    with pytest.raises(ValueError, match="sealed"):
        ExternalIntentBinder(registry, (CAPABILITY,))


def test_binder_rebuilds_plan_step_binding_from_trusted_controller():
    controller = _waiting_controller()
    binder = ExternalIntentBinder(_registry(), (CAPABILITY,))
    request = binder.bind(controller, _intent())
    assert request.plan_id == controller.plan.plan_id
    assert request.step_id == 1
    assert request.objective == "send exact external action"
    assert request.capability_id == CAPABILITY
    assert request.scope.as_dict() == {"target": "alpha"}
    assert request.payload_object() == {"message": "hello"}


def test_binder_rejects_capability_outside_catalog_and_invalid_scope():
    controller = _waiting_controller()
    binder = ExternalIntentBinder(_registry(), (CAPABILITY,))
    with pytest.raises(ExternalIntentContractError, match="outside trusted catalog"):
        binder.bind(
            controller,
            ExternalIntent.create(
                capability_id="test.other",
                scope={"target": "alpha"},
            ),
        )
    with pytest.raises(ExternalIntentContractError, match="registered capability"):
        binder.bind(
            controller,
            ExternalIntent.create(
                capability_id=CAPABILITY,
                scope={"wrong": "alpha"},
            ),
        )


def test_backend_receives_only_capability_contract_not_policy_or_approval():
    controller = _waiting_controller()
    binder = ExternalIntentBinder(_registry(), (CAPABILITY,))

    class Backend:
        seen = None

        def propose_external_intent(self, request):
            self.seen = request
            return _intent()

    backend = Backend()
    request = binder.propose_and_bind(controller, backend)
    assert request.capability_id == CAPABILITY
    assert backend.seen.capabilities[0].required_scope_keys == ("target",)
    assert backend.seen.capabilities[0].optional_scope_keys == ("mode",)
    assert not hasattr(backend.seen, "policy")
    assert not hasattr(backend.seen, "approval")
    assert not hasattr(backend.seen, "lease")


def test_intent_cannot_rebind_if_waiting_step_changes_during_proposal():
    controller = PlanController.create(
        "two external actions",
        (
            PlanStepSpec(StepKind.EXTERNAL, "first"),
            PlanStepSpec(StepKind.EXTERNAL, "second", dependencies=(1,)),
        ),
        created_ns=1,
    )
    controller.begin_step(1)
    binder = ExternalIntentBinder(_registry(), (CAPABILITY,))

    class RacingBackend:
        def propose_external_intent(self, request):
            controller.record_external_result(
                1,
                result="first-complete",
                confidence=1.0,
            )
            controller.begin_step(2)
            return _intent()

    with pytest.raises(ExternalIntentError, match="changed during intent proposal"):
        binder.propose_and_bind(controller, RacingBackend())


def test_binder_refuses_non_waiting_plan():
    controller = PlanController.create(
        "goal",
        (PlanStepSpec(StepKind.EXTERNAL, "external"),),
        created_ns=1,
    )
    binder = ExternalIntentBinder(_registry(), (CAPABILITY,))
    with pytest.raises(ExternalIntentError, match="WAITING_EXTERNAL"):
        binder.bind(controller, _intent())


def test_approval_prompt_is_exact_digest_one_shot_and_presentable():
    controller = _waiting_controller()
    request = ExternalIntentBinder(_registry(), (CAPABILITY,)).bind(
        controller,
        _intent(),
    )
    authority = ApprovalAuthority(b"m6c-approval-secret-material")
    coordinator = ExternalApprovalCoordinator(authority)
    prompt = coordinator.create_prompt(
        request,
        principal="runtime.user",
        prompt_ttl_ns=1_000,
        now_ns=10,
    )
    shown = json.loads(prompt.presentation_json)
    assert shown["request_digest"] == request.request_digest
    assert shown["objective"] == request.objective
    assert shown["scope"] == {"target": "alpha"}
    assert shown["payload"] == {"message": "hello"}

    token = coordinator.resolve(
        prompt,
        approved=True,
        approval_ttl_ns=500,
        now_ns=20,
    )
    assert token is not None
    authority.verify(token, request, principal="runtime.user", now_ns=21)
    with pytest.raises(ApprovalSessionError, match="already consumed"):
        coordinator.resolve(
            prompt,
            approved=True,
            approval_ttl_ns=500,
            now_ns=22,
        )


def test_denial_and_expiry_never_mint_approval():
    controller = _waiting_controller()
    request = ExternalIntentBinder(_registry(), (CAPABILITY,)).bind(
        controller,
        _intent(),
    )
    coordinator = ExternalApprovalCoordinator(
        ApprovalAuthority(b"m6c-approval-secret-material")
    )
    denied = coordinator.create_prompt(
        request,
        principal="runtime.user",
        prompt_ttl_ns=100,
        now_ns=10,
    )
    assert (
        coordinator.resolve(
            denied,
            approved=False,
            approval_ttl_ns=1,
            now_ns=20,
        )
        is None
    )

    expired = coordinator.create_prompt(
        request,
        principal="runtime.user",
        prompt_ttl_ns=10,
        now_ns=30,
    )
    with pytest.raises(ApprovalSessionError, match="validity window"):
        coordinator.resolve(
            expired,
            approved=True,
            approval_ttl_ns=10,
            now_ns=40,
        )


def test_tampered_prompt_is_rejected_and_consumed():
    controller = _waiting_controller()
    request = ExternalIntentBinder(_registry(), (CAPABILITY,)).bind(
        controller,
        _intent(),
    )
    coordinator = ExternalApprovalCoordinator(
        ApprovalAuthority(b"m6c-approval-secret-material")
    )
    prompt = coordinator.create_prompt(
        request,
        principal="runtime.user",
        prompt_ttl_ns=100,
        now_ns=10,
    )
    tampered = replace(prompt, objective="different")
    with pytest.raises(ApprovalSessionError):
        coordinator.resolve(
            tampered,
            approved=True,
            approval_ttl_ns=10,
            now_ns=20,
        )
    with pytest.raises(ApprovalSessionError):
        coordinator.resolve(
            prompt,
            approved=True,
            approval_ttl_ns=10,
            now_ns=21,
        )


def test_concurrent_approval_resolution_mints_at_most_one_token():
    controller = _waiting_controller()
    request = ExternalIntentBinder(_registry(), (CAPABILITY,)).bind(
        controller,
        _intent(),
    )
    coordinator = ExternalApprovalCoordinator(
        ApprovalAuthority(b"m6c-approval-secret-material")
    )
    prompt = coordinator.create_prompt(
        request,
        principal="runtime.user",
        prompt_ttl_ns=1_000,
        now_ns=10,
    )

    def resolve_once():
        try:
            return coordinator.resolve(
                prompt,
                approved=True,
                approval_ttl_ns=100,
                now_ns=20,
            )
        except ApprovalSessionError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: resolve_once(), range(2)))
    assert sum(result is not None for result in results) == 1


def test_pending_prompt_budget_is_bounded_and_expired_prompts_are_pruned():
    controller = _waiting_controller()
    request = ExternalIntentBinder(_registry(), (CAPABILITY,)).bind(
        controller,
        _intent(),
    )
    coordinator = ExternalApprovalCoordinator(
        ApprovalAuthority(b"m6c-approval-secret-material"),
        limits=ExternalApprovalLimits(
            max_pending_prompts=1,
            max_prompt_ttl_ns=100,
            max_approval_ttl_ns=50,
        ),
    )
    coordinator.create_prompt(
        request,
        principal="runtime.user",
        prompt_ttl_ns=10,
        now_ns=10,
    )
    with pytest.raises(ApprovalSessionError, match="too many"):
        coordinator.create_prompt(
            request,
            principal="runtime.user",
            prompt_ttl_ns=10,
            now_ns=11,
        )
    # Creating after expiry prunes the stale prompt first.
    coordinator.create_prompt(
        request,
        principal="runtime.user",
        prompt_ttl_ns=10,
        now_ns=20,
    )


def test_m6c_to_m6a_to_planner_integration():
    calls = []
    registry = _registry(calls)
    controller = _waiting_controller()
    binder = ExternalIntentBinder(registry, (CAPABILITY,))
    request = binder.bind(controller, _intent())

    principal = "runtime.user"
    authority = ApprovalAuthority(b"m6c-integration-secret-material")
    coordinator = ExternalApprovalCoordinator(authority)
    prompt = coordinator.create_prompt(
        request,
        principal=principal,
        prompt_ttl_ns=1_000,
        now_ns=10,
    )
    token = coordinator.resolve(
        prompt,
        approved=True,
        approval_ttl_ns=500,
        now_ns=20,
    )
    gate = DenyByDefaultAuthorityGate(
        [
            PolicyGrant(
                principal=principal,
                capability_id=CAPABILITY,
                scope=request.scope,
            )
        ],
        approval_authority=authority,
    )
    fabric = ExternalExecutionFabric(
        registry,
        gate,
        ImmutableActionAudit(),
    )
    result = fabric.execute_waiting(
        controller,
        request,
        principal=principal,
        approval=token,
        now_ns=30,
    )
    assert not result.replayed
    assert controller.plan.status == PlanStatus.COMPLETED
    assert controller.plan.step(1).result == "external-ok"
    assert calls == [request.request_digest]


class ScriptedIntentEngine:
    def __init__(self, output):
        self.output = output
        self.prompts = []

    def generate_text(self, prompt, *, max_new_tokens):
        self.prompts.append((prompt, max_new_tokens))
        return self.output

    def embed_text(self, text, *, vector_dim):
        raise AssertionError("external intent must not request an embedding")


def _adapter_request():
    return ExternalIntentRequest(
        plan_id="p",
        goal="g",
        step_id=1,
        objective="write note",
        capabilities=(
            ExternalCapabilityView(
                capability_id="file.write",
                required_scope_keys=("root", "path"),
                optional_scope_keys=(),
                approval_required=True,
                max_payload_utf8_bytes=4096,
            ),
        ),
    )


def test_vn97_adapter_proposes_strict_external_intent():
    engine = ScriptedIntentEngine(
        '{"capability_id":"file.write",'
        '"scope":{"root":"app","path":"note.txt"},'
        '"payload":{"text":"hello"}}'
    )
    adapter = VN97CognitionAdapter(engine)
    intent = adapter.propose_external_intent(_adapter_request())
    assert intent.capability_id == "file.write"
    assert dict(intent.scope) == {"path": "note.txt", "root": "app"}
    assert intent.payload_object() == {"text": "hello"}
    prompt, limit = engine.prompts[0]
    assert prompt.startswith("VN97COG1\noperation=external_intent\n")
    assert '"required_scope_keys":["root","path"]' in prompt
    assert '"approval_required":true' in prompt
    assert limit > 0


@pytest.mark.parametrize(
    "output",
    [
        '{"capability_id":"file.write","scope":{"root":1},"payload":{}}',
        '{"capability_id":"file.write","scope":{},"payload":[]}',
        '{"capability_id":"file.write","scope":{},"payload":{},"extra":1}',
    ],
)
def test_vn97_adapter_external_intent_schema_fails_closed(output):
    adapter = VN97CognitionAdapter(ScriptedIntentEngine(output))
    with pytest.raises(VN97CognitionOutputError):
        adapter.propose_external_intent(_adapter_request())
