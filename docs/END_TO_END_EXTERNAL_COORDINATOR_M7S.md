# M7S — End-to-End External Coordinator

M7S closes the Android external-action control loop without changing the locked model architecture:

`Code 1 → Code 2 hardware/mobile-aware upgrade → VN97 production`.

It connects the already-merged canonical pieces rather than creating a second planner, authority system, or tool runtime:

`M7N cognition boundary → M7O typed external intent/binding → M7P deny-by-default authority/execution → M7Q durable audit → M7R Android handlers → planner handback → M7N cognition resumes`.

## Single cognition backend

`M6EndToEndExternalCoordinator` is constructed from one `NativeTypedCognitionAdapter` and internally constructs the `NativeCognitionLoop` from that exact adapter. External intent proposal therefore uses the same typed VN97 cognition backend as planning/reasoning; M7S does not introduce another AI backend.

## Authority remains the source of truth

M7S does not copy `approvalRequired` logic and does not let cognition decide whether approval is needed.

When cognition reaches `WAITING_EXTERNAL`, M7S binds the model proposal through `M6ExternalIntentBinder` and submits the exact request to the existing M7P execution fabric without an approval token.

- If exact M7P policy permits execution without approval, execution proceeds immediately.
- If the trusted M7P gate throws `M6ApprovalRequiredException`, M7S creates the existing request-bound M7O prompt.
- Other authority failures keep their existing M7P behavior and durable DENIED receipt semantics.

This preserves descriptor defaults plus policy overrides as one authority decision path.

## Approval continuation

`M6PendingExternalApproval` carries only host-side immutable bindings needed to resume the same external request:

- plan ID;
- step ID;
- capability ID;
- scope digest;
- request digest;
- principal;
- approval prompt/presentation.

Before resolving a prompt, M7S revalidates that the controller is still at the same canonical `WAITING_EXTERNAL` plan/step/objective. Approval resolution never calls cognition again and never asks the model to reconstruct the pending request.

An approved token is handed to the same M7P execution fabric, which verifies the exact request digest/principal and enforces the bounded lease before the handler runs.

Explicit user rejection performs no side effect and fails the active external planner step closed with `retryable=false`.

## Replay ordering

M7S does not alter M7P replay ordering. A restored plan still binds its exact typed request, after which `M6ExternalExecutionFabric` checks M7Q for a prior successful receipt before approval, lease issuance/consumption, or handler invocation.

Therefore a successfully receipted request can resume cognition after process restoration without a second external effect or a second approval prompt.

## Production Android assembly

`AndroidPlatformRuntime.createProductionExternalCoordinator(...)` assembles:

- the caller-provided canonical VN97 typed cognition adapter;
- M7R's single production capability catalog/binder;
- Android Keystore-backed M6 approval handoff;
- M7R sealed production registry;
- M7P deny-by-default authority;
- M7Q durable no-backup audit.

External effects remain outside cognition/model code and still pass through Android permission checks inside the M7R handlers.

## Verification

The isolated coordinator gate was run locally with `kotlinc -Werror` and printed:

`M7S_END_TO_END_EXTERNAL_COORDINATOR_PASS`

The repository host gate compiles M7S with the full canonical Android runtime sources and covers:

- `WAITING_EXTERNAL` cognition boundary;
- typed external intent proposal/binding;
- authority-triggered approval prompt creation;
- no side effect before approval;
- request-bound approval continuation;
- planner result handback and final RESPOND continuation;
- successful receipt replay without a second handler call or prompt;
- explicit rejection without a handler call;
- no second `EXTERNAL_INTENT` model call while resolving an already-bound prompt.
