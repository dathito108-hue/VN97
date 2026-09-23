# M7O — External Intent + Approval Handoff

M7O connects the Android cognition/planner path to the existing M6 trust boundary without moving effectful execution into the model or runtime module.

Canonical path:

`M7N WAITING_EXTERNAL → M7J typed external intent → trusted platform binder → immutable M6 action request → request-bound approval prompt`

The architecture remains:

`Code 1 → Code 2 hardware/mobile-aware upgrade → VN97 production`.

## Module boundary

The implementation lives in `android/platform`, which already depends on `android/runtime`.

`android/runtime` does not depend back on platform.

This preserves the existing direction:

`runtime → pure model/planner state`
`platform → permissions / approval / external-effect integration`

## Trusted capability catalog

`M6CapabilityDescriptor` is host-owned trusted metadata. It declares:

- capability ID;
- required scope keys;
- optional scope keys;
- default approval requirement;
- payload byte bound;
- maximum lease lifetime;
- maximum lease use count.

Only descriptors passed to `M6ExternalIntentBinder` are visible to cognition.

The model receives `NativeExternalCapabilityView` values, never authority policy grants, approval keys, leases, Android permissions, or handlers.

## WAITING_EXTERNAL snapshot binding

The binder accepts only a canonical planner state with:

- plan status `WAITING_EXTERNAL`;
- exactly one step in `WAITING_EXTERNAL`;
- that step's immutable kind is `EXTERNAL`.

The backend request is rebuilt from trusted planner state:

- plan ID;
- goal;
- exact step ID;
- exact immutable objective;
- trusted capability views.

`proposeAndBind(...)` snapshots this request before model inference and recomputes it afterwards. If the waiting step changed while the model proposed an intent, binding fails closed.

## Intent validation

Untrusted `NativeExternalIntent` is checked against the trusted catalog:

- capability must be advertised;
- scope entry count is bounded;
- required scope keys must be present;
- unsupported scope keys are rejected;
- scope values are bounded;
- payload size is bounded by both global and descriptor limits.

The result is `M6ExternalActionRequest`, which cannot be constructed from model data without passing the binder.

## Python-compatible authority identity

`M6CapabilityScope` sorts scope entries exactly like Python M6A and computes:

`SHA256({"scope":[[key,value],...]})`

`M6ExternalActionRequest.requestDigest` is SHA-256 over canonical JSON with the same fields as Python M6A:

- capability_id;
- objective;
- payload object;
- plan_id;
- sorted scope entry pairs;
- step_id.

A cross-platform fixture is locked in the host test.

For:

- plan ID = `ab` repeated 32 bytes;
- step ID = 3;
- objective = `send exact external action`;
- capability = `test.echo`;
- scope = `mode=fast,target=alpha`;
- payload = `{"message":"hello","n":1}`;

the scope digest is:

`c1ccf2848b5cc4a0aa30dc1611c869d85205af7a61bc52a2520078e6572b02eb`

and the request digest is:

`6c9db355eba8988e1ccc8df7515b03b10cb75c4e57e66fb8c62881b581e54184`

## Approval handoff

`M6ExternalApprovalHandoff` creates the existing M6 approval prompt from:

- exact request digest;
- principal;
- canonical human-presentable request JSON.

Approval remains one-shot and request-bound through the existing M6 approval authority.

A small `M6ApprovalControllerPort` keeps the pure JVM handoff test independent of Android Keystore.

Production uses:

`AndroidApprovalControllerPort → AndroidApprovalController → Android Keystore HMAC`

so no second approval authority is introduced.

`AndroidPlatformRuntime.externalApprovals` exposes this production handoff.

## Deliberate non-goals

M7O does not:

- execute handlers;
- issue capability leases;
- define policy grants;
- persist action receipts;
- infer approval from `approvalRequired`;
- bypass Android permissions;
- call `recordExternalResult`.

Those are the next M6 execution-fabric boundary.

The capability descriptor's approval flag is only cognition-visible contract metadata. Final authorization remains deny-by-default policy owned by trusted M6 execution code.

## Verification

Isolated local PASS with `kotlinc -Werror`:

`M7O_EXTERNAL_HANDOFF_PASS`

The fixture covers:

- Python-compatible scope digest;
- Python-compatible action request digest;
- canonical approval presentation;
- trusted catalog binding;
- unsupported capability rejection;
- invalid scope rejection;
- typed backend propose-and-bind;
- approval prompt digest binding;
- approval token verification handoff.

The repository platform host script wires the same M7O test after the existing M6 approval compatibility test.

Full Android/Gradle execution must not be inferred unless that gate is actually run.

## Next boundary

M7P should port the remaining canonical M6A execution layer to Android platform:

`M6ExternalActionRequest → deny-by-default policy → optional request-bound approval → bounded lease → handler → immutable receipt → NativePlanController.recordExternalResult/failStep`.

This keeps every external side effect outside cognition and preserves the existing M6 authority model.
