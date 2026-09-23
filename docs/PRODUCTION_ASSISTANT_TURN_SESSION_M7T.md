# M7T — Production Assistant Turn Session

M7T turns the M7S end-to-end external coordinator into one bounded production-facing assistant-turn lifecycle without changing the locked VN97 architecture:

`Code 1 → Code 2 hardware/mobile-aware upgrade → VN97 production`.

The production assistant path is now:

`activated VN97 model → VN97COG1 native cognition → canonical planner → sovereign memory/evidence → M7O exact external intent → M7P authority/lease → M7Q audit → M7R Android handler → planner handback → final response`.

No Transformer/LLaMA/third-party AI backend, second planner, second authority fabric or model-side approval decision is introduced.

## `VN97AssistantSession`

`VN97AssistantSession` is a host-facing state machine above `M6EndToEndExternalCoordinator`.

It owns only turn lifecycle state. It does not own:

- model weights or another inference backend;
- authority policy;
- approval keys or token minting;
- capability registration;
- Android permissions;
- external side-effect handlers.

Those remain in the already-canonical M7S/M6 layers.

A session is single-flight. Starting a second user turn while one is still reasoning, yielded or waiting for approval fails closed instead of interleaving two planners through one UI turn.

## Bounded user-facing boundaries

A turn reports one of:

- `COMPLETED` — canonical RESPOND result is available;
- `APPROVAL_REQUIRED` — exact M6 request presentation must be resolved by trusted host UI;
- `APPROVAL_REJECTED` — explicit denial consumed the pending prompt and failed the external step closed;
- `YIELDED` — bounded cognition or external-handoff budget reached, safe to continue the same turn;
- `PAUSED`;
- `FAILED`;
- `CANCELLED`;
- `BUDGET_EXHAUSTED`;
- `STALLED`.

Only a completed turn exposes `finalResponse`. Only `APPROVAL_REQUIRED` exposes an approval object.

## Approval identity

`VN97AssistantApproval` is only a presentation/continuation wrapper around the already-bound `M6PendingExternalApproval` from M7S. It exposes:

- plan ID;
- step ID;
- capability ID;
- scope digest;
- request digest;
- canonical presentation JSON;
- expiry timestamp.

Resolving approval requires the same active turn object and the exact pending approval object. The session never reconstructs approval from model text and never asks cognition to regenerate the already-bound request while the prompt is being resolved.

## Multiple external actions

One plan may contain more than one EXTERNAL step. M7T automatically continues through successful no-approval/replayed external handbacks until it reaches a user-visible boundary, but the number of external handoffs per host advance is bounded.

If the bound is reached while another external step is waiting, the session yields rather than looping without limit.

M7P/M7Q replay ordering remains unchanged: a successful receipt is checked before a new approval, lease or handler call.

## Cold-process planner recovery

`resumeRestoredTurn(...)` accepts the canonical `NativePlanController` restored from VN97PLN1/M7M.

M7T intentionally does **not** persist or resurrect an old in-memory approval prompt. On a restored `WAITING_EXTERNAL` plan:

1. M7O rebinds the current immutable waiting step;
2. M7P/M7Q first replay an exact prior success if one exists;
3. otherwise the current authority policy is evaluated again;
4. if approval is still required, a fresh request-bound prompt is created.

This preserves M7M's rule that continuity metadata cannot mint or restore authority.

## Production activated-model factory

`AndroidPlatformRuntime.createProductionAssistantSession(...)` accepts a `NativeActivatedModel`, not a generic AI backend.

It constructs exactly:

`NativeActivatedModel → NativeCognitionInferenceEngine → NativeTypedCognitionAdapter → M7S coordinator → VN97AssistantSession`.

The same production factory also reuses:

- M7R's sealed Android capability registry and binder;
- Android Keystore-backed approval controller;
- M7P deny-by-default policy/lease gate;
- M7Q durable no-backup action audit.

This makes the production assistant route explicit and prevents accidental replacement of VN97 cognition with an unrelated model/backend at this assembly boundary.

## Bounds

`VN97AssistantSessionLimits` bounds:

- one user-turn UTF-8 size;
- cognition cycles per host advance;
- automatic external handoffs per host advance;
- approval-prompt TTL;
- approval-token TTL.

Planner transition/retry/memory budgets remain independently enforced by M7K/M7N.

## Verification

A local isolated `kotlinc -Werror` gate was executed before upload and printed:

`M7T_PRODUCTION_ASSISTANT_TURN_SESSION_PASS`

It covers:

- immediate completion;
- single-flight rejection of a second active turn;
- yield + continuation;
- approval continuation;
- explicit rejection;
- multiple bounded external handoffs;
- re-entry with a restored non-terminal planner.

The repository host regression additionally uses the real M7N/M7O/M7P/M7S classes and verifies:

- no side effect before approval;
- request-bound approval resumes to final RESPOND;
- successful audit replay after planner restoration executes no second effect and creates no second prompt;
- rejecting an already-bound prompt does not invoke `EXTERNAL_INTENT` again.

The repository currently has no `.github/workflows` directory, so these host scripts are not represented as GitHub Actions checks unless a host explicitly runs them.
