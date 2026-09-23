# M10C — Explicit M6 Approval UI

M10C exposes the existing M6 approval boundary to the installable Android app without moving
authority into the model.

## Narrow default policy

The app enables exactly one production policy eligibility by default:

- capability: `device.clipboard.write`;
- scope: `channel=system-clipboard`;
- approval required: always true;
- one use per lease;
- maximum 30-second lease.

The fixed grant is built by
`M6AndroidProductionCapabilities.userApprovedClipboardGrant(...)`.

`app.launch` remains deny-by-default because its package scope is dynamic and M10C does not
derive policy grants from model-proposed scopes.

## Approval identity

When the canonical M7 session returns `APPROVAL_REQUIRED`, `VN97AppAssistant` retains the exact
turn/update object in process and exposes only a presentation view:

- capability ID;
- request digest;
- scope digest;
- canonical M6 presentation JSON;
- expiry timestamp.

The app does not reconstruct or edit the pending request.

Approve/Reject calls `VN97AssistantSession.resolveApproval(...)` with the exact
`VN97AssistantApproval` object produced by M6. The Android Keystore approval controller remains
the token issuer/verifier.

A chained second approval is allowed only when M6 produces another `APPROVAL_REQUIRED` boundary.

## UI

The Activity shows the exact M6 presentation JSON and explicit Approve / Reject buttons only while
the app phase is `WAITING_APPROVAL`.

Reject returns the app to READY and records a system event, not a fabricated model response.

Approval is never triggered by avatar interaction or model output alone.

## Verification

Pure app state gate:

`M10C_APPROVAL_STATE_PASS`

The existing M7R host regression is also extended to assert that the production clipboard grant
has the exact canonical scope digest and mandatory one-use approval policy.

Full Android Keystore/UI instrumentation remains a device gate and is not inferred from host tests.
