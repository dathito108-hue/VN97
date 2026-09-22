# M7B2 — Android Authority + Platform Adapter

M7B2 realizes two platform-specific pieces that M6 intentionally left outside cognition:
approval-key custody and concrete Android app/device actions.

It does not change the M6 authorization order.

## Trust path

The intended production path remains:

`WAITING_EXTERNAL → M6C bound request → M6 policy → exact approval → lease → M6B handler
→ Android platform adapter → receipt → planner`

Android permission checks and AndroidKeyStore are implementations inside that path, not alternate
authority systems.

## AndroidKeyStore approval key

`AndroidKeystoreApprovalMac` creates or loads a secret key under a bounded alias in the
`AndroidKeyStore` provider.

The key algorithm is HmacSHA256 and its authorized use is MAC generation. The secret key object
is passed directly to `javax.crypto.Mac`; raw key bytes are never requested or exported.

The process-global creation lock prevents two VN97 controller instances in one process from
racing key creation for the same alias.

## Exact M6 approval compatibility

M7B2 deliberately preserves the M6 reference token fields:

- `approval_id`;
- `issuer`;
- `principal`;
- `request_digest`;
- `issued_ns`;
- `expires_ns`;
- `signature`.

The canonical MAC payload uses the same sorted JSON key order as Python M6:

`{"approval_id":"…","expires_ns":…,"issued_ns":…,"issuer":"…","principal":"…","request_digest":"…"}`

Identifiers use the same restricted M6 character set, approval IDs are 16-byte lowercase hex,
request digests are lowercase SHA-256 hex, and signatures are lowercase HMAC-SHA256 hex.

Android default timestamps are epoch wall-clock nanoseconds derived from
`System.currentTimeMillis()`, matching the epoch semantics of the reference M6 authority.
Approval TTL arithmetic is overflow checked.

A deterministic JVM compatibility regression verifies the canonical payload and this known
HMAC-SHA256 vector:

- approval id: `00112233445566778899aabbccddeeff`
- principal: `runtime.user`
- request digest: 64 lowercase `a` characters
- issued/expires: 100 / 200
- signature:
  `16006c3472602963c8d2a42cb03c3e331974d96ff6a76bc1ff60c3313f0923a3`

## One-shot Android approval controller

The raw `ApprovalMac`, `M6ApprovalAuthority` and `M6ApprovalCoordinator` are internal module
types. Public Android code receives `AndroidApprovalController`.

A controller may:

- create one bounded prompt for an exact request digest;
- resolve that exact prompt as approved or denied;
- verify an already created token;
- inspect the bounded pending-prompt count.

Resolution removes a prompt before token minting. Therefore duplicate or simultaneous callbacks
can mint at most one token. Denial returns no token. Expired, stale or modified prompts fail
closed.

Presentation JSON is display data supplied by the trusted M6C/runtime layer. The signature binds
the exact request digest, not model-authored prose.

## Android permission broker

`AndroidPermissionBroker` maps a trusted capability ID to an optional set of Android runtime
permissions. It can report missing permissions or fail execution if any requirement is absent.

The broker never requests permissions itself. Activity/UI code owns permission dialogs.

Most importantly:

`Android permission granted != VN97 authority granted`

An OS permission does not create policy, approval, lease or audit state.

## Exact package launch

The adapter validates the same Android package grammar used by M6B.

On Android 13 / API 33 and later it calls
`PackageManager.getLaunchIntentSenderForPackage(package)`. That API returns a launch
`IntentSender` without relying on ordinary package visibility queries.

On API 26–32 it uses `getLaunchIntentForPackage(package)`, adds
`FLAG_ACTIVITY_NEW_TASK`, and fails closed if the package manager cannot return a visible launch
activity. M7B2 does not request broad installed-package visibility merely to weaken that failure.

No arbitrary component, URI, action string or shell command is accepted.

## Clipboard write

Clipboard text is UTF-8 bounded to 16 KiB, matching M6B. The adapter obtains the system
`ClipboardManager` and writes one plain-text clip.

VN97 marks the clip as sensitive. API 33+ uses `ClipDescription.EXTRA_IS_SENSITIVE`; older
supported Android versions use the documented compatibility key.

No Android permission is added by the platform library for this operation.

## Visibility and API surface

`AndroidAppDeviceAdapter` and `PlatformCapabilityIds` are internal module types.
`AndroidPlatformRuntime` exposes the approval controller publicly but keeps the permission
broker and side-effect adapter internal.

This prevents the library from becoming a general app-launch/clipboard utility surface. A later
runtime assembly layer must call those internal objects only after the M6 decision path succeeds.

The `:platform` manifest is empty and declares no permission.

## Verification performed

Before commit, the exact pure-Kotlin approval source was compiled with warnings as errors and
passed:

- the deterministic Python-compatible HMAC vector;
- token verification;
- prompt count/pruning behavior;
- one-shot resolution;
- concurrent double-resolution (exactly one token);
- denial with no token.

Android-specific source was separately compiled with warnings as errors against API-shaped host
stubs, and a behavioral adapter regression passed:

- API 33 launch-sender path;
- API 32 fallback path;
- invalid package rejection;
- missing OS permission rejection;
- sensitive clipboard marking;
- missing launch target rejection;
- clipboard size bound.

The Android framework/Keystore implementation still requires Android SDK/device CI for final
platform verification.
