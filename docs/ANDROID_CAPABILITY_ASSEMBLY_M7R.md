# M7R — Production Android Capability Assembly

M7R connects the concrete Android app/device side effects that already existed in `android/platform` to the canonical M7O/M7P/M7Q trust path. It does not add a second authority system and does not move effects into cognition.

Canonical production path:

`M7N WAITING_EXTERNAL → M7O typed/bound request → M7R trusted capability catalog → M7P deny-by-default policy + request-bound approval + bounded lease → M7R typed Android handler → Android permission broker → concrete Android effect → M7Q durable receipt → planner handback`

The locked model architecture remains:

`Code 1 → Code 2 hardware/mobile-aware upgrade → VN97 production`.

## One trusted catalog

`M6AndroidProductionCapabilities` owns the production descriptors used by both:

- `intentBinder`, which exposes capability views to the typed cognition adapter; and
- `createSealedRegistry()`, which registers the corresponding trusted host handlers.

This prevents descriptor drift between what cognition is allowed to request and what Android can execute.

The registry is sealed before it is returned to the execution fabric.

## Production capabilities

M7R intentionally starts with the two Android effects that already existed before this milestone.

### `app.launch`

- required scope: `package`;
- payload: exactly `{}`;
- package is a bounded canonical ASCII Android package name;
- approval required by the descriptor;
- lease lifetime capped at 30 seconds;
- lease uses capped at one.

The exact package lives in `M6CapabilityScope`, so the M7P policy tuple remains exact:

`principal + app.launch + scope_digest(package)`.

### `device.clipboard.write`

- required scope: `channel=system-clipboard`;
- payload: exactly one canonical JSON string field: `text`;
- decoded clipboard text is capped at 16 KiB UTF-8;
- alternative/non-canonical JSON escaping and extra fields are rejected;
- approval required by the descriptor;
- lease lifetime capped at 30 seconds;
- lease uses capped at one.

The clipboard payload remains part of the M6 request digest, so each approval token is bound to the exact text request even though policy scope grants the system clipboard channel.

## Android effect boundary

`AndroidAppDeviceAdapter` implements the narrow `M6AndroidActionPort` used by the trusted handlers.

It still calls `AndroidPermissionBroker.requireGranted(...)` immediately before each real side effect. M6 authority and Android runtime permissions therefore remain separate mandatory gates:

1. M6 decides whether the principal/request/scope has authority and approval;
2. Android permission broker verifies operating-system permission requirements;
3. only then does the adapter perform the app launch or clipboard write.

No model output receives the adapter, permission broker, approval key, lease state, or handler reference.

## Production runtime assembly

`AndroidPlatformRuntime.externalIntentBinder` exposes the single trusted production intent catalog.

`createProductionExternalExecutionFabric(...)` creates a fresh sealed production registry and uses M7Q durable audit in `Context.noBackupFilesDir`.

Using the no-backup directory preserves receipts across process death/reboot while avoiding Android backup/restore of successful side-effect receipts into a different installation.

The existing generic in-memory and explicit durable factories remain available for tests and specialized hosts.

## Replay ordering

M7R does not modify M7P execution ordering. A successful M7Q receipt is still checked before approval, lease issuance/consumption, and handler invocation. The M7R host test explicitly replays an `app.launch` request and verifies that the action port is not called a second time.

## Verification

The isolated M7R core gate was run locally with `kotlinc -Werror` and printed:

`M7R_PRODUCTION_CAPABILITY_ASSEMBLY_PASS`

The repository host test additionally compiles M7R against the canonical M6 approval, M7O handoff, M7P execution fabric, and M7P runtime stubs. It covers:

- the exact two-capability production catalog;
- the same descriptors feeding binder and registry;
- mandatory approval and one-use/30-second lease bounds;
- sealed registry rejection of late registration;
- exact-package app launch handler binding;
- successful M7P receipt + planner handback;
- successful-receipt replay without a second side effect;
- invalid package rejection;
- canonical clipboard JSON parsing and Unicode/escape handling;
- fixed clipboard channel scope;
- extra-field/non-canonical escape rejection;
- 16 KiB decoded clipboard bound.

M7R deliberately does not add file, network, shell, accessibility, or arbitrary-intent capabilities. Those should be introduced only as separate typed descriptors/handlers with explicit Android platform constraints and M6 policy scopes.
