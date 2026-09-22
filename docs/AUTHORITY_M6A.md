# M6A — Typed Capability Registry + Deny-by-Default Authority Gate

M6A is the first VN97 Tool + Authority Fabric milestone. It connects the M5
`WAITING_EXTERNAL` lifecycle boundary to registered side-effect handlers without giving the
cognition model direct authority to execute tools.

## Trust boundary

M5 remains unchanged:

- cognition may propose an `EXTERNAL` plan step;
- `CognitionLoop` stops before backend/model execution for that step;
- `PlanController.begin_step()` moves it to `WAITING_EXTERNAL`;
- only M6 may supply a result through `PlanController.record_external_result()`.

M6A therefore treats cognition output as untrusted intent. Capability registration, policy,
approval verification, lease accounting, handler invocation and audit are trusted runtime
responsibilities outside the model.

## Typed action request

`ExternalActionRequest` binds one external action to:

- immutable `plan_id` and positive `step_id`;
- the exact immutable planner step objective;
- a registered `capability_id`;
- canonical capability scope;
- strict canonical JSON payload.

The SHA-256 `request_digest` covers all of those fields. A change to the payload, target scope,
capability, plan or objective creates a different authorization object.

## Typed capability registry

`TypedCapabilityRegistry` maps a capability ID to:

- a `CapabilityDescriptor`;
- one side-effect handler;
- an optional trusted payload validator.

Descriptors declare required/optional scope keys, a payload byte bound, whether approval is
required, and maximum lease lifetime/use count. Unknown capabilities fail closed. The registry
must be sealed before `ExternalExecutionFabric` can execute anything, preventing runtime
capability injection after assembly.

M6A intentionally defines the registry/execution contract rather than production internet,
file, app or device implementations. Those concrete capabilities can be added in M6B without
changing the authority boundary.

## Deny-by-default policy

`DenyByDefaultAuthorityGate` contains explicit `PolicyGrant` values. A grant matches only the
exact tuple:

`principal + capability_id + scope_digest`

There is no wildcard fallback. Missing policy means deny.

A capability descriptor may require approval. Policy can retain that requirement or explicitly
configure a trusted no-prompt capability when the runtime owner chooses to do so. The gate does
not accept a model assertion as approval.

## Request-bound approval

`ApprovalAuthority` issues bounded `ApprovalToken` values using HMAC-SHA256. A token is bound to:

- issuer;
- principal;
- random approval ID;
- exact action request digest;
- issue time;
- expiry time.

Substituting a different payload, path/scope, step or plan invalidates the approval. Approval
verification uses constant-time digest comparison.

The HMAC secret is supplied by the trusted host runtime. M6A does not prescribe Android Keystore
storage yet; secure platform key storage belongs to the Android/runtime integration milestone.

## Capability leases

After policy and approval validation, the authority gate issues a `CapabilityLease`. Leases are
bound to principal, capability and scope digest, and are limited by both descriptor and policy:

- maximum lifetime;
- maximum use count.

Lease usage is tracked by the issuing gate. A fabricated, foreign, expired, exhausted or
scope-mismatched lease fails closed. Process-local leases are deliberately not restored after a
restart; the runtime must re-authorize unfinished work unless an already-successful receipt can
be replayed.

## Immutable action audit

`ImmutableActionAudit` stores frozen `ActionReceipt` records. Receipts include the request digest,
planner identity, capability/scope, principal, lease/approval IDs, outcome and previous receipt
ID. Every receipt ID is SHA-256 over canonical receipt content, forming a contiguous hash chain.

When a path is configured, records are append-only JSONL and are `fsync`ed before being exposed
in memory. Startup verifies field shape, sequence continuity, previous-receipt linkage and
receipt digest. A torn or modified record fails closed.

The audit hash chain is an integrity/idempotency mechanism, not a cryptographic author signature
against an attacker who can rewrite the entire audit file. Authorization authenticity comes
from the trusted approval secret and runtime policy boundary.

## External execution sequence

`ExternalExecutionFabric.execute_waiting()` performs this order:

1. verify that request `plan_id`, `step_id`, objective and `EXTERNAL` kind exactly match the
   planner step currently in `WAITING_EXTERNAL`;
2. validate the typed capability request in the sealed registry;
3. check for a prior successful receipt for the exact request digest;
4. otherwise validate policy/approval, issue or validate a bounded lease, and consume one use;
5. invoke the registered handler only after authorization succeeds;
6. append a `DENIED`, `FAILED` or `SUCCEEDED` receipt;
7. only after a successful receipt exists, deliver the typed result to
   `PlanController.record_external_result()`.

Denied authorization never calls the handler and leaves the planner waiting. A handler exception
is receipted and fails the planner closed. A typed unsuccessful outcome uses its explicit retry
policy through the existing M5 planner state machine.

## Crash/replay safety

The successful receipt is appended before M6 mutates the planner with the external result. If a
process dies after a side effect succeeds but before the planner checkpoint advances, the same
request digest finds the prior `SUCCEEDED` receipt on restart and replays the recorded result
without invoking the side effect a second time.

This gives M6A an idempotency boundary for exact request replay. It does not make arbitrary
third-party effects transactional; capability implementations still need their own stronger
idempotency semantics where applicable.

## Mobile-first properties

- no third-party AI/service dependency is introduced;
- bounded payloads, scopes, lease lifetimes and use counts;
- no unbounded model-owned tool state;
- small canonical JSON/HMAC/SHA primitives only;
- append-only audit suitable for compact local persistence;
- production tool implementations can remain platform-native;
- cognition receives no registry handler, policy table, approval secret or lease-mutation API.

## M6A non-goals

M6A does not yet add production HTTPS, file, app-launch, clipboard/device or Android permission
brokers. It also does not define UI approval surfaces or Android Keystore integration. Those are
subsequent M6/M7 layers built on this fixed authority contract.
