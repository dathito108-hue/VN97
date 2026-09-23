# M7P — Deny-by-Default Android Execution Fabric

M7P ports the canonical M6A execution boundary to Android platform code and completes the trusted path from a bound external action request to planner handback.

Canonical path:

`M7O M6ExternalActionRequest → sealed typed registry → exact deny-by-default policy → optional request-bound approval → bounded capability lease → typed handler → hash-chained receipt → M7K planner handback`

The architecture remains:

`Code 1 → Code 2 hardware/mobile-aware upgrade → VN97 production`.

## Trust boundary

M7P executes only an `M6ExternalActionRequest` already bound by M7O from:

- the exact waiting planner step;
- a trusted capability catalog;
- canonical scope;
- canonical payload;
- a Python-compatible request digest.

The model cannot:

- register handlers;
- add policy grants;
- mint approval;
- issue leases;
- choose receipt contents;
- call `recordExternalResult` directly.

## Typed capability registry

`M6TypedCapabilityRegistry` owns:

- one trusted `M6CapabilityDescriptor`;
- one side-effect handler;
- an optional payload validator.

The registry must be sealed before `M6ExternalExecutionFabric` can be constructed.

After sealing, new capability injection is rejected.

Request validation checks:

- exact capability ID;
- all required scope keys;
- no unsupported scope keys;
- payload byte limit;
- optional trusted payload validator.

The same descriptor/request validation is repeated inside lease issuance so direct use of the authority gate cannot bypass the registry validation assumptions.

## Deny-by-default policy

`M6PolicyGrant` is keyed by the exact tuple:

`principal + capability_id + scope_digest`

There is no wildcard or fallback grant.

Missing policy produces `M6AuthorizationDeniedException`.

A grant may retain the descriptor approval requirement or explicitly override it. That decision is trusted host policy, never model output.

## Request-bound approval

When approval is required, the gate verifies the existing `M6ApprovalToken` through `M6ApprovalControllerPort`.

Verification is bound to:

- exact request digest;
- exact principal;
- current time.

Production uses the existing Android Keystore-backed controller through `AndroidApprovalControllerPort`.

M7P introduces no second approval authority.

## Capability leases

After policy and approval validation, the authority gate issues `M6CapabilityLease`.

The lease binds:

- random 16-byte lease ID;
- principal;
- capability ID;
- scope digest;
- issue/expiry time;
- maximum use count.

Effective lifetime and use budget are the minimum of descriptor and policy limits.

Consumption fails closed on:

- unknown/forged lease;
- principal mismatch;
- capability mismatch;
- scope mismatch;
- use outside validity window;
- use-budget exhaustion.

## Execution

`M6ExternalExecutionFabric.executeWaiting(...)` first revalidates the immutable planner binding:

- plan ID matches;
- plan status is `WAITING_EXTERNAL`;
- step ID exists;
- step is `WAITING_EXTERNAL`;
- step kind is `EXTERNAL`;
- immutable objective matches the request.

Only then may authorization and handler execution occur.

A typed handler returns `M6ActionOutcome` with:

- success/failure;
- non-empty result;
- finite confidence in `[0,1]`;
- positive unique evidence IDs;
- retryability.

Successful outcomes are handed back through `NativePlanController.recordExternalResult(...)`.

Typed failures are handed back through `failStep(...)`.

If a handler throws before producing a typed outcome, the step fails non-retryably and the fabric raises `M6ExternalExecutionException`.

## Replay protection

The audit is consulted before lease issuance or handler execution.

If the same exact request digest already has a successful receipt:

- the handler is not called again;
- no new approval/lease is required;
- the durable prior result is replayed into the restored waiting planner step.

This matches canonical M6A idempotent replay behavior and prevents duplicated side effects after recovery.

## Receipts

M7P defines a hash-chained `M6ActionReceipt` contract with statuses:

- DENIED;
- SUCCEEDED;
- FAILED.

Each receipt binds:

- sequence;
- timestamp;
- previous receipt ID;
- request digest;
- plan/step/capability/scope;
- principal;
- lease ID;
- approval ID;
- typed result/confidence/evidence/retryable state;
- error type.

`receipt_id` is SHA-256 over Python-compatible canonical JSON of every field except the ID itself.

Cross-platform fixture:

`e59095fc030b9fba6ab644534ff36bf10776db06a7ffd652c3284be4caf03f84`

M7P also rejects unpaired UTF-16 surrogates while hashing receipt strings so the JVM cannot silently replace malformed text and diverge from Python canonical identity.

## Audit durability boundary

M7P provides `M6InMemoryActionAudit` and the abstract `M6ActionAudit` contract.

Durable append-only, fsync-backed audit storage is deliberately left to M7Q.

This keeps M7P focused on authority/execution semantics while preserving a stable audit interface.

## Android production assembly

`AndroidPlatformRuntime.createExternalExecutionFabric(...)` wires:

- trusted sealed registry;
- explicit policy grants;
- existing Android Keystore approval controller;
- caller-selected audit implementation.

The default audit is in-memory until M7Q supplies the durable implementation.

Android permission enforcement remains in the concrete platform capability handlers/adapters and is not bypassed by this fabric.

## Verification

The isolated local M7P gate was run with `kotlinc -Werror` and prints:

`M7P_EXECUTION_FABRIC_PASS`

Coverage includes:

- Python-compatible M7O request/scope identity;
- Python-compatible receipt identity;
- explicit request-bound approval;
- exact policy grant;
- successful handler execution;
- evidence/result planner handback;
- successful-request replay without a second handler call;
- deny-by-default missing-policy rejection;
- missing-approval rejection;
- malformed surrogate rejection during receipt hashing.

The repository platform host script wires the same M7P regression after M7O.

Full Android/Gradle build is not claimed unless executed separately.

## Next boundary

M7Q should add a durable Android action audit with:

- append-only records;
- canonical JSON;
- hash-chain verification on load;
- fsync before acknowledging a committed side effect;
- torn-final-record rejection;
- bounded recovery;
- replay lookup from durable successful receipts.

That will close the crash-consistency gap between real external effects and replay protection.
