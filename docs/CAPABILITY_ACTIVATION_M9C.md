# M9C — Transactional Activation, Rollback and VN97INV1 Inventory

M9C completes the controlled capability-acquisition chain:

`VN97CAP1 stage → VN97SIG1 trust → compatibility plan → transactional activation → VN97INV1`

It is the first M9 block allowed to change a live capability backend.

## Mandatory trust revalidation

A previous M9B `VerifiedCapability` is not sufficient by itself.

At activation time the coordinator receives the original detached signature envelope plus the
current stage root, trust store and signature verifier. It calls M9B verification again, which
reopens/reparses the staged bytes and reapplies current publisher scope/version/revocation policy.

The freshly verified object must equal the caller's previously verified object. The compatibility
plan is then recomputed from that fresh object and current profile/adapters and must exactly equal
the submitted plan.

Thus these all fail before backend preparation:

- staged package bytes changed after M9B verification;
- publisher key revoked;
- trust scope/version changed;
- compatibility profile changed;
- adapter selection changed;
- stale/tampered compatibility plan.

## Trusted backend transaction contract

`CapabilityActivationBackend` exposes:

- `prepare(verified, plan) → PreparedActivation`;
- `commit(token) → runtime_revision`;
- `inspect(token) → BackendStatus`;
- `rollback(token)`.

The backend is selected by trusted host/runtime code, never by model output.

Prepare must build/validate the candidate artifact without changing the live runtime. Commit is
the live-state mutation boundary. Rollback must be idempotent and capable of reversing a committed
transaction to the runtime revision that preceded that transaction. Inspect must distinguish
PREPARED, COMMITTED and ROLLED_BACK after process restart.

## Write-ahead protocol

Only one unresolved inventory transaction is allowed at a time.

Activation:

1. current trust/package/plan is revalidated;
2. VN97INV1 writes a `reserved` pending transaction;
3. backend prepare runs;
4. returned backend token + artifact SHA-256 are persisted as `prepared`;
5. backend commit runs;
6. runtime revision and full provenance are atomically finalized into the activation stack.

If prepare raises a normal exception, the reservation is cleared. If commit fails and rollback
succeeds, the prepared record is cleared. If process death or a non-recoverable backend error
interrupts this sequence, the durable pending record remains for `recover()`.

## Crash recovery

For a reserved transaction, no live mutation was permitted yet, so recovery safely clears it.

For a prepared activation:

- COMMITTED → finalize activation using persisted identity + backend revision;
- PREPARED → invoke rollback, then clear;
- ROLLED_BACK → clear.

For a pending rollback:

- if the backend has not yet rolled back, invoke rollback;
- once rolled back, pop the corresponding inventory stack entry.

Recovery requires the exact backend ID referenced by the pending transaction. Missing backends
fail closed and keep the pending record intact.

## Activation stack and rollback

VN97INV1 stores up to 64 committed activation records per capability. The public active view is
only the top record.

If v1 is active, then v2 commits, the stack contains v1 → v2. Rolling back v2 calls the backend
rollback token for v2 and then pops v2; v1 becomes active again with its original provenance,
artifact digest and runtime revision.

Rolling back the final entry returns the capability to no active imported version.

Exact already-active package/signature/profile/plan/backend activation is idempotent and does not
prepare a second backend transaction. Same-version replacement and downgrades are denied by
default; trusted host code must opt in explicitly.

## VN97INV1 persistence

Inventory is strict canonical UTF-8 JSON with:

- schema `VN97INV1`;
- monotonic generation;
- sorted per-capability activation stacks;
- bounded event history;
- one optional pending WAL transaction.

Each committed activation records capability/package version, package SHA-256, publisher key,
signature digest, compatibility profile ID/fingerprint, complete plan digest, runtime API,
backend/artifact/revision and source origin/hash/license.

Backend tokens are persisted internally because rollback/recovery require them, but
`InventorySnapshot` deliberately omits them.

Reference bounds:

- inventory file: 4 MiB;
- stack depth: 64 per capability;
- history: 4096 events.

The inventory root must already exist and must not be a symlink. Reads use O_NOFOLLOW and require
a bounded regular file. Writes use a same-directory random 0600 temporary, fsync, atomic replace
and directory fsync.

## Authority boundary

M9C activation changes only a trusted capability backend selected by host code.

It does not:

- let cognition choose/mint a backend;
- register an M6 external capability handler;
- grant policy, approval or leases;
- download packages;
- create arbitrary file/network authority;
- load package-provided executable/plugin/native-library sections.

External package acquisition remains M6-governed. M9A still rejects executable sections.

## Verification

The isolated M9C fault-injection suite passes 9/9 tests plus Python bytecode compilation.

Coverage includes:

1. activation idempotence, v1→v2 upgrade and real v2→v1 rollback;
2. stale-plan, same-version replacement and downgrade denial before prepare;
3. current trust revocation and staged-byte mutation rejection before prepare;
4. crash after reservation and safe recovery;
5. crash after backend commit and provenance-preserving finalize recovery;
6. crash after backend rollback and previous-version restoration;
7. missing recovery backend fail-closed and public token non-disclosure;
8. noncanonical/symlink inventory corruption rejection;
9. pending WAL identity tamper rejection.

The two production M9C module blobs committed to the branch are byte-identical to the locally
tested snapshots.
