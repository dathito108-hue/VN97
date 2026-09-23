# M7Q — Durable Android Action Audit

M7Q closes the crash-replay gap in the Android M6 execution path by replacing the optional in-memory receipt history with a durable append-only audit implementation.

Canonical effect path remains:

`M7O bound request → M7P deny-by-default authority / lease / handler → M7Q durable receipt → planner handback`

The architecture remains:

`Code 1 → Code 2 hardware/mobile-aware upgrade → VN97 production`.

M7Q does not change policy, approval, lease, capability, planner, model, or handler semantics.

## Durable receipt format

`M6DurableActionAudit` implements the existing `M6ActionAudit` interface.

Each receipt is persisted as exactly one canonical UTF-8 JSON object followed by `\n`.

The on-disk object contains the same M7P receipt fields plus `receipt_id`, in canonical sorted-key order:

- approval_id
- capability_id
- confidence
- error_type
- evidence_record_ids
- lease_id
- plan_id
- previous_receipt_id
- principal
- receipt_id
- request_digest
- result
- retryable
- scope_digest
- sequence
- status
- step_id
- timestamp_ns

The receipt ID remains SHA-256 over the M7P canonical payload that excludes `receipt_id`.

No new receipt identity is introduced.

## Cross-platform parity

The M7Q fixture uses the same M7P successful receipt identity:

`e59095fc030b9fba6ab644534ff36bf10776db06a7ffd652c3284be4caf03f84`

The complete first JSONL record, including its trailing newline, is exactly 654 bytes and has SHA-256:

`34ca083dace3b8ccaf62ab443446bf84f8a21f360a064201df433d06578e2536`

This locks both the receipt hash and the durable representation to the Python M6 canonical contract.

## Append commit rule

A new receipt is acknowledged only after:

1. constructing the canonical receipt;
2. parsing the exact bytes through the recovery parser as a self-check;
3. verifying configured record/file bounds;
4. confirming the existing target has not changed outside this audit instance;
5. opening the target with append + `NOFOLLOW_LINKS`;
6. writing every record byte including the final newline;
7. `FileChannel.force(true)` on the audit file;
8. forcing the containing directory;
9. re-checking file size/identity;
10. updating the in-memory committed receipt list.

The receipt is therefore not exposed as committed in memory before the durable write has been forced.

## Recovery

Startup recovery is bounded and fail-closed.

It validates:

- root is a real non-symlink directory;
- target is a real regular file and not a symlink;
- file size is within the configured maximum;
- bounded read completes without file mutation;
- non-empty files end in a newline;
- no empty records;
- each record fits the configured byte bound;
- total record count is bounded;
- strict UTF-8;
- exact fixed receipt schema/order;
- strict JSON string escapes and surrogate pairs;
- finite confidence in `[0,1]`;
- positive unique evidence IDs;
- lowercase SHA-256 fields;
- positive sequence/step IDs;
- non-negative timestamp;
- receipt ID recomputation;
- exact canonical record re-encoding;
- contiguous sequence numbers;
- exact previous-receipt hash chain.

There is no truncate/repair mode and no heuristic fallback that silently discards a corrupt tail.

A torn, tampered, reordered, non-canonical, or chain-broken audit fails closed.

## External mutation defense

The durable audit tracks the committed file size and, where the filesystem supplies it, the file key.

Before each append it verifies that the target still matches the last committed state.

If another writer appends/replaces/removes the file behind the instance, the next append fails rather than extending an untrusted history.

## Replay after restart

M7P already checks `successfulReceipt(requestDigest)` before approval, lease issuance, or handler execution.

Because M7Q reloads validated successful receipts on startup, a process restart can replay an already committed successful external result into a restored `WAITING_EXTERNAL` planner step without invoking the side effect again.

This is the key M7Q safety property.

## Bounds

Default limits are intentionally finite:

- audit file: 64 MiB;
- one record: 256 KiB;
- receipts: 100,000.

The constructor permits tighter host-selected bounds.

Capacity exhaustion fails explicitly; the implementation does not silently rotate or delete authority history.

## Android assembly

The existing `AndroidPlatformRuntime.createExternalExecutionFabric(...)` remains unchanged and keeps its in-memory default.

M7Q adds the explicit durable factory:

`createDurableExternalExecutionFabric(..., auditRoot, auditFileName)`

which wires `M6DurableActionAudit` into the same M7P execution fabric.

This avoids silently changing persistence behavior for existing callers while providing the production durable path.

## Verification actually run

The isolated M7Q gate was run with the final local sources and `kotlinc -Werror`.

The final jar was rerun and prints:

`M7Q_DURABLE_AUDIT_PASS`

Coverage includes:

- exact 654-byte durable record fixture;
- exact full-record SHA-256 parity;
- reopen/recovery of a committed receipt;
- successful receipt lookup after restart;
- M7P durable replay with zero second handler calls;
- two-record hash-chain recovery;
- torn final record rejection;
- tampered result rejection;
- non-canonical numeric representation rejection;
- configured record-count capacity rejection;
- external file mutation detection.

The repository platform host script wires the same test after M7P.

Full Android/Gradle execution is not claimed unless run separately.

## Next boundary

After M7Q, Android external side effects have:

- trusted intent binding;
- request-bound approval;
- deny-by-default policy;
- bounded leases;
- typed execution;
- durable hash-chained receipts;
- crash-safe replay suppression.

The next milestone should inspect concrete Android capability registration/assembly and connect app/device handlers to this fabric without weakening the existing Android permission broker.
