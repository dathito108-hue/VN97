# M7W — Idempotent Turn Memory Write-Back

M7W adds the crash-safe commit contract that M7V intentionally deferred. It writes completed assistant turns into the existing native VN97MEM1 store without creating a second memory engine or alternate embedding backend.

Locked architecture remains:

`Code 1 → Code 2 hardware/mobile-aware upgrade → VN97 production`.

## Two-phase write-back

The sidecar `VN97TWJ1` journal records only transaction metadata:

- monotonically increasing sequence;
- `PREPARED` or `COMMITTED` phase;
- durable turn key;
- expected VN97MEM1 record ID;
- source/content SHA-256;
- previous-entry SHA-256 and current entry SHA-256.

It does **not** contain memory vectors or a second copy of the turn content.

Write ordering is:

1. derive a durable turn key from plan ID + planner creation time + principal;
2. fsync `PREPARED` with the expected next native record ID;
3. embed canonical turn content with the activated VN97 `NativeCognitionInferenceEngine`;
4. durable append one EPISODIC record to native VN97MEM1;
5. verify the exact expected native record by ID/source/content;
6. fsync `COMMITTED`.

Repeating the same completed turn returns the already-committed record ID without another native append.

## Crash recovery

If the process dies after `PREPARED` but before VN97MEM1 append, recovery sees the native last-record ID below the expected ID and performs the append once.

If the process dies after VN97MEM1 append but before `COMMITTED`, recovery reads the exact expected record through `MemoryStore::ViewRecord`/M7U JNI. A matching record is committed without a second append.

If that record exists but does not match the prepared turn, recovery fails closed. The writer never guesses that an unrelated record belongs to the transaction.

Only one unresolved PREPARED transaction is allowed. A different turn cannot bypass it.

## VN97TWJ1 integrity and bounds

`VN97TWJ1` is a fixed-frame sidecar:

- 16-byte header;
- 192-byte entries;
- fixed magic/version/entry size;
- contiguous sequence;
- SHA-256 entry chain;
- exclusive file lock;
- fsync per journal append;
- directory fsync on creation/torn-tail truncation;
- bounded default maximum of 200,000 entries.

Only an incomplete final entry can be recovered as a torn tail. A complete entry with a digest or chain mismatch fails closed.

Journal hashes provide integrity/idempotency metadata, not authority or signatures.

## Native record verification

M7W exposes `NativeMemoryStore.record(recordId)` as a typed Kotlin view over the already-existing M7U `nativeMemoryRecordInfo/nativeMemoryRecordRead` path.

No new native index is added and VN97MEM1 bytes/retrieval math are unchanged.

## Production assembly

`AndroidPlatformRuntime.createProductionTurnMemoryWriter(...)` requires:

- the exact activated `NativeActivatedModel`;
- the already-open native `NativeMemoryStore`;
- exact `dModel == vectorDim`.

It creates the embedder from `NativeCognitionInferenceEngine`, so Android production write-back cannot silently substitute a Transformer/LLaMA/third-party embedding backend.

M7W exposes an explicit `commitCompletedTurn(...)` transaction API. Automatic invocation from every completed M7T host turn remains a separate orchestration step so answer delivery semantics are not coupled to storage failure.

## Verification

The isolated M7W gate was run with `set -euo pipefail` and `kotlinc -Werror` and printed:

`M7W_IDEMPOTENT_TURN_MEMORY_WRITEBACK_PASS`

Coverage includes:

- repeated completed-turn commit without duplicate append;
- durable PREPARED recovery after append failure;
- mismatch at the expected native record ID fails closed;
- torn final journal entry recovery;
- complete-entry hash corruption rejection.

The repository host gates are extended to compile M7W against the canonical planner/session types and to verify the new direct native record view in the real M7U JNI bridge.
