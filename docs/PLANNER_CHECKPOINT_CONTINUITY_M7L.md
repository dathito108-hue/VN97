# M7L — VN97PLN1 Planner Checkpoint Continuity

M7L brings the canonical M5A planner checkpoint format to Android without changing the planner state machine or introducing a second persistence format.

Canonical planner continuity:

`NativePlanController → VN97PLN1 → atomic file replace + fsync → strict restore → NativePlanController`

Runtime continuity remains VN97RUN1/VN97RUN2.

## Byte-compatible format

Android uses the existing Python M5A binary format exactly:

- magic: `VN97PLN1`
- version: 1
- little-endian header
- payload size: uint32
- payload SHA-256: 32 bytes
- canonical UTF-8 JSON payload
- maximum payload: 16 MiB

The fixed header is 48 bytes.

The JSON payload preserves the canonical fields for:

- immutable plan ID, goal and reasoning budget;
- creation timestamp;
- plan status and counters;
- pause/terminal reasons;
- every step spec;
- attempts, status, candidate result/confidence;
- evidence record IDs;
- verification/failure notes.

## Cross-platform parity

The M7L fixture represents a two-step plan after:

1. REASON step begins;
2. result is completed;
3. verification passes;
4. EXTERNAL step enters WAITING_EXTERNAL.

Python and Kotlin encode the exact same 861-byte VN97PLN1 blob with SHA-256:

`eba8df81ddf32de6f7b2165bda26f8177a5de178ce57b040411ef7e7474c03b5`

The immutable plan ID inside the fixture is also produced by the M7K cross-platform plan-identity contract.

## Safe-point rule

A plan checkpoint is rejected while an internal step is `RUNNING`.

Hosts must first pause an in-flight internal step, which returns it to PENDING and discards any non-durable candidate, matching the Python M5A safe-point contract.

`WAITING_VERIFICATION` and `WAITING_EXTERNAL` are durable states and may be checkpointed.

## Strict restore

Restore validates before exposing a controller:

- magic/version/length;
- payload SHA-256;
- strict UTF-8;
- strict bounded JSON;
- exact object keys;
- integer-vs-float field contracts;
- enum codes;
- dense ordered step IDs;
- dependency ordering;
- transition/memory budgets;
- plan ID recomputation;
- active-step cardinality;
- candidate/result/confidence invariants;
- positive unique evidence IDs;
- canonical JSON byte equality.

Malformed or non-canonical checkpoints fail closed.

## Atomic persistence

M7L factors the existing runtime checkpoint file logic into one internal `AtomicBinaryStore`.

Both stores now share:

- trusted single-component filename validation;
- non-symlink root/target checks;
- bounded read/write;
- temporary file write;
- file fsync;
- atomic replace only;
- directory fsync;
- delete + directory fsync.

Existing `AtomicCheckpointStore` keeps its public API and runtime size bounds.

`AtomicPlannerCheckpointStore` adds:

- default filename `planner.vn97pln1`;
- 48-byte minimum;
- 16 MiB + header maximum;
- typed save/load through `NativePlannerCheckpoint`.

## Authority and architecture

M7L persists planner state only.

It does not:

- execute EXTERNAL steps;
- mint approval;
- alter M6 authority;
- activate capabilities;
- change M9 trust;
- change VN97RUN formats;
- add another model/backend.

The architecture remains:

`Code 1 → Code 2 hardware/mobile-aware upgrade → VN97 production`.

## Verification

Local isolated PASS gates:

- checkpoint codec compile with `kotlinc -Werror`;
- Python/Kotlin complete-byte VN97PLN1 parity;
- round-trip restore;
- bad magic rejection;
- bad digest rejection;
- corrupted payload rejection;
- atomic-store sources compile with `kotlinc -Werror`.

Repository host regression additionally exercises `AtomicPlannerCheckpointStore` using the real JNI directory-fsync path.

## Next continuity boundary

The next milestone should bind planner + runtime persistence as one generation/epoch so Android recovery cannot combine an old VN97PLN1 plan with a newer VN97RUN2 recurrent state.

That composite continuity layer should use commit-last metadata and verify exact model identity before resuming cognition.
