# M7M — Composite Continuity Epochs

M7M binds the already-canonical runtime and planner checkpoints into one crash-consistent Android continuity generation.

It does not replace or modify either existing checkpoint format:

- recurrent/model-bound runtime state remains VN97RUN1 / VN97RUN2;
- planner state remains VN97PLN1.

M7M adds only a commit record:

`VN97CNT1`

The architecture remains:

`Code 1 → Code 2 hardware/mobile-aware upgrade → VN97 production`.

## Problem solved

Persisting VN97RUN2 and VN97PLN1 independently can produce an invalid recovery pair after a crash:

- newer recurrent state + older planner state; or
- newer planner state + older recurrent state.

M7M makes those two files one committed generation.

## Bounded A/B slots

Storage uses two reusable generations rather than an unbounded epoch directory:

- `runtime-a.vn97run`
- `planner-a.vn97pln1`
- `runtime-b.vn97run`
- `planner-b.vn97pln1`
- `continuity.vn97cnt1`

A save chooses the inactive slot and performs:

1. write + fsync inactive runtime checkpoint;
2. atomic replace + directory fsync;
3. write + fsync inactive planner checkpoint;
4. atomic replace + directory fsync;
5. encode the new VN97CNT1 manifest;
6. atomic replace + directory fsync the manifest **last**.

The manifest is therefore the commit point.

A crash before step 6 leaves the previous manifest authoritative. Partially written or fully written inactive-slot data is ignored.

The next save alternates A/B, bounding persistent checkpoint data to two generations.

## VN97CNT1

The manifest uses the same integrity style as existing sovereign binary formats:

- magic: `VN97CNT1`;
- version: 1;
- 48-byte little-endian header;
- canonical UTF-8 JSON payload;
- SHA-256 over the exact payload;
- bounded payload size.

Canonical payload fields are:

- `epoch`;
- `slot` (`a` or `b`);
- `runtime_sha256`;
- `planner_sha256`;
- `model_id`;
- `plan_id`;
- `sequence_position`.

The model ID is the exact 32-byte model identity already bound by M7D/M7E and stored as lowercase hex.

## Save invariants

Composite persistence requires:

- runtime lifecycle is SUSPENDED;
- runtime is already model-bound;
- model ID is exactly 32 bytes;
- planner is at a VN97PLN1-safe point;
- sequence position is non-negative.

`NativeRuntimeOwner.suspendAndSnapshot()` captures runtime checkpoint, runtime info, and model binding from the same owned session after suspension.

The planner is independently encoded through the existing VN97PLN1 codec.

## Restore invariants

Loading a committed generation verifies:

1. VN97CNT1 header, payload SHA-256 and canonical JSON;
2. active runtime slot exists and its SHA-256 equals the manifest;
3. active planner slot exists and its SHA-256 equals the manifest;
4. VN97PLN1 decodes and its immutable plan ID equals the manifest;
5. VN97RUN restores successfully;
6. runtime config equals the caller's requested config;
7. restored lifecycle is SUSPENDED;
8. restored sequence position equals the manifest;
9. the **actual restored runtime checkpoint** is model-bound;
10. its exact 32-byte model ID equals the manifest.

The manifest cannot forge model binding. A manifest that claims a model ID around an unbound runtime checkpoint is rejected during runtime restore.

## Crash/corruption behavior

The regression exercises two committed epochs:

- epoch 1 commits slot A;
- epoch 2 commits slot B.

Corrupting inactive slot A does not affect epoch 2 recovery because the committed manifest points to B.

Corrupting active slot B causes fail-closed load because its digest no longer matches the committed manifest.

There is no heuristic fallback from a corrupt committed generation to an older slot. Such fallback could silently rewind cognition/runtime state and is intentionally excluded.

## Compatibility

Existing standalone `AtomicCheckpointStore` remains supported.

M7M adds:

- `NativeRuntimeCheckpointSnapshot`;
- `NativeContinuityManifest`;
- `NativeCompositeContinuity`;
- `AtomicCompositeContinuityStore`;
- `NativeRuntimeOwner.suspendAndSnapshot()`;
- `NativeRuntimeOwner.restoreComposite(...)`.

No VN97RUN or VN97PLN1 bytes are changed.

## Authority

M7M is continuity only.

It does not:

- execute an EXTERNAL step;
- mint or restore approval outside the existing M6 contracts;
- activate a capability;
- weaken M9 artifact trust;
- infer a model identity from manifest metadata.

External effects still require the canonical M6 authority path after recovery.

## Verification

Isolated M7M A/B-slot gate PASS:

`M7M_COMPOSITE_STORE_PASS`

Coverage includes:

- epoch/slot alternation;
- commit-last manifest semantics;
- inactive-slot corruption tolerance;
- active-slot corruption rejection;
- bounded deletion.

Repository host regression additionally uses a real native VN97 runtime checkpoint and proves that a manifest model ID cannot turn an unbound runtime checkpoint into a bound one.

The host regression prints:

`M7M_COMPOSITE_CONTINUITY_PASS`

when the full JNI-backed gate succeeds.

## Next boundary

After M7M, Android has native inference, typed cognition, canonical planner state, and crash-consistent paired continuity.

The next milestone should connect those pieces through the existing M5 cognition lifecycle:

`M7J typed cognition → M7K planner → memory/evidence → M6 external handoff → M7M safe persistence → final response`

That coordinator must port the canonical M5B behavior rather than create another reasoning architecture.
