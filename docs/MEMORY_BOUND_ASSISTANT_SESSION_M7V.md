# M7V — Memory-Bound Production Assistant Session

M7V binds the canonical M7U native VN97MEM1 store to the existing M7T production assistant session. It is a direct production assembly upgrade, not a second memory engine, planner, session architecture or model backend.

Locked architecture remains:

`Code 1 → Code 2 hardware/mobile-aware upgrade → VN97 production`.

The retrieval path is now assembled as:

`NativeActivatedModel → VN97COG1 → M7N cognition → M7T assistant session → bound M7U NativeMemoryStore → native M4 VN97MEM1 index/retrieval`.

## One sovereign memory source per turn

`VN97AssistantSession` accepts an optional trusted `defaultMemory`.

When a memory source is bound:

- callers may omit the per-call memory parameter;
- start, yield/continue, approval resolution and restored-plan resumption all use the same exact retriever instance;
- a caller cannot replace that retriever with a different memory source during the turn;
- attempting to override a session-bound memory fails before plan construction.

When no default memory is configured, a caller may still supply an explicit retriever when the turn starts, but that exact retriever is frozen for the remainder of the active turn.

This prevents one plan from retrieving evidence from different stores across host boundaries.

## Android production assembly

`AndroidPlatformRuntime.createProductionMemoryBackedAssistant(...)`:

1. opens or creates the existing M7U app-private VN97MEM1 store;
2. verifies its vector dimension against the activated model `dModel`;
3. constructs the existing one-model M7T assistant path;
4. binds that native store as the session default memory;
5. returns `VN97ProductionAssistantResources`, which exposes the session and store and closes the native store explicitly.

The existing `createProductionAssistantSession(...)` remains available for callers that intentionally do not use persistent retrieval.

No generic memory-engine plugin point is added at the Android production assembly boundary.

## Authority and evidence invariants

M7V does not change M6 policy, approval, leases or external-action execution.

Memory record IDs remain native VN97MEM1 IDs. M7N still attaches evidence only from the retriever result actually returned by the bound native store.

Changing memory source is not an approval mechanism and does not grant external side-effect authority.

## Write-back boundary

M7V binds durable retrieval only. It does not automatically append user/assistant turns to VN97MEM1.

Automatic turn write-back needs its own idempotent commit contract so a process crash cannot duplicate a completed turn in sovereign memory. That is intentionally deferred rather than adding a best-effort duplicate-prone write path.

## Verification

The isolated M7V gate was run locally with `set -euo pipefail` and `kotlinc -Werror` and printed:

`M7V_MEMORY_BOUND_ASSISTANT_SESSION_PASS`

It covers:

- session-default memory use;
- memory binding across yield/continue;
- fail-closed source replacement during an active turn;
- fail-closed override before plan construction;
- approval continuation retaining the same memory binding;
- restored-plan resumption using the bound memory.

The repository host gate is extended with a real M7N RETRIEVE step and verifies that a memory-bound M7T session reaches the retriever without the caller passing memory on each host call.

M7V changes no VN97MEM1 bytes or native retrieval math.
