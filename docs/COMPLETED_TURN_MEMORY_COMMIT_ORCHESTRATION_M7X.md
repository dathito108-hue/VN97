# M7X — Completed-Turn Memory Commit Orchestration

M7X connects the existing M7W crash-safe VN97 turn-memory write-back contract to the canonical M7T/M7V production assistant lifecycle.

## Locked architecture

The path remains one VN97 production path:

`NativeActivatedModel -> VN97COG1 -> canonical planner -> M7V VN97MEM1 retrieval -> M6 authority/execution -> final response -> M7W VN97MEM1 write-back`

M7X does not add a model, inference backend, planner, memory engine, authority gate, approval policy, or side-effect path.

## Completion semantics

When the canonical cognition boundary becomes `COMPLETED`, `VN97AssistantSession` obtains the final response from the completed canonical RESPOND step and attempts M7W write-back with the same completed turn.

A successful write-back returns:

- `VN97AssistantMemoryCommitState.COMMITTED`;
- the native VN97MEM1 record ID.

A normal persistence exception returns:

- the assistant turn as `COMPLETED`;
- the exact final response unchanged;
- `VN97AssistantMemoryCommitState.RETRY_REQUIRED`;
- bounded failure metadata for the host.

Memory persistence failure therefore does not retroactively turn successful cognition or an already completed external action into an assistant failure.

## Retry

`retryCompletedTurnMemoryCommit(update, timestampNs)` retries only an update that is already `COMPLETED` and marked `RETRY_REQUIRED`.

The retry goes through the same M7W `VN97TurnMemoryWriter`. M7W's PREPARED/COMMITTED journal and exact expected-record verification remain authoritative, so a retry after an append/journal interruption is idempotent.

A previously `COMMITTED` update is returned unchanged and does not append again.

## Android production ownership

`createProductionMemoryBackedAssistant(...)` now assembles one resource bundle containing:

- the existing M7T/M7V assistant session;
- the canonical native VN97MEM1 store;
- the M7W `VN97TurnMemoryWriter` bound to that same store and activated VN97 model.

The resource closes the writer first and the native memory store second.

The non-memory `createProductionAssistantSession(...)` path remains available and does not attempt write-back.

## Recovery boundary

M7X provides in-process typed retry and preserves M7W's durable pending transaction. Reconstructing a completed turn after cold-process death in order to reconcile a pending M7W transaction is intentionally left to the next continuity milestone; M7X does not invent a second durable turn store to bypass VN97MEM1/VN97TWJ1.
