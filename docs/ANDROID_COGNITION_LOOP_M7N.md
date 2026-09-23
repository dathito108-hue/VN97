# M7N — Android Cognition Loop Coordinator

M7N ports the canonical M5B `CognitionLoop` orchestration semantics onto the Android runtime surface.

It connects already-completed native pieces:

`M7J typed cognition → M7K planner → bounded memory retrieval → verification/retry → final RESPOND`

without introducing another model, planner, memory database, or authority mechanism.

The architecture remains:

`Code 1 → Code 2 hardware/mobile-aware upgrade → VN97 production`.

## Scope

M7N adds `NativeCognitionLoop`, which mirrors the Python M5B coordinator:

- build a typed plan through M7J;
- validate plan-size/external-step/objective limits;
- optionally refine only an unstarted READY plan;
- execute first-ready steps through M7K;
- collect succeeded dependency results;
- bound dependency context by UTF-8 bytes;
- issue typed memory queries for RETRIEVE steps;
- bound memory context by UTF-8 bytes;
- inherit evidence record IDs deterministically;
- generate typed step proposals;
- enter verification when required by the step or confidence threshold;
- retry/fail through the canonical planner budget;
- stop at a bounded lifecycle boundary;
- surface the latest succeeded RESPOND result as final response.

## Boundaries

The Android boundary enum mirrors M5B:

- COMPLETED
- WAITING_EXTERNAL
- PAUSED
- FAILED
- CANCELLED
- BUDGET_EXHAUSTED
- YIELDED
- STALLED

A run has a bounded cycle limit. Reaching that limit without another lifecycle boundary returns YIELDED rather than spinning indefinitely.

Transition-budget exhaustion raised by the planner is translated into the BUDGET_EXHAUSTED loop boundary, matching the Python coordinator.

## Backend failure semantics

M7J is a strongly typed backend, but model parsing/generation can still throw.

M7N preserves the M5B distinction:

- failures while invoking cognition generation are backend failures and follow the retry path;
- loop-owned validation after a typed object is returned is fail-closed.

Examples of loop-owned validation include:

- plan must end in RESPOND when configured;
- maximum plan/external-step counts;
- objective/result/note UTF-8 limits;
- memory retriever must not return more than bounded topK.

## Memory boundary

Android does not create a second sovereign-memory implementation in M7N.

Instead, `NativeMemoryRetriever` is a narrow host-owned interface:

- exposes the authoritative vector dimension;
- receives the M7J typed `NativeMemoryQuery`;
- receives planner-bounded topK;
- returns typed `NativeMemoryContextItem` values with provenance scores.

The coordinator consumes the planner memory-query budget **before** retrieval, matching M5B.

Context is then bounded before it is sent back into cognition.

A later Android M4 memory port can implement this interface without changing cognition-loop semantics.

## Evidence provenance

Evidence is inherited in stable order:

1. dependency evidence record IDs;
2. retrieved memory record IDs.

Duplicates are removed without reordering.

The resulting evidence list is committed into M7K when the step result becomes durable and is then available to downstream dependencies and verification.

## External actions

M7N deliberately does not call `proposeExternalIntent` or execute tools inside the loop.

When an EXTERNAL step begins:

- M7K moves it to WAITING_EXTERNAL;
- M7N returns WAITING_EXTERNAL;
- no model output is interpreted as execution authority.

The external continuation remains:

`WAITING_EXTERNAL → trusted capability catalog → typed external intent → M6 authority/approval → effect → recordExternalResult(...)`

This preserves M6 deny-by-default semantics.

## Continuity

M7N does not add another persistence format.

At safe boundaries the host can persist:

- model-bound VN97RUN through M7M;
- the same plan as VN97PLN1 in the same M7M composite epoch.

Therefore a yielded, paused, waiting-external, or completed cognition state can use the existing composite continuity layer.

## Verification

The repository host regression covers the coordinator path with scripted native-cognition outputs:

- PLAN → REASON;
- MEMORY_QUERY → RETRIEVE with typed memory provenance;
- RESPOND → COMPLETED final answer;
- inherited evidence propagation;
- verification wait + verify + continuation;
- YIELDED at an explicit cycle limit;
- EXTERNAL → WAITING_EXTERNAL without execution;
- rejection of a plan that violates the final RESPOND requirement.

The host test prints:

`M7N_COGNITION_LOOP_PASS`

on success.

Full repository host execution is a separate gate and must not be inferred from source/test wiring alone.

## Next boundary

After M7N, the remaining Android end-to-end orchestration gap is external handoff and memory implementation, not another reasoning architecture.

The next milestone should bind an EXTERNAL waiting step to the existing M6 capability/authority contracts and keep all effectful execution outside model/cognition code.
