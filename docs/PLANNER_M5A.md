# VN97 M5A reasoning and planning controller

M5A defines the deterministic orchestration contract around VN97 reasoning. It does not replace
the recurrent language core; it controls how bounded reasoning, memory retrieval, verification,
interruption and future external actions are sequenced.

## Immutable plan definition

A plan contains:

- goal;
- ordered step graph;
- reasoning budget.

Each step has:

- kind;
- objective;
- dependencies;
- optional verification requirement;
- minimum confidence threshold.

Dependencies must reference earlier step IDs. This makes the graph a DAG with deterministic
topological order and avoids maintaining a second graph representation.

The plan ID is:

    SHA256(canonical_json(goal + immutable steps + budget))

created_ns is intentionally excluded from identity. Two identical plan definitions therefore
produce the same plan ID even when created at different times.

## Step kinds

M5A defines:

    REASON
    RETRIEVE
    VERIFY
    RESPOND
    EXTERNAL

EXTERNAL is a boundary marker only. M5A never executes side effects. Beginning an EXTERNAL step
moves it to WAITING_EXTERNAL and the controller accepts only a result supplied by a future
authority/tool layer. M6 owns the capability and approval boundary.

## Bounded reasoning

ReasoningBudget bounds:

- total state transitions;
- retries per step;
- memory queries;
- memory hits returned per query.

When the transition budget is exhausted, active work is cancelled into a checkpoint-safe state
and the plan becomes BUDGET_EXHAUSTED. The controller does not leave a half-running internal
step that could be mistaken for a durable result.

## Confidence and verification

A completed internal/external candidate carries confidence in [0,1].

A step enters WAITING_VERIFICATION when either:

- requires_verification is true; or
- confidence is below min_confidence.

Failed verification requeues the step while retry budget remains. Once retries are exhausted the
step and plan fail deterministically.

Evidence can reference stable VN97MEM1 record IDs. Evidence IDs are positive and unique.

## Sovereign-memory retrieval hook

PlanController.retrieve_context() consumes the M4 MemoryJournal retrieval contract directly.
The controller limits returned hits to max_memory_hits and limits query count to
max_memory_queries.

Returned context retains:

- memory record ID;
- content;
- source provenance;
- total score;
- semantic score;
- recency score;
- importance score.

M5A does not prescribe how a query embedding is produced. That remains a cognition/model
capability while the planner/memory boundary stays stable.

## Interruption and resume

pause() converts an in-flight internal RUNNING step back to PENDING because a partial internal
computation is not a durable result. WAITING_EXTERNAL and WAITING_VERIFICATION states can be
checkpointed as explicit durable boundaries.

resume() recomputes the plan lifecycle from the stored step states.

## VN97PLN1 checkpoint

Checkpoint header is 48 bytes:

    offset 0   magic[8] = "VN97PLN1"
    offset 8   version u32 = 1
    offset 12  payload_size u32
    offset 16  payload_sha256 u8[32]

The payload is canonical UTF-8 JSON:

- keys sorted;
- compact separators;
- NaN/Infinity prohibited;
- maximum payload size 16 MiB.

The payload contains plan identity, immutable plan definition, runtime step states, budget usage,
pause/terminal reason and evidence record IDs.

A checkpoint cannot contain a RUNNING internal step. The caller must first pause/interruption
transition to a safe point.

save_checkpoint() writes a same-directory temporary file, fsyncs it and atomically replaces the
destination.

SHA-256 provides checkpoint integrity/content identity. It is not an author signature or an
authority grant.

## State invariants

The loader rejects checkpoints when:

- plan ID does not match the immutable plan definition;
- dependencies are invalid;
- step IDs are not dense/ordered;
- budgets are exceeded;
- more than one internal/external active step exists;
- WAITING_EXTERNAL is used by a non-EXTERNAL step;
- completed/verification candidates lack result/confidence;
- evidence IDs are invalid/duplicated;
- lifecycle status contradicts core step state;
- canonical JSON bytes do not reproduce the stored payload exactly.

## M5 scope

M5A establishes the state-machine/checkpoint contract and M4 retrieval integration.

M5B will connect this controller to a cognition backend contract so VN97 can propose bounded
reasoning outputs, verification decisions and plan refinements while preserving the M5A state
machine. Tool/device execution remains excluded until M6 authority/tool fabric.
