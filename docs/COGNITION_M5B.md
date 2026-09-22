# VN97 M5B cognition/reasoning loop

M5B connects the M5A deterministic plan state machine to a typed cognition backend contract.
The backend may propose reasoning content, memory queries and verification decisions, but it
cannot directly mutate planner lifecycle state or execute an external side effect.

## Backend contract

CognitionBackend exposes four typed operations:

- propose_plan(PlanDraftRequest) -> PlanDraft
- memory_query(MemoryQueryRequest) -> MemoryQuery
- propose_step(StepReasoningRequest) -> StepProposal
- verify_step(VerificationRequest) -> VerificationDecision

The loop validates response types and size/shape constraints before a response is committed to
PlanController.

Backend output never contains authority grants or trusted memory evidence IDs.

## Plan proposal

PlanDraft contains immutable PlanStepSpec entries. CognitionLoop validates:

- at least one step;
- max_plan_steps;
- max_external_steps;
- per-objective UTF-8 byte limit;
- dependency structure through M5A PlanController;
- by default the final step must be RESPOND.

Plan identity remains the M5A SHA-256 identity over goal + immutable graph + ReasoningBudget.

## Safe refinement

refine_unstarted_plan() can replace only a completely unstarted READY plan:

- no transitions used;
- no memory queries used;
- every step still PENDING.

Refinement never mutates the existing graph in place. It creates a replacement plan with a new
plan ID and returns the previous plan ID as lineage metadata.

Once execution starts, the immutable M5A graph remains fixed.

## Step proposal loop

For an internal step the loop:

1. captures previous retry/verification failure feedback;
2. starts the M5A step, consuming the normal transition budget;
3. gathers only explicitly declared dependency results;
4. optionally performs bounded VN97MEM1 retrieval for RETRIEVE;
5. calls propose_step();
6. validates result type, UTF-8 byte limit and confidence;
7. attaches evidence IDs itself;
8. commits through PlanController.complete_step().

The backend cannot invent evidence record IDs. Evidence is the deterministic ordered union of:

- evidence already carried by succeeded dependency steps;
- VN97MEM1 record IDs actually returned by the current retrieval.

## Memory-query boundary

For a RETRIEVE step, memory_query() returns:

- exact query vector;
- top_k;
- optional episodic/semantic kind filter;
- semantic/recency/importance weights;
- optional now_ns;
- recency half-life.

The vector length must exactly equal the opened VN97MEM1 journal vector dimension. M5A still
enforces max_memory_queries and max_memory_hits.

Memory source/content passed to cognition is additionally bounded by an M5B UTF-8 context budget.
Truncation affects only backend context text, never the stable evidence record IDs.

## Dependency-context boundary

Dependency results are included only for the current step's declared dependencies. Objective and
result text are copied into a bounded UTF-8 context budget.

StepReasoningRequest.context_truncated explicitly tells a backend when dependency or memory text
was truncated.

## Verification / reflection

When M5A places a candidate in WAITING_VERIFICATION, M5B sends a VerificationRequest containing:

- candidate result and confidence;
- step objective/kind/attempt;
- declared dependency context;
- trusted evidence record IDs attached by the controller.

A normal negative VerificationDecision consumes the existing M5A retry policy.

A verifier exception is treated as a retryable verification failure. A verifier response with
the wrong schema is a contract violation and fails the step non-retryably.

## Backend failure semantics

A proposal/memory backend exception is converted into a retryable step failure so the existing
M5A retry and transition budgets remain authoritative.

A schema/shape/size contract violation is non-retryable and fails closed.

The loop never leaves a RUNNING step after one loop cycle returns.

## External boundary

Beginning an EXTERNAL step still transitions immediately to WAITING_EXTERNAL.

M5B does not call propose_step() for that step and does not execute any tool, file, network,
device or app action. M6 remains the only authority/capability boundary.

## Run boundaries

run_until_boundary() stops on:

- COMPLETED;
- WAITING_EXTERNAL;
- PAUSED;
- FAILED;
- CANCELLED;
- BUDGET_EXHAUSTED;
- explicit per-run YIELDED safe point;
- STALLED if no legal next transition exists.

The per-run cycle limit is an execution-yield control. Durable global bounds remain the M5A
ReasoningBudget persisted in VN97PLN1.

## Current capability boundary

M5B defines orchestration and backend schemas. It does not claim the untrained reference
VN97LanguageCore can already produce high-quality plans or verification decisions.

A production VN97 cognition adapter must implement CognitionBackend while preserving this
contract. That adapter can later use the native/tokenizer/model runtime without changing plan,
memory or authority semantics.
