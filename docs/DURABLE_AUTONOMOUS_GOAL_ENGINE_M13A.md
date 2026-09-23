# M13A — Durable Autonomous Goal Engine

M13A turns the existing M7Z background continuation foundation into an app-facing
autonomous-work engine without introducing a second agent, planner, model,
memory store or authority path.

The canonical architecture remains:

`Code 1 -> Code 2 hardware/mobile-aware -> VN97 production`

The production autonomous path is:

`user goal
-> same VN97 cognition/planner
-> model-bound seed runtime
-> VN97CNT1 + VN97ACB1
-> VN97GOA1 orchestration ledger
-> persisted Android JobScheduler
-> cold-process VN97Application provider
-> same planner + sovereign memory + M6 authority
-> bounded wake
-> next VN97CNT1 epoch or terminal result`.

## Goal creation

The Android app exposes **Run autonomously** next to the normal Send action.

Creating an autonomous goal requires the normal trusted VN97 model to be active
and no foreground turn/approval to be in flight.

The same native VN97 cognition stack drafts the plan. M13A uses a larger but
bounded autonomous reasoning budget:

- up to 24 plan steps;
- up to 8 EXTERNAL steps;
- 128 planner transitions;
- 3 retries per step;
- 16 memory queries;
- 8 memory hits per query.

These are limits, not a promise that every goal uses the maximum.

## Model-bound seed

M7Z requires every persisted VN97CNT1 runtime snapshot to be bound to an exact
model ID. M13A adds `NativeRuntimeOwner.createModelBoundSeedSnapshot()`.

It:

1. creates a batch-1 runtime matching the activated VN97 geometry/backends;
2. activates it;
3. encodes the canonical BOS control token using the activated VN97TK1;
4. prefills exactly that one token through the activated VN97 model;
5. verifies the native runtime is bound to the exact activated model ID;
6. suspends the runtime;
7. returns the normal M7M runtime checkpoint snapshot.

No temporary model, alternate backend or synthetic model ID is used.

## Immutable continuation identity

The existing M7Z path remains authoritative:

- VN97CNT1 stores runtime + planner epoch;
- VN97ACB1 freezes principal + plan ID + model ID;
- JobScheduler extras are routing metadata only.

M13A adds job ID to the in-memory continuation context so the cold-process
provider can bind the wake to the correct VN97GOA1 record.

A background wake requires all of these to agree:

- JobScheduler job ID;
- VN97GOA1 job ID;
- VN97GOA1 plan ID;
- VN97ACB1 plan ID;
- restored VN97PLN1 plan ID;
- VN97GOA1 principal;
- VN97ACB1 principal;
- VN97GOA1 model ID;
- VN97ACB1 model ID;
- currently activated signed VN97 model ID;
- persisted goal text and restored planner goal text.

Any mismatch fails closed.

## VN97GOA1 autonomous ledger

`VN97GOA1` is an app-private orchestration record. It is not a planner
checkpoint.

It stores bounded metadata only:

- job ID;
- immutable plan ID;
- immutable model ID;
- immutable principal;
- immutable goal;
- state;
- created/updated timestamps;
- wake count;
- final response for completed goals;
- bounded terminal/status reason.

States:

- SCHEDULED
- RUNNING
- WAITING_APPROVAL
- PAUSED
- COMPLETED
- FAILED
- CANCELLED
- BUDGET_EXHAUSTED

The file has:

- magic `VN97GOA1`;
- version 1;
- bounded binary payload;
- SHA-256 payload integrity;
- strict UTF-8 decoding;
- symlink rejection;
- fsync before atomic replace;
- directory fsync;
- post-write read verification.

Identity fields and creation time are immutable across updates. Wake count and
update time cannot move backwards.

The app currently caps the ledger at 128 records and at most 32 non-terminal
active goals.

## Android JobScheduler

Autonomous job IDs use the positive high-bit namespace
`0x40000000..0x7fffffff`, deterministically seeded from the plan ID with
bounded collision probing.

Before scheduling, M13A:

1. persists VN97CNT1/VN97ACB1;
2. writes and verifies VN97GOA1;
3. schedules the existing persisted M7Z JobService.

If VN97GOA1 persistence fails, M13A removes the just-created continuation so no
orphan background job state remains.

If Android refuses initial scheduling, VN97GOA1 moves to PAUSED while durable
continuation state remains available for explicit recovery.

Persisted JobScheduler behavior still follows Android OS policy. M13A does not
claim an always-running process.

## Cold-process execution

`VN97Application` now implements the existing
`VN97AssistantContinuationWorkProvider`.

Every runnable wake:

1. loads VN97GOA1 by exact JobScheduler ID;
2. verifies all continuation identities;
3. increments the durable wake counter;
4. opens the currently activated signed VN97MI1;
5. requires exact model-ID continuity;
6. reconstructs the exact production M6 grants;
7. opens the same VN97MEM1-backed production assistant stack;
8. resumes the restored canonical planner;
9. advances only a bounded amount;
10. records the resulting VN97GOA1 state;
11. lets M7Z commit the next VN97CNT1 epoch.

The application has one fair reentrant **sovereign execution gate** shared by
foreground chat/voice/vision and background autonomous work. This prevents a
JobService wake from racing foreground VN97 memory/action work in the same
process.

## Power-aware bounded wakes

M13A maps the existing M7C compute mode to assistant cycles per wake:

- LOW_POWER: 2 cycles;
- BALANCED: 4 cycles;
- PERFORMANCE: 8 cycles.

Every wake permits at most one external handoff.

BLOCKED compute budgets never invoke autonomous work because the existing
JobService compute governor reschedules before calling the provider.

The global autonomous wake ceiling is 256. A goal that reaches that ceiling
without terminal completion fails closed instead of creating an unbounded
background loop.

## Retry behavior

Step-level retry remains the canonical planner behavior. M13A does not invent a
second retry engine.

Retryable cognition/memory/verification failures use the planner's existing
per-step retry budget. A non-terminal wake-level exception is recorded and asks
JobScheduler for normal backoff/rescheduling.

If the planner has already become terminal when an exception occurs, M13A
reconciles VN97GOA1 from the terminal planner before M7Z removes VN97CNT1.
This prevents a completed/failed planner from leaving a misleading RUNNING
ledger record.

## External authority

Background autonomy never bypasses M6.

If the planner reaches an EXTERNAL step:

`typed external intent
-> exact scope policy
-> M6 deny-by-default gate
-> explicit approval when required
-> one-use short lease
-> durable action receipt`.

If approval is required, the wake records WAITING_APPROVAL and M7Z intentionally
does not reschedule the WAITING_EXTERNAL plan.

M13A does not persist an approval token or automatically approve a task.

Foreground re-binding/approval and post-approval autonomous continuation are
the next M13B layer.

## User controls

The main app exposes:

- **Run autonomously** — persists and schedules the current text as a goal;
- **Cancel latest goal** — cancels the JobScheduler entry and removes its
  VN97CNT1/VN97ACB1 continuation before marking VN97GOA1 CANCELLED;
- a bounded status view for the five newest goals with state and wake count.

Normal Send remains the immediate foreground assistant path.

## Honest boundary

M13A establishes durable autonomous planning/execution across process death and
reboot, including bounded retries and power-aware wakes.

It intentionally stops at WAITING_APPROVAL. M13B must add the foreground
approval rebind/resume flow and plan-generation replan support after terminal
failure. M13A therefore does not claim fully unattended external side effects.
