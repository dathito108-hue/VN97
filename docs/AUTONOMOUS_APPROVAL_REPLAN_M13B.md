# M13B — Autonomous Approval Resume + Replanning Generations

M13B completes the two continuity gaps left intentionally by M13A:

1. a durable autonomous goal paused at an M6 approval boundary can be safely reopened in the foreground, explicitly approved/rejected, committed, and returned to background execution;
2. a failed/exhausted/stalled autonomous plan can create a bounded successor generation without mutating the immutable VN97ACB1 binding of the previous generation.

The canonical architecture remains one VN97 production path:

`Code 1 -> Code 2 hardware/mobile-aware -> VN97 production`

No second agent, planner, memory engine, Transformer/LLaMA backend, cloud runtime or approval path is introduced.

## Foreground approval reconstruction

M13A deliberately persisted only:

- VN97CNT1 runtime + planner continuity;
- VN97ACB1 principal/plan/model binding;
- VN97GOA1 orchestration metadata.

It did not persist an approval prompt/token.

M13B keeps that rule.

When a VN97GOA1 record is WAITING_APPROVAL, the app:

1. cancels any pending JobScheduler entry for that exact job ID;
2. acquires the process-wide sovereign execution gate;
3. releases idle foreground assistant/model/memory resources;
4. opens the exact currently activated signed VN97 model;
5. requires its model ID to equal VN97GOA1/VN97ACB1;
6. loads and verifies VN97CNT1 + VN97ACB1;
7. restores the same canonical planner/runtime;
8. opens the same production VN97MEM1 + M6 authority stack;
9. calls `resumeRestoredTurn(...)`;
10. lets M6 reconstruct a fresh approval prompt for the immutable WAITING_EXTERNAL step.

The reconstructed approval is held only in the live foreground session.

If the app/process dies before the user chooses Approve/Reject, no approval secret/token is durable. The previous VN97CNT1 epoch remains authoritative and the prompt can be reconstructed again later.

## Approval decision

Approve/Reject uses the existing `VN97AssistantSession.resolveApproval()` path.

Approve:

`reconstructed M6 prompt
-> local approval token
-> exact request/scope/principal validation
-> one-use authority lease
-> external execution
-> durable M6 receipt/audit
-> same canonical planner`

Reject:

`reconstructed M6 prompt
-> no token
-> failStep(retryable=false)
-> terminal canonical planner`

After resolution M13B commits exactly one new continuation state:

- terminal planner -> VN97CNT1/VN97ACB1 are removed;
- non-terminal planner -> same immutable plan/model/principal binding is persisted as the next VN97CNT1 epoch;
- another approval boundary -> VN97GOA1 remains WAITING_APPROVAL;
- yielded internal work -> VN97GOA1 becomes SCHEDULED and the same job is scheduled again;
- paused/terminal states -> no background polling.

The foreground assistant is reopened after the autonomous approval session closes so it observes the latest sovereign memory state.

Foreground chat/voice/vision are disabled while an autonomous approval session owns the production model/memory resources.

## VN97GOA1 v2 lineage

M13B evolves VN97GOA1 to version 2.

New immutable lineage fields:

- `rootJobId`
- `generation`
- `previousJobId`

Generation zero requires:

- `rootJobId == jobId`
- `previousJobId == 0`

Successor generations require a distinct positive predecessor job.

VN97GOA1 v1 remains readable and is interpreted as generation zero with its own job ID as root.

All M13A identity, integrity, atomic-write, fsync, strict UTF-8, symlink-rejection and monotonic timestamp/wake constraints remain.

## Why replanning uses a new job/binding

VN97ACB1 intentionally freezes:

- principal;
- plan ID;
- model ID.

M13B does not weaken that invariant.

A new plan therefore cannot replace the plan inside an existing VN97ACB1 generation.

Instead:

`generation N terminal
-> bounded failure evidence
-> same VN97 model proposes revised canonical plan
-> new plan ID
-> new model-bound seed runtime
-> new JobScheduler ID
-> new VN97CNT1/VN97ACB1
-> VN97GOA1 generation N+1`

The logical goal, root job ID, model ID and principal remain the same.

## Terminal-only replanning

`NativeCognitionLoop.replanTerminalPlan(...)` accepts only a terminal previous plan.

The new plan receives:

- the same user goal;
- the previous plan ID;
- previous step specifications;
- bounded failure/stall evidence;
- the same reasoning budget definition, with fresh counters.

The replacement must produce a different plan ID. If the model returns the same effective plan, replanning fails closed rather than looping unchanged.

## Replan triggers

Automatic replanning is allowed for:

- FAILED;
- BUDGET_EXHAUSTED;
- STALLED.

STALLED is first converted into a terminal cancelled planner with the internal reason that it was superseded by replanning, then a new generation is created.

Automatic replanning is not triggered for:

- user rejection;
- user cancellation;
- successful completion;
- approval waiting;
- explicit paused/external-blocked state;
- global wake-ceiling termination.

This prevents autonomy from routing around an explicit user denial.

## Bounded generations

The current production limit is three successor generations after generation zero.

Therefore one logical autonomous goal can have at most:

- generation 0;
- generation 1;
- generation 2;
- generation 3.

After that, failure/exhaustion remains terminal.

Each generation still has the M13A per-generation wake ceiling of 256 wakes and the existing planner retry/transition/memory budgets.

## Replan evidence

The feedback passed to the same native VN97 planning path is bounded to 8 KiB and includes:

- terminal status;
- terminal reason;
- step ID/kind/status;
- bounded objective;
- bounded result;
- bounded failure reason.

The instruction explicitly asks for a materially revised plan and not to repeat failed steps unchanged.

This feedback is planning context only. It is not a second planner or external intelligence source.

## Generation persistence ordering

For a successor generation M13B:

1. records the previous generation terminal;
2. creates the revised plan with the same activated VN97 model;
3. creates a model-bound seed runtime;
4. persists new VN97CNT1/VN97ACB1 under a new job ID;
5. writes/verifies the successor VN97GOA1 record;
6. schedules the successor JobScheduler job;
7. updates the predecessor terminal reason with successor job/generation linkage.

If successor ledger persistence fails, the new continuation is deleted.

If scheduling fails, the successor remains durable but PAUSED for explicit recovery.

The predecessor remains terminal in all cases.

## UI

The main Android activity now shows:

- autonomous job ID;
- state;
- generation;
- wake count;
- bounded result/reason;
- a dedicated autonomous approval panel when needed;
- Approve autonomous action;
- Reject autonomous action.

The existing foreground chat approval UI remains separate.

## Verification targets

M13B adds regression coverage for:

- VN97GOA1 generation lineage persistence;
- invalid successor lineage rejection;
- terminal-only canonical replanning;
- different successor plan identity;
- goal and reasoning-budget continuity across replans;
- rejection of replan attempts from non-terminal plans.

Full Android APK compilation remains the integration gate for the foreground continuation session, model ownership, activity wiring and JobScheduler interaction.

## Result

After M13B, a VN97 autonomous goal can:

`plan
-> work
-> checkpoint
-> reboot/process-death resume
-> request user authority
-> reconstruct approval safely
-> approve/reject
-> continue
-> fail/stall
-> bounded replan generation
-> continue again
-> complete or terminate`

All of this remains inside the same native VN97 model/planner/memory/authority architecture.
