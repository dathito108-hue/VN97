# M7Z — Android Reboot / Background Assistant Continuity

M7Z connects the existing M7C persisted Android JobScheduler path to the canonical M7M
runtime+planner continuity and the M7T–M7Y production assistant lifecycle.

The architecture remains one direct VN97 production path:

`NativeActivatedModel -> VN97COG1 -> canonical planner -> VN97CNT1 -> persisted Android JobService -> restored planner -> M6 authority/execution -> VN97MEM1`

No Transformer/LLaMA backend, second planner, second memory engine, or second authority path is
introduced.

## Durable state ownership

M7Z does not create another runtime or planner checkpoint format.

M7M remains authoritative for cognition/runtime continuity:

- VN97RUN1/VN97RUN2 stores recurrent runtime state;
- VN97PLN1 stores the canonical planner;
- VN97CNT1 is the commit point binding one runtime+planner epoch, model ID, plan ID and sequence
  position.

M7Z adds only the small app-private `VN97ACB1` assistant-continuation binding. It freezes the
identity needed to safely resume M6 policy evaluation:

- principal;
- plan ID;
- activated model ID.

VN97ACB1 is immutable for the lifetime of a scheduled assistant continuation. It contains no model
weights, planner state, memory records, approval token, capability lease, or action result.

## VN97ACB1 integrity

The binding uses:

- magic `VN97ACB1`;
- version 1;
- a bounded 48-byte header;
- bounded UTF-8 principal;
- exact 32-byte plan ID;
- exact 32-byte model ID;
- SHA-256 over the payload;
- fsync before atomic create;
- directory fsync;
- symlink rejection.

JobScheduler extras carry the same identity only as routing metadata. Cold-process restoration
requires the extras to match the durable VN97ACB1 file exactly. The JobService never treats extras
alone as authority.

## Persist and schedule

`AndroidContinuationScheduler.persistAssistant(...)` persists one safe non-terminal assistant
epoch through the existing `AtomicCompositeContinuityStore`.

It requires:

- a non-terminal canonical plan;
- a model-bound runtime snapshot;
- a stable principal;
- exact agreement with any previously durable VN97ACB1 binding.

Only after that durable state exists should the host call `scheduleAssistant(...)`.

The scheduled JobInfo is persisted, uses the existing `VN97ContinuationJobService`, and therefore
survives process death and reboot subject to Android OS scheduling rules. The platform manifest
already declares `RECEIVE_BOOT_COMPLETED`; the JobService remains non-exported and protected by
`BIND_JOB_SERVICE`.

This is not an always-running foreground service. Android may defer, stop, or kill background work.

## Cold-process assistant restore

Assistant-mode JobService execution performs this sequence:

1. sample the M7C Android compute governor;
2. decode the persisted runtime configuration;
3. reconstruct the expected principal/plan/model binding from JobScheduler extras;
4. require exact agreement with app-private VN97ACB1;
5. load and verify the committed VN97CNT1 generation;
6. require VN97CNT1 plan ID and model ID to match VN97ACB1;
7. reject terminal planners as background continuation input;
8. restore the exact model-bound runtime through `NativeRuntimeOwner.restoreComposite(...)`;
9. require the restored lifecycle to be SUSPENDED, then explicitly resume;
10. invoke the host `VN97AssistantContinuationWorkProvider` with that same planner controller;
11. if the job was stopped, do not commit a newer epoch, leaving the previous VN97CNT1 generation
    authoritative;
12. otherwise suspend the runtime and commit the mutated planner/runtime as the next M7M epoch.

A callback failure also leaves the previous committed epoch authoritative and requests JobScheduler
rescheduling.

## M6 authority boundary

M7Z never persists or restores an approval prompt/token as execution authority.

A restored planner that reaches `WAITING_EXTERNAL` must still traverse the existing chain:

`typed external intent -> M6 approval/policy -> deny-by-default execution -> durable audit`

The background JobService persists WAITING_EXTERNAL but does not automatically reschedule that
state. This prevents an approval-waiting turn from becoming a background polling/execution loop.

PAUSED turns are also persisted without automatic rescheduling.

## Production assistant resume

`AndroidPlatformRuntime.resumeProductionAssistantContinuation(...)` is the canonical production
bridge from the restored JobService context back into the existing assistant stack.

Before cognition continues it:

1. rejects a stopped continuation;
2. requires the supplied `NativeActivatedModel` to match the exact VN97ACB1 model ID;
3. opens the canonical M7V/M7Y memory-backed production assistant;
4. performs any pending M7Y VN97TMR1/VN97TWJ1/VN97MEM1 recovery;
5. resumes the exact restored planner with the exact durable M6 principal.

The same controller object is mutated by the assistant and then committed by the JobService in the
next VN97CNT1 epoch.

## Terminal completion

When the callback drives the planner terminal, M7Z does not schedule another background wake.
The runtime is safely suspended, then the committed VN97CNT1 generations and VN97ACB1 binding are
removed.

Completed-turn memory durability remains owned by M7X/M7Y. If a terminal response has a pending
memory write-back, M7Y's app-private recovery envelope remains independent of the deleted
non-terminal continuation state and is reconciled the next time the production memory-backed
assistant is opened.

## Verification

Local Kotlin gates executed during M7Z development:

- `M7Z_CONTRACT_SYNTAX_PASS` — pure continuation-contract syntax with API-compatible stubs;
- `M7Z_ANDROID_CONTINUATION_SYNTAX_PASS` — AndroidContinuation syntax with Android/runtime stubs;
- `M7Z_ASSISTANT_CONTINUATION_CONTRACT_PASS` — plan/model/terminal binding behavior;
- `M7Z_DURABLE_BINDING_FAIL_CLOSED_PASS` — immutable durable binding and corruption rejection.

The repository host regression also includes
`M7ZAssistantContinuationContractTest.kt` and prints
`M7Z_ASSISTANT_CONTINUATION_CONTRACT_PASS` when that gate executes successfully.

These are not a claim of full Android device instrumentation. Device/reboot instrumentation remains
a later production hardening gate.
