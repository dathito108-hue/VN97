# M7C — Android Continuity + Compute Governance

M7C provides the Android lifecycle contract for continuing VN97 work after process death and
across reboot, subject to Android OS scheduling limits.

## Persisted scheduling

`AndroidContinuationScheduler` submits a persisted `JobInfo` targeting the library
`VN97ContinuationJobService`. Runtime shape/backend configuration is stored in a
`PersistableBundle`; checkpoint bytes remain in the internal VN97RUN1 checkpoint store.

The platform manifest declares `RECEIVE_BOOT_COMPLETED`, which Android requires for persisted
jobs. The JobService is non-exported and protected by `BIND_JOB_SERVICE`.

This is not an always-running foreground service. Android remains free to defer, stop or kill
background work.

## Cold-process reconstruction

When Android starts the JobService in a fresh process:

1. the host Application must implement `ContinuationWorkProvider`;
2. the governor samples current battery/charging/thermal state;
3. blocked budgets return to JobScheduler for rescheduling;
4. otherwise the service rebuilds `NativeRuntimeConfig` from persisted job extras;
5. `NativeRuntimeOwner` restores VN97RUN1 or creates a fresh session;
6. CREATED sessions activate; restored SUSPENDED sessions explicitly resume;
7. the host continuation callback runs with a `ContinuationContext`;
8. the runtime is suspended and atomically checkpointed before job completion.

A restored session is never auto-running before the service explicitly resumes it.

## Compute policy

The pure `VN97ComputePolicy` is independent from Android APIs and maps `ComputeSignals` to:

- BLOCKED — severe thermal state or <=10% battery while unplugged;
- LOW_POWER — moderate thermal state or <=20% battery while unplugged;
- BALANCED — normal unplugged operation;
- PERFORMANCE — charging with light/no thermal pressure.

Each `ComputeBudget` includes max tokens per wake, max run milliseconds and retry-delay hint.
Callbacks must treat these values and `ContinuationContext.isStopped()` as hard cooperative
ceilings.

## Cancellation and JobScheduler semantics

`onStopJob()` marks the active continuation context stopped and returns true, requesting that
Android reschedule the job. The library does not report interrupted work as completed.

Unexpected continuation failures also request rescheduling.

## Authority boundary

M7C has no direct capability handler and cannot mint approval, policy or lease state.
`RECEIVE_BOOT_COMPLETED` supports only OS scheduling continuity.

Any external action arising from continued cognition must still traverse:

`WAITING_EXTERNAL → M6C intent/approval → M6A policy/lease/audit → M6B handler`

## Host regression

The platform host test now compiles the pure compute policy with `-Werror` and verifies:

- charging/cool → PERFORMANCE;
- normal unplugged → BALANCED;
- low battery → LOW_POWER;
- moderate thermal → LOW_POWER;
- severe thermal → BLOCKED;
- critical unplugged battery → BLOCKED;
- invalid battery inputs fail closed.

Android JobScheduler/PowerManager integration requires Android build/device instrumentation, while
the policy itself remains deterministic on ordinary CI hosts.
