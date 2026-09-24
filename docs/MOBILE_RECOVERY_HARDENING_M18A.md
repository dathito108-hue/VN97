# M18A — Mobile Recovery Hardening

M18A hardens VN97 process death, reboot, package replacement, and production
execution entrypoints without adding a second runtime or model path.

The canonical architecture remains:

```text
Code 1
  -> Code 2 hardware/mobile-aware
  -> signed VN97MI1 production model
  -> canonical planner / VN97MEM1 / M6 authority
  -> Android production surfaces
```

M18A adds only a recovery barrier around existing durable subsystems.

## Recovery domains

The barrier has three bounded domains:

1. **ACTIVATION**
   - reconciles the existing transactional capability activation state;
   - reconciles pending M17C promotion attempts;
   - preserves the exact active-model inventory as the authority.

2. **AUTONOMOUS**
   - uses the existing M13 persisted autonomous goal store and JobScheduler
     bindings;
   - no second scheduler or planner is created.

3. **PAPER_TRADING**
   - uses the existing M15 durable paper session store and persisted jobs;
   - no live trading authority is introduced.

Activation recovery is always first because autonomous/game/trading execution
must not proceed against an unresolved model inventory.

## Trigger policy

M18A deliberately distinguishes process creation from an actual system restart.

### PROCESS_START

A normal Android process start performs only activation/promotion recovery.

It does **not** reschedule every autonomous or paper job. Persisted Android
JobScheduler state already owns those schedules, so replaying full reboot
reconciliation on every process creation would unnecessarily mutate schedule
counters and could create duplicate work.

### EXECUTION_ENTRY

Before a production surface opens the activated model, it passes through the
same activation recovery barrier.

M18A installs this barrier on:

- the main foreground assistant attach path;
- restored autonomous assistant continuation work;
- floating assistant initialization;
- governed game-agent episodes;
- scheduled paper-trading episodes.

This closes the race where a component started in a fresh process could open
VN97INV1 while an interrupted activation or M17C promotion still needed
reconciliation.

### SYSTEM_RESTART / PACKAGE_REPLACED

The existing boot/package receiver now routes through the unified recovery
coordinator.

After activation recovery succeeds it independently reconciles:

- M13 autonomous jobs;
- M15 paper-trading jobs.

If activation recovery fails, both dependent domains are skipped fail-closed.
If autonomous recovery fails, paper recovery still runs, and vice versa.

## Failure containment

Expected durable-state failures are represented as bounded recovery results
instead of escaping through `Application.onCreate()` and causing a process
crash loop.

Each domain reports one of:

- `SUCCEEDED`;
- `FAILED`;
- `SKIPPED`.

Failure text is newline-normalized and bounded to 1024 UTF-8 bytes with a
linear-time code-point truncation path.

Fatal VM/linkage failures are not converted into ordinary recovery results.

The latest process-local recovery report is retained by the coordinator for
diagnostics.

## Sovereign serialization

All coordinated recovery executes under the existing
`VN97Application.withSovereignExecution` lock.

M18A therefore does not introduce another authority lane. Recovery cannot race
a sovereign foreground/background action inside the same process.

The recovery coordinator itself also has a fair lock so cold-start recovery and
an immediate execution-entry barrier serialize deterministically.

## Paper job retry

A scheduled paper episode that cannot cross the activation recovery barrier does
not execute against ambiguous state.

The JobService asks the existing paper-session policy whether the job remains
retryable and reports that decision back to JobScheduler. No new retry policy is
invented.

## Floating assistant

The floating 3D service may still restore its overlay at boot when the user has
enabled it, but its interaction controller cannot open the VN97 model until the
activation barrier succeeds.

This preserves overlay continuity without allowing stale model execution.

## Authority boundary

M18A does not:

- activate an unreviewed model;
- approve an M17 candidate;
- bypass M17C human promotion confirmation;
- create M6 grants;
- execute external actions during recovery;
- add Transformer/LLaMA/cloud inference;
- create another planner, memory engine, scheduler, or model inventory.

## Regression contract

`M18AMobileRecoveryTest.kt` verifies:

- normal process start touches activation only;
- execution-entry recovery touches activation only;
- reboot recovery executes activation -> autonomous -> paper;
- failure in autonomous does not suppress paper recovery;
- activation failure skips dependent recovery domains;
- activation readiness fails closed;
- recovery error detail remains inside its UTF-8 bound.

Android APK compilation remains the integration gate for the actual Application,
receiver, foreground service, JobService, and game-service wiring.
