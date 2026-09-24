# M18C — Android Failure Containment + Watchdog Hardening

M18C adds bounded failure containment around the existing VN97 Android
execution paths. It does not introduce a second scheduler, model, planner,
memory engine, or authority system.

The canonical architecture remains:

```text
Code 1
  -> Code 2 hardware/mobile-aware
  -> signed VN97 production model
  -> canonical planner / VN97MEM1 / M6
  -> Android production surfaces
```

## Durable execution health

M18C adds the `VN97HEALTH1` execution-health record.

A record is scoped by:

- execution domain;
- stable non-sensitive key;
- monotonically increasing generation;
- start/update/deadline timestamps;
- terminal state;
- consecutive infrastructure-failure count;
- bounded failure detail;
- optional cooldown deadline.

Health records are written with:

- a bounded file size;
- SHA-256 integrity over the canonical payload;
- fsync of the file;
- atomic replace;
- directory fsync;
- post-write verification.

Goal text, game prompts, market data, user messages and approval payloads are
not used as health keys.

## Execution domains

M18C currently protects:

- process-start mobile recovery;
- persisted VN97 continuation jobs;
- paper-trading JobService wakes;
- governed game-agent episodes.

These wrappers do not replace the work itself. They only record and bound the
existing execution path.

## Stale-run recovery

Every protected run starts by creating a durable `RUNNING` lease with a
deadline.

If the process dies before the lease reaches a terminal state, a later start can
identify the previous lease as abandoned once:

```text
now > deadline + stale grace
```

The stale run becomes an `ABANDONED` infrastructure failure before a new
generation is considered.

This provides evidence for process death or hard termination that could not run
normal `finally` cleanup.

## Crash-loop guard

Infrastructure states that increase the failure streak are:

- `FAILED`;
- `TIMED_OUT`;
- `ABANDONED`.

After three consecutive failures for the same domain/key, automatic work is
suppressed for 15 minutes.

A successful execution resets the streak.

`CANCELLED` is distinct from failure. Android preemption and explicit user
cancellation therefore do not create a false crash-loop penalty.

Foreground user actions are not silently redirected to another runtime when a
background path is suppressed.

## Cooperative watchdog

M18C adds a watchdog that uses the work's existing bounded cancellation flag.

The watchdog:

1. waits until the durable lease deadline;
2. sets the same cooperative cancellation flag already understood by the work;
3. atomically records `TIMED_OUT`;
4. never destroys a native handle from another thread;
5. never force-unlocks a mutex;
6. never mints authority or executes an external action.

If the worker later exits normally, its final write cannot overwrite the
already-terminal timeout evidence because lease generation/state are checked.

## Continuation jobs

The existing `VN97ContinuationJobService` now:

- creates one health lease per JobScheduler job ID;
- derives the watchdog duration from the existing
  `ComputeBudget.maxRunMillis`;
- requests reschedule if a start is suppressed or times out;
- records cancellation without incrementing the failure streak;
- still persists/suspends runtime state through the existing VN97 continuation
  code.

Android `onStopJob()` remains the source of OS-preemption semantics.

## Paper trading

The paper JobService uses the existing Android compute governor for its
watchdog duration and the existing paper-session retry policy for rescheduling.

The underlying M15 paper engine is unchanged.

## Game agent

The game agent uses one non-sensitive `game-agent` health identity because the
service is already single-flight.

Its watchdog covers:

- foreground-game wait;
- the existing maximum game episode duration;
- a small bounded shutdown margin.

The existing game cancellation checks, action cap, episode timer, screen
authorization and M6 game-control grants remain authoritative.

## Bounded sovereign lock acquisition

M18C does not try to force-unlock VN97 state.

Instead, background paths that can otherwise wait indefinitely now use bounded
lock acquisition:

- mobile recovery;
- autonomous continuation;
- paper trading wake;
- game episode.

If the sovereign or recovery lock cannot be acquired inside the bound, the work
fails/retries through its existing path and health evidence records the
infrastructure failure.

This converts a possible unbounded worker wait into a bounded failure without
risking state corruption.

## Process-start recovery

Automatic process-start recovery receives its own health lease and watchdog.

Repeated failed/abandoned automatic process-start recoveries can therefore enter
cooldown instead of repeatedly replaying the same failing bootstrap.

The M18A execution-entry recovery barrier remains intact. A foreground or
production execution entry still performs its normal activation recovery check;
M18C does not bypass activation or promotion reconciliation.

## Authority boundary

M18C does not:

- approve an M17 candidate;
- alter active-model inventory;
- bypass M6;
- execute tools from a watchdog;
- create another planner or memory store;
- introduce cloud, Transformer or LLaMA fallback;
- force-close native resources owned by another execution thread.

## Regression contract

`M18CExecutionHealthTest.kt` verifies:

- first lease creation;
- overlapping-start suppression;
- successful terminal transition and failure-streak reset;
- repeated failure counting;
- crash-loop cooldown after three failures;
- restart after cooldown;
- cancellation does not add a failure;
- stale `RUNNING` detection and abandoned-run accounting;
- UTF-8 failure-detail bounds;
- tampered durable evidence fails verification.

Android APK compilation remains the integration gate for the actual
JobService/Service watchdog wiring and bounded sovereign lock paths.
