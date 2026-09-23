# M13C — Long-Horizon Autonomous Operations

M13C extends the existing M13A/M13B autonomous execution path without adding a second model, planner, memory engine, authority path, or backend.

The locked architecture remains:

`Code 1 -> Code 2 hardware/mobile-aware -> VN97 production`

with one VN97 model, one canonical planner, VN97MEM1 continuity and M6 authority.

## Durable scheduling contract

VN97GOA1 is upgraded to version 3 while retaining read compatibility with v1/v2.

Each autonomous generation can now persist:

- `notBeforeWallTimeMillis`
- `deadlineWallTimeMillis`
- `dependencyKey`
- `dependencySatisfied`
- `powerPolicy`
- `scheduleAttemptCount`
- `lastScheduledWallTimeMillis`
- `terminalNotificationSent`

The original identity fields from M13B remain immutable:

- job ID
- root job ID
- generation
- predecessor job ID
- plan ID
- model ID
- principal
- goal
- creation time

The new schedule identity is also immutable per generation:

- not-before time
- deadline
- dependency key
- power policy

Mutable scheduling state is monotonic. Dependency satisfaction cannot move from true back to false, schedule attempt counters cannot move backwards, and a terminal notification cannot become unsent again.

## Delayed execution and deadlines

`VN97AutonomousSchedulePolicy` lets a new goal be created with a future not-before time and an optional absolute deadline.

Android JobScheduler receives:

- minimum latency derived from the not-before time;
- an override deadline derived from the durable absolute deadline;
- the same persisted VN97CNT1/VN97ACB1 binding.

The planner is not run before the durable time gate is satisfied.

The continuation also re-checks the absolute deadline immediately before advancing the canonical planner. If the deadline has expired, the planner is cancelled with a terminal deadline reason and the generation completes without performing more work.

This prevents a stale Android scheduling wake from executing work after the user-defined deadline.

## Durable event dependency

A goal may specify one bounded `dependencyKey`.

Until the event is satisfied:

- VN97GOA1 is `WAITING_DEPENDENCY`;
- any pending Android job is cancelled;
- VN97CNT1/VN97ACB1 remain durable;
- the canonical planner is not advanced.

`signalDependency(key)` marks matching active goals satisfied and schedules them again through the same persisted continuation.

Dependency satisfaction is one-way and survives process death/reboot.

The event gate is orchestration only. It does not replace planner reasoning and does not grant authority for an external action.

## Power-aware scheduling

M13C adds a persisted per-goal power policy:

- `ADAPTIVE`
- `BATTERY_NOT_LOW`
- `CHARGING_ONLY`

These policies become native JobScheduler constraints.

At actual execution time, the existing `AndroidComputeGovernor` remains authoritative for dynamic battery/thermal state and selects BLOCKED, LOW_POWER, BALANCED or PERFORMANCE mode.

Therefore M13C uses two layers:

1. coarse Android scheduling constraints to avoid unnecessary wakeups;
2. the existing VN97 compute governor for per-wake thermal/battery budgets.

No second compute policy is introduced.

## Reboot and package-update reconciliation

The existing boot/package-update receiver now also launches bounded autonomous reconciliation.

For every active goal, ordered by nearest deadline first:

- expired deadlines are terminalized;
- `WAITING_APPROVAL` stays foreground-only;
- unresolved event dependencies remain `WAITING_DEPENDENCY`;
- planner-paused goals remain paused;
- stale `RUNNING` records are reset to `SCHEDULED`;
- a missing persisted JobScheduler entry is recreated from the exact VN97GOA1/VN97ACB1 identity;
- existing pending jobs are not duplicated.

The reconciliation pass is bounded to 32 active goals.

## Multiple long-running goals

The existing M13 bound of 32 active autonomous goals remains.

M13C reconciliation prioritizes nearer deadlines and never creates a second planner for concurrency. Each goal continues to own one immutable generation binding and the process-wide sovereign execution lock still serializes production model/memory ownership when actual cognition runs.

Explicit event fan-out is bounded to 32 goals per signal.

Explicit schedule attempts are bounded to 64 per generation. Reaching the bound fails the generation rather than looping indefinitely.

Existing M13B generation and wake limits also remain:

- generation 0 plus at most 3 replan generations;
- at most 256 cognition wakes per generation.

## Replanning continuity

When M13B creates a successor generation, M13C carries forward:

- original not-before time;
- original deadline;
- dependency identity and satisfaction state;
- power policy.

The successor still receives a new plan ID, new job ID and new VN97CNT1/VN97ACB1 binding.

A replan therefore cannot escape the logical goal's time/dependency/power envelope.

## Completion and failure notification

M13C adds a dedicated autonomous notification channel.

Notifications are sent only for terminal logical outcomes:

- COMPLETED
- FAILED
- BUDGET_EXHAUSTED

Intermediate failed generations that already have a successor are not notified as logical-goal failures.

Notification text intentionally does not expose the goal contents. It contains only the root job/generation and terminal state.

On Android 13+, the app requests `POST_NOTIFICATIONS` once. If permission is unavailable, the durable terminal state remains authoritative and reboot reconciliation can retry notification later.

## M6 authority remains unchanged

Scheduling eligibility is not authority.

A long-horizon goal that reaches an external action still follows the exact M6 path:

`canonical planner -> immutable external request -> M6 gate -> user approval when required -> one-use lease -> execution -> durable audit`

A delayed wake, event signal, reboot reconciliation, power-state change or deadline does not create or persist approval tokens.

Reject and Cancel remain terminal decisions and cannot be bypassed by replanning.

## Result

After M13C, VN97 autonomous work can survive a long horizon as:

`create
-> wait for time/event
-> reboot/process death
-> reconcile
-> power-aware wake
-> canonical cognition
-> checkpoint
-> approval if required
-> continue
-> bounded replan generation
-> deadline/completion/failure
-> user notification`

All execution remains inside the same production VN97 architecture.
