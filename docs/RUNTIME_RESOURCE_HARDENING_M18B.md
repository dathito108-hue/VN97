# M18B — Runtime Resource Hardening

M18B hardens the canonical VN97 mobile runtime against Android memory pressure,
thermal pressure, low battery, and idle native-resource retention.

It does not add a second runtime, model, scheduler, planner, memory engine, or
authority path.

The production architecture remains:

```text
Code 1
  -> Code 2 hardware/mobile-aware
  -> signed VN97MI1
  -> canonical VN97 runtime/planner/VN97MEM1/M6
  -> Android production surfaces
```

## Extended compute signals

The existing M7C `ComputeSignals` contract now also carries bounded memory
pressure:

- `MEMORY_NORMAL`;
- `MEMORY_MODERATE`;
- `MEMORY_CRITICAL`.

Existing three-argument construction remains source-compatible through the
normal-memory default.

`AndroidComputeGovernor` samples:

- battery percentage;
- charging state;
- Android thermal status;
- `ActivityManager.MemoryInfo.lowMemory`;
- available-memory / low-memory-threshold ratio.

If Android memory information cannot be obtained, VN97 falls back to
`MEMORY_MODERATE` rather than assuming unlimited memory.

## M7C policy preservation

The existing M7C battery/thermal policy remains authoritative for background
continuations.

M18B only adds:

- critical memory -> `BLOCKED`;
- moderate memory -> `LOW_POWER`.

Therefore the existing persisted Android continuation JobService becomes
memory-aware without creating another scheduler or continuation implementation.

## Execution classes

The app-level resource policy classifies new work into three resource classes.

### INTERACTIVE

Used for user-directed chat, voice and knowledge proposal work.

Interactive work:

- is blocked by critical memory pressure;
- is blocked by severe thermal pressure;
- remains available at critically low battery because the request is
  user-initiated;
- drops the foreground advance budget from 8 to 4 under moderate memory,
  moderate thermal, or <=20% unplugged battery.

This keeps the assistant responsive while reducing sustained computation.

### BACKGROUND

Used for autonomous seed work and paper/background execution.

Background work directly follows the M7C compute budget. A blocked budget
remains blocked; LOW_POWER remains runnable but bounded.

### HEAVY

Used for:

- governed game episodes;
- native visual perception/verification;
- mobile evidence benchmarking.

Heavy work requires a non-LOW_POWER compute budget and normal memory pressure.
It therefore fails closed before allocating additional expensive inference
work when Android is already resource-constrained.

## Idle assistant release

`VN97Application` now initializes one process-wide
`VN97RuntimeResourceCoordinator`.

The coordinator tracks visible Activities using Android
`ActivityLifecycleCallbacks`.

When Android reports trim/low-memory pressure:

- visible UI -> the assistant is retained;
- no visible Activity -> VN97 may release the idle assistant/model;
- an active turn is never force-closed;
- a pending M6 approval is never discarded;
- a pending knowledge-acquisition review is never discarded.

The release path reuses `VN97AppAssistant.closeLocked()`, so production
assistant resources and the activated native model are closed through their
existing ownership paths.

No explicit GC call and no unsafe native-handle invalidation is introduced.

## Reopen semantics

Resource release is intentionally reversible.

The same activated VN97 inventory remains authoritative. Interactive execution
reopens the exact activated model through the existing
`NativeActivatedInventoryModelLoader` and production memory-backed assistant
construction path.

The main Activity also reattaches on resume when resources were released while
the UI was hidden. Reattach is idempotent when app state is already READY.

The floating assistant re-runs the M18A recovery barrier and M18B resource gate
before every text or voice turn, closing the race where the overlay remained
alive after Android trimmed the native assistant.

## Production surface integration

M18B gates:

- foreground model attach;
- chat turns;
- local voice turns;
- knowledge-gap proposal inference;
- native visual perception;
- visual post-action verification;
- autonomous continuation seed creation;
- governed game-agent episodes;
- scheduled paper-trading episodes;
- mobile performance evidence collection;
- persisted M7C continuation execution through the extended compute governor.

Paper JobScheduler retry behavior from M18A is preserved when resource policy
temporarily blocks execution.

## Failure behavior

Resource denial happens before new expensive model work begins.

No resource-policy failure:

- activates a different model;
- changes inventory;
- approves a capability;
- mints M6 authority;
- modifies a planner;
- writes alternate memory state;
- falls back to cloud/Transformer/LLaMA inference.

The caller receives the normal fail-closed exception/status and may retry after
the device resource condition changes.

## Regression contract

`M18BRuntimeResourcePolicyTest.kt` verifies:

- old M7C compute-policy behavior remains valid;
- moderate memory produces LOW_POWER;
- critical memory produces BLOCKED;
- interactive work is reduced, not blocked, by moderate pressure;
- interactive work remains available on very low unplugged battery but with a
  reduced advance budget;
- background work remains blocked by the M7C critical-battery rule;
- severe thermal blocks interactive work;
- heavy work is blocked under moderate memory;
- trim thresholds request idle-resource release;
- invalid memory-pressure values fail closed.

Full Android APK compilation remains the integration gate for
`ActivityManager`, Activity lifecycle callbacks, Application trim callbacks,
foreground/floating assistant wiring, JobService, and game-service integration.
