# M15D — Durable User-Started Paper-Trading Sessions

M15D turns the M15A–M15C paper-trading path into an explicitly user-started,
durable Android session without creating any live-money authority.

## Canonical path

```text
explicit startSession(config)
  -> VN97PTS1 durable session identity
  -> persisted Android JobScheduler wake
  -> one read-only fresh VN97MKT1 snapshot
  -> persist snapshot consumption before cognition
  -> same activated VN97 M15B reasoning
  -> M15A deterministic paper risk / VN97TRD1 journal
  -> VN97MEM1 episodic outcome record
  -> bounded next wake or terminal session
```

The architecture still contains one VN97 model, one canonical cognition path,
one sovereign memory system, and paper-only deterministic execution.

## VN97PTS1 durable ledger

Each paper session persists app-private under `noBackupFilesDir` and binds:

- immutable session ID and JobScheduler ID;
- exact activated VN97 model ID;
- exact HTTPS market endpoint;
- exact source ID and sorted symbol allowlist;
- user goal;
- fixed interval, deadline and episode budget;
- power constraints;
- wake/schedule counters;
- consumed snapshot ID and observation time;
- latest plan, decision and outcome;
- explicit lifecycle state.

Writes are atomic, fsynced, SHA-256 protected, strict UTF-8 and post-write
verified. Identity fields cannot mutate after creation. Terminal records are
immutable.

States are:

`SCHEDULED -> RUNNING -> SCHEDULED`

plus explicit `PAUSED`, user `STOPPED`, bounded `COMPLETED`, and
fail-closed `FAILED`.

## Fresh-snapshot crash invariant

Before VN97 cognition sees an accepted market snapshot, M15D durably advances:

- `episodesAttempted`;
- `lastSnapshotId`;
- `lastObservedNs`.

Therefore process death, JobService interruption, app update, or reboot cannot
cause the same consumed snapshot to be traded again. A later wake must supply a
different snapshot with strictly newer observation time.

An unchanged feed is not treated as a new episode. It records
`NO_FRESH_SNAPSHOT` and schedules another bounded wake.

## Scheduling and power

M15D uses a non-exported `JobService` protected by
`android.permission.BIND_JOB_SERVICE`.

Each job is one-shot and persisted. The manager explicitly schedules the next
wake only after state is durably committed. Sessions are bounded by:

- maximum active sessions;
- maximum episodes;
- maximum wakes;
- maximum schedule attempts;
- fixed interval bounds;
- optional wall-clock deadline;
- battery-not-low and charging constraints.

If Android stops a job because constraints change, it is eligible for system
retry only while the durable session is still non-terminal and not paused.
User pause/stop therefore cannot be bypassed by JobScheduler retry.

## Process death and reboot

The existing boot/package-update receiver now reconciles both autonomous M13
work and M15D paper sessions.

A durable `RUNNING` record after process death is reset to `SCHEDULED`.
Its already-consumed snapshot remains in VN97PTS1, so reconciliation schedules
new evidence rather than replaying the interrupted market state.

## VN97 ownership and memory

A scheduled paper wake obtains the application sovereign-execution lock and
temporarily hands off the foreground assistant before opening the same activated
VN97 model. If the assistant cannot safely hand off, the wake records
`VN97_BUSY` and reschedules instead of creating a parallel intelligence path.

Every completed reasoning episode appends a durable VN97MEM1 EPISODIC record
under source `vn97.trading.paper.episode`. Terminal state is also written best
effort under `vn97.trading.paper.terminal`.

Each session has a separate app-private VN97TRD1 paper account journal, avoiding
portfolio interference between concurrent simulations.

## User controls and reporting

`VN97PaperTradingSessionManager` exposes:

- `startSession(config)`;
- `pause(jobId)`;
- `resume(jobId)`;
- `stop(jobId)`;
- `report(jobId)`;
- `listReports()`;
- `reconcileAfterSystemRestart()`.

No session starts automatically merely because trading support exists.

## Security scope

M15D remains simulation only. Market networking is inherited from M15C and is
read-only HTTPS GET. The session does not contain broker credentials, broker
SDKs, transfer capability, order-routing endpoints, or live-money authority.

A future live trading route, if ever added, must be a separate M6
deny-by-default capability with explicit user authorization, exact broker/account
scope, deterministic risk ceilings, revocation, durable receipts, and no
inheritance of paper-session authority.

## Validation

`M15DPaperTradingSessionStoreTest.kt` validates durable round trips, lifecycle
updates, immutable identity, monotonic wake/episode/observation counters,
terminal immutability, HTTPS endpoint constraints, canonical symbol order and
SHA-256 tamper detection.

The Android APK build remains the integration gate for JobService,
VN97Application, boot reconciliation, model ownership, M15C acquisition,
M15B cognition, VN97MEM1 and M15A journal integration.
