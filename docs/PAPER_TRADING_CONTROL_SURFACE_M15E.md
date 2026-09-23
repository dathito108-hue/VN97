# M15E — Android Paper-Trading Control & Observability

M15E exposes the durable M15D paper-trading session engine to the Android user
without adding any live-money authority.

## User surface

The main VN97 activity contains one **Paper trading** entry point. It opens the
private, non-exported `VN97PaperTradingActivity`, which is scrollable and sized
for a mobile screen.

The paper screen supports:

- HTTPS `VN97MKTFEED1` endpoint configuration;
- exact source ID;
- bounded canonical symbol set;
- paper-only user objective;
- 15 second to 6 hour interval;
- 1 to 2048 episode budget;
- battery-not-low / charging-only scheduling constraints;
- explicit **Start paper session**;
- exact `jobId` selection;
- **Pause**, **Resume**, and **Stop**;
- **Refresh sessions**;
- visible state, episode/wake counters, last decision, last paper outcome,
  terminal reason, and next scheduled wall-clock time.

The screen states prominently that it is simulation-only.

## Architecture

```text
VN97MainActivity
  -> private VN97PaperTradingActivity
  -> bounded VN97PaperTradingControlSurface parser
  -> existing VN97PaperTradingSessionManager (M15D)
  -> persisted VN97PTS1 session
  -> M15C read-only HTTPS market snapshot
  -> same canonical VN97 M15B cognition
  -> M15A deterministic paper risk engine
  -> per-session VN97TRD1 + VN97MEM1 outcome memory
```

No second model, second planner, cloud AI backend, broker SDK, broker credential,
deposit/withdrawal route, or live order capability is introduced.

## UI input boundary

`VN97PaperTradingControlSurface` is pure Kotlin and validates UI text before it
reaches the durable manager:

- HTTPS only;
- no URL credentials or fragments;
- bounded source ID;
- uppercase/deduplicated/canonical symbols;
- max 64 symbols;
- bounded UTF-8 goal;
- interval 15..21600 seconds;
- episode budget 1..2048;
- positive exact session job IDs.

M15D performs the authoritative validation again when the configuration reaches
the session manager and persistence layer.

## Exact-session lifecycle control

Pause/resume/stop operate on an explicit positive Android JobScheduler `jobId`.
The UI does not infer a hidden account or silently control every session.

If the job-ID field is empty, refresh selects the newest non-terminal session
only as a convenience and displays its ID before any lifecycle action can be
pressed.

M15D remains authoritative for legal transitions:

- terminal sessions cannot resume;
- only PAUSED can resume;
- durable PAUSED/STOPPED state precedes scheduler cancellation;
- reboot reconciliation cannot revive a stopped session;
- consumed snapshots remain non-replayable.

## Observability

The UI displays up to eight recent session reports, including:

- state;
- `episodesAttempted / maxEpisodes`;
- wake count;
- last canonical VN97 decision;
- last deterministic paper result;
- terminal reason or next-run wall time.

Decision/outcome text is single-line and bounded before rendering.

## Regression coverage

`M15EPaperTradingControlSurfaceTest.kt` verifies:

- canonical trimming/casing/deduplication;
- HTTPS-only endpoint validation;
- URL credential rejection;
- empty-symbol rejection;
- interval bounds;
- episode-budget bounds;
- positive job IDs;
- paper-only report labeling;
- decision/outcome/terminal rendering.

APK compilation verifies the Android activity, manifest registration and the
binding to the existing M15D manager.

## Scope

M15E completes the **user-visible paper-trading control surface**. It does not
authorize real trades.

A future live-money milestone, if ever added, must be architecturally separate
from paper authority and pass M6 deny-by-default controls with explicit broker
and account scope, explicit user authorization, revocation, deterministic hard
risk limits and durable receipts.

## Next

M15F should close the trading-agent milestone with portfolio/strategy evaluation
and bounded paper-performance evidence across sessions, then M15 can be assessed
for architectural closure before moving to M16 Capability Acquisition.
