# M15F — Paper Portfolio Performance Evidence & Retrospective Strategy Signals

M15F closes the VN97 M15 trading-agent architecture with deterministic,
historical **paper-simulation evidence**. It does not add live-money trading.

## Canonical path

```text
fresh M15C market snapshot
  -> mark every currently held paper position at current bid
  -> same canonical VN97 M15B decision
  -> M15A deterministic paper risk / simulated fill
  -> post-decision conservative mark-to-market
  -> immutable VN97PPE1 evidence
  -> VN97MEM1 paper-performance episode
  -> bounded cross-session retrospective aggregate
```

There is still one VN97 model, one canonical cognition path, one existing
VN97MEM1 memory system, and no broker execution backend.

## Conservative deterministic valuation

`VN97PaperPerformanceEvaluator` never calls the model. For every open long
paper position it requires a fresh quote in the supplied `VN97MKT1` snapshot
and values the position at **bid**, the conservative liquidation side.

It calculates with overflow-checked arithmetic:

- cash;
- gross book cost;
- marked position value;
- marked equity;
- unrealized PnL;
- total PnL versus starting cash;
- total return in basis points;
- per-position marked value and unrealized PnL.

Scaled multiplication and return ratios use integer `BigInteger` arithmetic and
convert back with `longValueExact()`; no floating-point PnL arithmetic is used.

If a currently held symbol has no quote in the fresh snapshot, the wake fails
closed **before cognition or a new paper order**.

## Durable VN97PPE1 evidence

Each successfully evaluated paper episode records one immutable evidence file
under app-private no-backup storage.

Identity is bound to:

- session SHA-256 ID;
- Android paper-session job ID;
- episode number;
- market snapshot SHA-256 ID;
- market observation timestamp.

Evidence includes account sequence, marked equity, PnL/return, position count,
canonical VN97 decision, and deterministic paper outcome.

Properties:

- atomic create;
- SHA-256 payload integrity;
- strict UTF-8;
- symlink rejection;
- bounded record/session counts;
- filename must match decoded episode;
- directory must match decoded session ID;
- one session cannot contain multiple job identities;
- evidence episodes and observation times must be strictly increasing;
- identical re-append is idempotent;
- conflicting evidence for the same episode is rejected.

A crash can leave an explicit evidence gap. M15F permits such a gap rather than
inventing/reconstructing evidence that was never durably recorded.

## VN97MEM1 integration

After a paper decision and post-decision mark, the same production VN97MEM1
receives a bounded episodic record with source:

`vn97.trading.paper.performance`

The record contains session/episode/snapshot identity, marked equity, total PnL
and return basis points, and explicitly labels its maximum authority as:

`paper_simulation_only`

This gives later canonical VN97 cognition historical performance context without
creating a second memory database or a second trading brain.

## Cross-session retrospective aggregate

The evidence ledger derives bounded session summaries:

- evidence count;
- first/latest evidenced episode;
- latest marked equity;
- latest total PnL;
- latest return basis points;
- peak marked equity;
- maximum drawdown in micros and basis points;
- HOLD count;
- ORDER count.

The Android Paper Trading screen displays these as:

**Historical paper-performance evidence (simulation only; not a forecast).**

These values describe recorded simulation history. They are not a claim of
profitability, future performance, or live-trading fitness.

## Safety boundary

M15F never:

- stores broker credentials;
- creates exchange/broker SDK routes;
- deposits or withdraws funds;
- sends real orders;
- converts evidence into execution authority;
- relaxes M15A deterministic risk controls;
- bypasses fresh-snapshot anti-replay.

Any future live-money capability would require a separate M6 deny-by-default
design with explicit user authorization, exact broker/account scope, revocation,
hard deterministic risk limits, and durable action receipts.

## Regression coverage

`M15FPaperPerformanceEvidenceTest.kt` verifies:

- conservative bid-side valuation;
- equity, PnL and return arithmetic;
- rejection when a held symbol cannot be marked;
- immutable/idempotent evidence;
- explicit evidence gaps;
- latest performance;
- peak equity;
- maximum drawdown;
- HOLD/ORDER retrospective counts.

APK compilation verifies the M15D session integration, VN97MEM1 write path, and
M15E performance display.

## M15 closure

With M15A through M15F, the paper-trading agent has:

- deterministic paper execution and risk;
- canonical VN97 market reasoning;
- read-only fresh market acquisition;
- durable scheduled sessions and reboot reconciliation;
- user control and observability;
- deterministic portfolio/performance evidence and retrospective signals.

M15 can therefore be considered **architecturally complete as a governed
paper-trading agent**. It does not claim universal profitability and it has no
live-money authority.

The next roadmap milestone is **M16 — Capability Acquisition**.
