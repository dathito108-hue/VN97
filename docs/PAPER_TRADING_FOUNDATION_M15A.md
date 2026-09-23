# M15A — Paper Trading Foundation

M15A establishes the deterministic trading substrate for VN97 without introducing a
broker dependency, a second AI model, or live-money execution.

## Architecture

M15A remains inside the locked VN97 architecture:

`one VN97 model -> canonical planner/cognition -> VN97 platform runtime`

The milestone adds a local paper-trading account that can later be driven by the same
VN97 reasoning path. It does not create any live broker/network order capability.

## Fixed-point representation

Trading arithmetic does not use floating point.

- price scale: 1,000,000 micros per currency unit;
- quantity scale: 1,000,000 microunits per asset unit;
- notional is computed with integer/BigInteger multiply-divide and checked back into
  signed 64-bit bounds.

This makes replay deterministic across Android processes.

## Market quote contract

A quote contains:

- canonical uppercase symbol;
- positive bid;
- ask >= bid;
- monotonic/non-negative timestamp supplied by the caller.

A paper order is permanently bound to the exact quote timestamp it observed. Execution
rejects a supplied quote if its timestamp differs from the order, preventing accidental
execution against a different snapshot.

## Paper execution

M15A supports long-only BUY and SELL simulation:

- BUY fills at ask;
- SELL fills at bid;
- SELL cannot exceed held quantity;
- duplicate order IDs are rejected;
- execution cannot predate order creation;
- failed orders are never journaled.

The engine returns a deterministic fill receipt with notional, fee, cash-after,
position quantity-after, book-cost-after and journal sequence.

## Risk envelope

`VN97PaperTradingRiskPolicy` bounds:

- maximum order notional;
- maximum gross book cost;
- minimum cash reserve;
- simulated fee basis points;
- maximum simultaneously held symbols.

The default account is simulation-only and creates no external financial authority.

## Durable journal — VN97TRD1

The account is reconstructed from an append-only, fsync-backed journal.

Genesis record:

`VN97TRD1|G|...`

Order record:

`VN97TRD1|O|orderId|symbol|side|quantity|bid|ask|quoteTs|createdNs|executedNs`

On restart the account replays every recorded order through the same deterministic
transition function. The configured starting cash and risk policy must exactly match
the journal genesis record.

Malformed or inconsistent journal content fails closed; M15A does not silently repair
or reinterpret trading history.

## Android production storage

`AndroidPlatformRuntime.openProductionPaperTradingAccount()` places the journal under
app-private no-backup storage in `vn97-trading/` and rejects symlinked roots/targets.

No credential, API key, brokerage SDK, exchange session or cloud AI dependency is
added.

## Regression contract

The M15A host regression covers:

- deterministic BUY fill;
- deterministic partial SELL fill;
- fixed fees;
- duplicate order rejection;
- quote timestamp binding;
- maximum order notional;
- oversell/short rejection;
- canonical symbol validation;
- failed-order non-persistence;
- restart replay equality;
- risk-policy mismatch rejection.

APK compilation remains the Android integration gate for the production runtime
factory.

## Next milestone

M15B should add a bounded market-observation layer and a VN97 trading reasoning loop
that consumes explicit market snapshots and produces paper-order proposals. Live-money
execution should remain a separate later capability with its own explicit authority,
risk and approval boundary.
