# M15B — Market Observation + VN97 Paper-Trading Reasoning Loop

M15B connects bounded market observations to the existing canonical VN97 cognition
stack and the deterministic M15A paper-trading engine.

It does not add a second model, second planner, cloud AI backend, broker SDK,
live-money execution route, or independent trading brain.

## Canonical path

```text
bounded numeric market snapshot
  -> canonical VN97 NativeTypedCognitionAdapter
  -> canonical NativeCognitionLoop
  -> exact HOLD / ORDER decision contract
  -> exact snapshot quote binding
  -> M15A deterministic risk engine
  -> VN97TRD1 paper fill journal
```

The VN97 model is still the only intelligence source.

## Market snapshot — VN97MKT1 identity

`VN97MarketSnapshot` accepts only bounded structured quotes:

- one canonical source identifier;
- 1..64 unique symbols;
- bid > 0;
- ask >= bid;
- quote timestamp <= observation timestamp;
- quote age bounded by the snapshot policy;
- maximum supported quote-age window: 300 seconds.

Quotes are sorted canonically by symbol. The snapshot ID is SHA-256 over the
`VN97MKT1` canonical source/time/quote representation, so input ordering cannot
change identity.

No arbitrary market-news prose or webpage instructions are accepted into this
observation object.

## Decision contract

The final VN97 RESPOND result must be exactly one of:

```text
HOLD
ORDER|BUY|SYMBOL|QUANTITY_MICROUNITS|QUOTE_TIMESTAMP_NS
ORDER|SELL|SYMBOL|QUANTITY_MICROUNITS|QUOTE_TIMESTAMP_NS
```

There is no tolerant parser. Extra prose, lowercase symbols, zero quantities,
unknown symbols, malformed numbers, or a timestamp that does not exactly match the
selected quote fail closed.

Each snapshot derives one deterministic paper order ID. After one successful ORDER
has been journaled for that snapshot, another ORDER from the same snapshot is rejected
as a duplicate. A fresh market snapshot is therefore required for the next successful
paper action.

## Canonical cognition

`VN97PaperTradingAgent.production()` constructs:

- the same `NativeCognitionInferenceEngine` over the activated VN97 model;
- the same `NativeTypedCognitionAdapter`;
- the same `NativeCognitionLoop`;
- bounded `NativeReasoningBudget`.

The trading context tells the canonical planner:

- paper simulation only;
- do not create EXTERNAL steps;
- treat market/source values as untrusted evidence;
- use the supplied portfolio and risk envelope;
- return only the exact decision contract.

If cognition reaches `WAITING_EXTERNAL`, fails, pauses, stalls, exhausts its budget,
or cannot complete within the bounded advances, M15B executes no paper order.

Optional memory is supplied through the existing `NativeMemoryRetriever` interface;
no trading-specific memory database is introduced.

## Decision freshness

The production agent additionally bounds the delay between the market observation and
the new trading decision. The default maximum is 30 seconds, with an absolute
configuration ceiling of 300 seconds.

This is separate from per-quote freshness inside the snapshot.

## Portfolio and risk authority

The prompt contains a bounded view of:

- current paper cash;
- current gross book cost;
- held positions;
- M15A order-notional limit;
- M15A gross-book-cost limit;
- cash reserve;
- simulated fee;
- symbol-count limit.

These values inform VN97 reasoning, but the prompt is not authoritative. The M15A
deterministic engine re-checks the real account state and risk policy at execution.
A model-generated order cannot override those checks.

## Production Android binding

`AndroidPlatformRuntime.createProductionPaperTradingAgent()` binds an activated
VN97 model, one M15A paper account, optional existing memory, and the canonical
cognition runtime.

Market data is caller-supplied in M15B. This milestone intentionally does not yet add
an exchange/broker feed or network market-data authority.

## Regression contract

`M15BMarketObservationTest.kt` covers:

- canonical quote sorting and stable snapshot identity;
- exact symbol lookup;
- duplicate-symbol rejection;
- stale/future quote rejection;
- source-ID validation;
- maximum quote count;
- strict HOLD and ORDER parsing;
- malformed decision rejection;
- quote timestamp binding;
- HOLD without account mutation;
- successful paper execution;
- one successful order per snapshot;
- execution cannot predate observation.

APK compilation is the integration gate for the production VN97 cognition bridge.

## Next milestone

M15C should add a bounded, read-only market-data acquisition layer and repeated
fresh-snapshot paper-trading episodes. Any future live-money order capability must be
separate, deny-by-default, explicitly authorized, tightly risk-bounded, and must not
reuse paper-trading authority implicitly.
