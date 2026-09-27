# VN97-R2E4 Dynamic Runtime Autotuning

## Purpose

R2-E4 converts one verified R2-E3 real-device profiling receipt into a
persisted Android runtime tuning profile.

The chain is:

```
R2-E2 explicit-state ONNX bundle
-> R2-E3 real-device provider profiling
-> R2-E4 measured tuning profile
-> Android runtime adaptive provider/chunk decisions
```

R2-E4 does not alter VN97 weights, graph equations, checkpoint identity or
precision.

It remains one VN97 model executed through ONNX Runtime.

## Input truth

E4 requires:

- one verified E2 bundle;
- one verified E3 receipt bound to that exact bundle;
- exact architecture fingerprint match;
- exact fast/deep profile match;
- `device_measured=true`;
- `synthetic=false`;
- successful profiling for every E2 graph;
- a successful non-fallback CPU measurement for every graph.

The CPU requirement guarantees a measured final fallback instead of assuming
CPU compatibility.

## Output

Schema:

```
VN97R2E4TUNE1
```

The profile binds:

- E2 `bundle_id`;
- architecture fingerprint;
- fast/deep profile;
- E3 receipt ID;
- device/OS identity;
- graph-specific provider order;
- graph-specific measured p95 latency;
- preferred chunk ordering from measured throughput;
- thermal/memory/latency hysteresis policy;
- provider failure quarantine policy;
- XNNPACK thread limits.

The profile also locks:

```
same_weights_semantics = true
quantization_used = false
```

## Device identity

E4 tuning is intentionally device/OS specific.

The stable matching fields are:

- Android SDK level;
- logical core count;
- hardware string;
- SoC manufacturer;
- SoC model.

Available RAM and thermal status are runtime telemetry and are not part of the
stable device key.

If the phone receives an Android upgrade or the profile is copied to a
different SoC/device class, Android rejects the profile and should fall back to
the conservative E1 scheduler until a new E3 profile is measured.

## Graph-specific provider choice

Provider selection is not global.

Example:

```
step.onnx     -> XNNPACK -> CPU
chunk-32.onnx -> NNAPI -> CPU
chunk-64.onnx -> NNAPI -> XNNPACK -> CPU
```

The exact order comes from E3 measurements.

Fallback measurements are not considered provider wins.

If CPU is actually the fastest successful provider for a graph, the generated
chain is:

```
CPU
```

E4 does not force an accelerator ahead of a measured CPU winner.

## Chunk preference

For chunk graphs, E4 ranks the preferred non-fallback provider measurement by
measured p50 throughput.

The resulting list may look like:

```
64, 32, 16, 8
```

or another order if the measured phone behaves differently.

Runtime consumes this order greedily while respecting thermal, memory and
latency pressure.

The order is therefore measured evidence, not a hard-coded assumption that
larger chunks are always faster.

## Thermal hysteresis

Current control guardrails:

```
HOT enter      = 4
HOT exit       = 2
CRITICAL enter = 5
CRITICAL exit  = 3
```

These are runtime protection thresholds, not performance claims.

Behavior:

- NORMAL: full measured chunk preference;
- HOT: cap chunk size to the middle measured size and reduce XNNPACK threads;
- CRITICAL: use the smallest measured chunk and one XNNPACK thread.

Hysteresis prevents rapid oscillation when temperature moves around one
boundary.

## Memory-pressure hysteresis

Memory pressure is relative to free memory observed during E3 profiling.

Current guardrails:

```
enter pressure <= 30% of profiled available memory
exit pressure  >= 45% of profiled available memory
```

Under memory pressure E4 selects the smallest measured chunk size.

This reduces temporary inference working-set pressure without changing model
weights or state semantics.

## Latency hysteresis

E4 compares moving runtime latency against the E3 p95 for the graph that would
normally run next.

Current control guardrails:

```
enter SLOW >= 125% of measured p95
exit SLOW  <= 110% of measured p95
```

When SLOW, chunk size is capped similarly to HOT mode.

This lets runtime react to background load, sustained thermals and other
conditions not present during the original profile.

## Provider quarantine

Repeated provider failures are isolated without permanently disabling that
provider.

Current guardrails:

```
failure threshold = 2
cooldown decisions = 16
```

After two failures, the provider is removed from graph provider chains for 16
scheduler decisions.

After cooldown it becomes eligible again.

A successful invocation immediately clears prior failure/quarantine state.

CPU is never quarantined by this mechanism.

## Realtime path

E4 keeps sequential recurrence for:

- realtime workloads;
- sequence length <= 4;
- strong state dependency >= 0.85.

Those workloads use repeated `step.onnx` invocations and the provider policy
measured specifically for the step graph.

This is important because the fastest prefill provider does not have to be the
fastest recurrent-token provider.

## Hybrid and parallel path

For non-realtime workloads E4 decomposes the request with the measured chunk
preference.

If the entire workload fits one selected chunk invocation, the decision can be
classified PARALLEL.

If multiple chunk/step invocations are required, it is HYBRID.

State continues explicitly between invocations using the E2 recurrent-state
contract.

## Build tuning profile

After an official E3 device receipt exists:

```bash
vn97-r2-onnx-autotune build \
  --bundle-dir /data/vn97-onnx \
  --receipt /data/s21fe-e3-profile.json \
  --output /data/s21fe-e4-tuning.json
```

Verify:

```bash
vn97-r2-onnx-autotune verify \
  --bundle-dir /data/vn97-onnx \
  --profile /data/s21fe-e4-tuning.json
```

Inspect:

```bash
vn97-r2-onnx-autotune status \
  --bundle-dir /data/vn97-onnx \
  --profile /data/s21fe-e4-tuning.json
```

## Android runtime

Android loads the persisted profile with:

```
OrtAutotuneProfile.load(...)
```

The load path verifies:

- tuning profile SHA identity;
- E2 bundle ID;
- device/SDK identity;
- same-weights lock;
- no-quantization lock.

Then:

```
VN97OrtAutotuner.decide(...)
```

returns an ordered list of graph invocations, each containing:

- graph filename;
- sequence length;
- provider chain;
- XNNPACK thread count.

Provider success/failure is fed back through:

```
recordProviderSuccess(...)
recordProviderFailure(...)
```

## Persistence

The E4 JSON profile itself is the persisted device tuning artifact.

Transient thermal/memory/latency bands and provider quarantine state remain
runtime state and are rebuilt after process restart.

This is deliberate: a stale thermal or temporary failure condition is not
persisted across reboot/process death as if it were permanent hardware truth.

## S21 FE rule

E4 still does not claim an S21 FE winner before E3 is run on the actual phone.

Possible valid outcomes include:

- XNNPACK wins `step.onnx`;
- NNAPI wins one or more chunk graphs;
- CPU wins a graph;
- NNAPI fails for a graph and is omitted there.

E4 will encode whichever result the measured E3 receipt proves.

## CI truth boundary

CI may construct synthetic fixture receipts only to test algorithms.

A production E4 profile for the target phone must be built from an official
real-device E3 receipt.

Therefore CI PASS does not mean the S21 FE has already been benchmarked.

## Next milestone

R2-E5 will perform and seal the actual Galaxy S21 FE benchmark/hardening
campaign:

- install/export candidate E2 graph set;
- run E3 on-device measurement;
- build E4 tuning profile;
- run sustained warm/cold latency tests;
- verify thermal recovery;
- verify memory-pressure fallback;
- verify provider quarantine/recovery;
- select the smallest useful graph set for the final APK.
