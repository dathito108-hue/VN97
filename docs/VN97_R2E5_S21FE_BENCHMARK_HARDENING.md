# VN97-R2E5 Galaxy S21 FE Benchmark and Hardening

## Purpose

R2-E5 is the first milestone in the ONNX runtime line that requires a real
target phone for production evidence.

It consumes:

- one verified R2-E2 explicit-state ONNX bundle;
- one verified R2-E3 real-device provider profiling receipt;
- one verified R2-E4 tuning profile built from that E3 receipt;
- one on-device R2-E5 benchmark/hardening run.

It produces:

```
VN97R2E5HARDEN1
```

The seal records whether the measured target run satisfies the project
hardening gates and which ONNX graphs are sufficient to retain in the final
APK candidate.

## Truth boundary

CI can validate:

- Python run/seal parsers;
- SHA/provenance binding;
- hardening arithmetic;
- Android harness compilation;
- E4 control-state behavior.

CI cannot claim:

- Galaxy S21 FE latency;
- NNAPI speed;
- XNNPACK speed;
- thermal stability on the phone;
- final APK performance.

A production E5 seal requires:

```
device_measured = true
synthetic = false
```

and, by default, an Android model in the Samsung Galaxy S21 FE model family.

## Target family

The initial target contract is:

```
target_family = galaxy_s21_fe
Samsung model prefix = SM-G990
```

This covers the S21 FE family without forcing one SoC variant.

E3/E4 still use the actual reported:

- hardware;
- SoC manufacturer;
- SoC model;
- Android SDK level;
- logical core count.

Therefore an Exynos and Qualcomm S21 FE do not silently share one tuning
profile.

## Android harness

The runtime exposes:

```
VN97OrtHardeningHarness
```

It consumes the exact E4 profile and an `OrtHardeningExecutor`.

The executor is intentionally an adapter interface rather than a second
inference engine. It must execute the same E2 explicit-state graph/state-carry
path used by production VN97.

This avoids benchmarking a special implementation that the final APK would
never use.

## Required measured phases

An official `VN97R2E5RUN1` contains exactly one canonical phase of each kind:

```
cold
warm_step
prefill
sustained
recovery
```

The default Android recommendation is:

- cold: one recurrent token after state reset;
- warm-step: 32 recurrent token iterations;
- prefill: 8 medium/large prefill iterations;
- sustained: 32 longer prefill iterations;
- recovery: 8 recurrent iterations after a 15-second cooldown window.

These are project benchmark defaults, not universal performance standards.

The phase specification remains explicit in the run evidence.

## Per-phase evidence

Every measured phase records:

- phase name/kind;
- iteration count;
- tokens per iteration;
- per-iteration latency in nanoseconds;
- graph invocation counts;
- actual provider counts;
- thermal status before/after;
- available memory before/after;
- execution failure count.

The run binds all phase evidence into one SHA identity.

## Why tokens_per_iteration is mandatory

Warm recurrent execution and sustained prefill do not perform the same amount
of work.

E5 therefore never compares raw sustained latency directly to one-token warm
latency.

The hardening seal normalizes:

```
p95_latency_per_token =
    phase_p95_latency / tokens_per_iteration
```

and compares sustained per-token p95 against warm-step per-token p95.

This avoids a false failure merely because a sustained phase intentionally
processes a longer sequence.

## Sustained degradation guard

Current project guardrail:

```
sustained_per_token_p95 <= 2.0 * warm_step_per_token_p95
```

represented as:

```
max_sustained_p95_over_warm_ppm = 2_000_000
```

This is a hardening acceptance guard, not a claim that 2x is an optimal
industry threshold.

The raw p95 values and ratio are retained in the seal so the criterion can be
reviewed later without hiding the evidence.

## Execution failure guard

Official hardening requires:

```
failure_count = 0
```

across the canonical measured phases.

Provider fallback is not itself a failure when the production provider chain
successfully completes the invocation.

Unhandled graph/provider execution failures do fail the E5 hardening seal.

## Thermal recovery guard

The recovery phase begins after a cooldown interval.

E5 requires:

```
recovery_thermal_after <= sustained_thermal_after
```

This only proves that the measured recovery phase did not end at a worse
thermal status than the sustained phase.

It does not claim complete device cooldown.

## Safe control-path tests

E5 does not intentionally allocate memory until OOM or heat the device into a
dangerous state.

The Android harness separately tests E4 control logic with telemetry injection:

- thermal hysteresis;
- memory-pressure response;
- provider quarantine;
- provider recovery;
- CPU fallback presence.

These are recorded as boolean control tests.

They verify state-machine behavior without pretending the injected telemetry
is a physical benchmark measurement.

## Provider quarantine control

For a non-CPU provider present in the tuning profile, E5:

1. records the configured number of failures;
2. verifies the provider is absent during quarantine;
3. advances the configured cooldown decision count;
4. verifies the provider becomes eligible again.

If the measured E4 profile legitimately contains CPU-only policies, quarantine
and recovery are treated as not-applicable/pass rather than manufacturing a
fake accelerator failure.

CPU is never quarantined.

## Run identity

Schema:

```
VN97R2E5RUN1
```

The Android run binds:

- E2 bundle ID;
- E3 receipt ID;
- E4 tuning ID;
- Samsung manufacturer/model;
- SDK/core/hardware/SoC identity;
- all measured phases;
- all control tests;
- real-device/synthetic flags.

Identity:

```
SHA256("VN97R2E5RUN1\0" + canonical_json(body))
```

## Verify copied run evidence

After copying the JSON from the phone:

```bash
vn97-r2-onnx-harden verify-run \
  --bundle-dir /data/vn97-onnx \
  --e3-receipt /data/s21fe-e3.json \
  --e4-profile /data/s21fe-e4.json \
  --run /data/s21fe-e5-run.json
```

The verifier rejects a run bound to another:

- E2 bundle;
- E3 receipt;
- E4 tuning profile.

## Seal hardening

```bash
vn97-r2-onnx-harden seal \
  --bundle-dir /data/vn97-onnx \
  --e3-receipt /data/s21fe-e3.json \
  --e4-profile /data/s21fe-e4.json \
  --run /data/s21fe-e5-run.json \
  --output /data/s21fe-e5-hardening.json
```

By default the device must match the Galaxy S21 FE family.

For development-only evidence on another Android device:

```
--allow-non-s21fe
```

does not set `target_device_match=true`; it only allows the seal operation to
complete for comparison work.

## Hardening result

Schema:

```
VN97R2E5HARDEN1
```

The result includes:

- exact E2/E3/E4/run identities;
- target-device match;
- warm-step p95;
- sustained p95;
- normalized per-token p95 values;
- sustained/warm ratio;
- execution failure count;
- control-test result;
- thermal recovery result;
- final `hardening_passed`;
- minimal APK graph recommendation.

Integrity verification and acceptance are distinct.

A valid seal may have:

```
hardening_passed = false
```

and still be useful evidence explaining what failed.

## Minimal APK graph recommendation

E5 recommends a conservative graph subset:

1. `step.onnx`;
2. the highest-throughput preferred E4 chunk;
3. the smallest measured E4 chunk for pressure fallback, when different.

Example:

```
step.onnx
chunk-64.onnx
chunk-8.onnx
```

This recommendation is only a candidate for later APK packaging.

Graphs must not be removed from production solely because CI fixtures did not
exercise them.

The recommendation becomes meaningful only after the real S21 FE campaign is
sealed.

## One-model rule

E5 does not introduce another AI model or backend.

The whole mobile path remains:

```
one VN97 checkpoint
-> E2 ONNX graphs with shared weights semantics
-> ONNX Runtime
-> E3 measured provider evidence
-> E4 adaptive tuning
-> E5 hardening
```

The graph variants are execution shapes for the same VN97 model, not separate
models.

## Quantization

E5 still locks:

```
quantization_used = false
same_weights_semantics = true
```

Production INT4/ternary/QAT lowering remains blocked until dense model training
and fresh quality validation succeed.

## What remains after E5 infrastructure

Once an actual validated VN97 checkpoint and E2 bundle exist, the physical S21
FE campaign is:

1. deploy the exact E2 bundle to the phone;
2. run E3 provider profiling;
3. build E4 tuning;
4. run the E5 canonical measured phases;
5. copy the E5 run off-device;
6. verify and seal it;
7. investigate any failed hardening gate;
8. repeat only after a real code/profile change;
9. use the successful seal to guide final APK graph packaging.

Until steps 1-6 happen on the real phone, VN97 must not claim measured S21 FE
performance.
