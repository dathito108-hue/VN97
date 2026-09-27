# VN97-R2E3 Android ONNX Provider Profiling

## Purpose

R2-E3 measures the real Android execution behavior of one verified R2-E2
explicit-state ONNX bundle.

It does not change VN97 weights, graph equations, quantization, or training.

The chain is:

```
R2-E2 ONNX bundle
-> R2-E3 device profiler
-> NNAPI / XNNPACK / CPU / optional QNN measurements
-> VN97R2E3PROFILE1 receipt
-> R2-E4 runtime autotuning
```

E3 infrastructure can be compiled and tested in CI, but an official E3 receipt
must come from a real Android device.

## What is measured

For every selected ONNX graph and provider:

- requested provider;
- actual provider selected by the E1 session factory;
- fallback occurrence;
- session-creation time;
- warmup count;
- steady-state inference samples;
- p50 and p95 steady-state latency;
- minimum and maximum steady-state latency;
- p50 throughput;
- available memory before/after;
- thermal status before/after;
- provider/session failures.

Step and chunk graphs are measured independently.

This is necessary because a provider that is fastest for `chunk-64.onnx` may
not be fastest for `step.onnx`.

## No allocation noise in steady-state timing

The Android profiler uses a two-stage callback:

```
prepare(session)
-> reusable OrtPreparedInvocation
-> run(session)
```

The caller should create input tensors, recurrent-state tensors and reusable
output bindings during `prepare`.

Warmup and steady-state loops call only `run`.

Therefore tensor allocation/setup is not intentionally mixed into the reported
steady-state inference latency.

Session creation is measured separately.

## Provider isolation

From R2-F1 onward, accelerator sessions also set
`session.disable_cpu_ep_fallback=1`. A QNN/NNAPI/XNNPACK trial therefore
fails session creation when that provider cannot cover the graph instead of
silently assigning unsupported nodes to the ORT CPU EP. CPU is profiled in its
own isolated trial.


E3 profiles one requested provider at a time.

For NNAPI, for example, the isolated session chain is:

```
NNAPI -> CPU
```

If NNAPI session creation fails and E1 falls back to CPU, E3 records:

```
requested_provider = NNAPI
actual_provider = CPU
fallback_used = true
```

A fallback measurement is never ranked ahead of a successful non-fallback
provider merely because its measured latency is lower.

For the S21 FE Exynos path the normal candidates are:

```
NNAPI
XNNPACK
CPU
```

QNN is recorded as unavailable unless the device is Qualcomm and an explicit
QNN backend is packaged.

## Integer-only identity evidence

Receipt identity must be reproducible in both Kotlin/JVM and Python.

Floating-point JSON formatting can differ across runtimes, so all timing fields
used in E3 identity are integers:

```
session_create_ns
latency_p50_ns
latency_p95_ns
latency_min_ns
latency_max_ns
tokens_per_second_milli_p50
```

`tokens_per_second_milli_p50` means throughput multiplied by 1000.

Example:

```
4,000,000
```

means:

```
4000.000 tokens / second
```

This keeps the receipt hash portable without reducing raw timer resolution.

## Percentiles

E3 uses deterministic nearest-rank percentiles.

For N sorted latency samples:

```
rank = ceil(q * N)
```

and selects the corresponding 1-based sample.

This avoids interpolation floats in identity evidence.

## Receipt

Schema:

```
VN97R2E3PROFILE1
```

A receipt binds:

- R2-E2 `bundle_id`;
- architecture fingerprint;
- fast/deep profile;
- device SDK/core/memory/thermal/SoC evidence;
- per-graph ranked provider measurements;
- provider failures;
- `device_measured=true`;
- `synthetic=false`.

The receipt ID is SHA-256 over canonical JSON with prefix:

```
VN97R2E3PROFILE1\0
```

## Python verification

After copying the receipt from Android:

```bash
vn97-r2-onnx-profile verify \
  --bundle-dir /data/vn97-onnx \
  --receipt /data/s21fe-profile.json
```

Status:

```bash
vn97-r2-onnx-profile status \
  --bundle-dir /data/vn97-onnx \
  --receipt /data/s21fe-profile.json
```

The verifier requires:

- valid E3 receipt identity;
- exact E2 bundle ID;
- exact architecture fingerprint;
- exact fast/deep profile;
- real-device flag;
- non-synthetic flag.

## Recommended profiling protocol on S21 FE

For each graph in the E2 bundle:

```
step.onnx
chunk-8.onnx
chunk-16.onnx
chunk-32.onnx
chunk-64.onnx
chunk-128.onnx
```

only profile graphs actually present in that bundle.

Recommended initial protocol:

```
warmup = 3
steady = 12
batch = 1
```

Run the device close to a normal operating thermal state.

Do not compare NNAPI measured on a hot device against XNNPACK measured after a
long cooldown as if that were a provider-only difference.

The receipt records thermal before/after so E4 can reject or down-weight
measurements taken under strong thermal drift.

## S21 FE interpretation

E3 does not assume NNAPI is always faster.

For an S21 FE Exynos 2100 device, E3 may find, for example:

- XNNPACK best for recurrent step latency;
- NNAPI best for one or more fixed chunk sizes;
- CPU fallback for unsupported graph/operator combinations.

Those are hypotheses until measured on the real device.

The final E4 policy must be derived from the actual receipt.

## Partitioning and fallback evidence

The Java ONNX Runtime surface does not expose a stable, portable,
operator-by-operator partition map suitable for the canonical VN97 contract.

E3 therefore records the evidence that can be made robust across devices:

- provider requested;
- provider/session creation success or failure;
- actual primary provider selected by the VN97 session factory;
- fallback use;
- end-to-end graph latency and throughput.

Provider-internal operator partition diagnostics may be collected as optional
debug evidence later, but they are not used as canonical performance identity.

This avoids treating provider-specific debug formats as a stable VN97 format.

## What E3 does not prove

A CI PASS proves only that:

- profiling contracts compile;
- receipt/ranking logic is correct;
- Android code integrates with the pinned ONNX Runtime API.

It does not prove:

- NNAPI speed on an S21 FE;
- XNNPACK speed on an S21 FE;
- final chunk selection;
- production 1B mobile performance.

Those require a real E2 bundle from a validated checkpoint and real device
profiling.

## Next milestone

R2-E4 consumes a verified E3 receipt and builds the adaptive runtime policy:

- graph-specific provider choice;
- chunk-size selection;
- latency feedback;
- thermal hysteresis;
- memory-pressure adaptation;
- safe provider failure quarantine;
- persisted device tuning profile.

E4 must not hard-code NNAPI or XNNPACK as universally fastest.
