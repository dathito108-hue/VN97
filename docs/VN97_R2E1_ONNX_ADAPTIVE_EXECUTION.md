# VN97-R2E1 ONNX Adaptive Sequential-Parallel Execution Fabric

## Purpose

R2-E1 locks the Android execution fabric before production-scale GPU training.

It does **not** replace the VN97-R2 model, change its equations, introduce a
second model, or quantize production weights.

The execution stack is:

```
single VN97-R2 weights + recurrent state
        |
        v
adaptive schedule
  sequential / hybrid / parallel
        |
        v
ONNX Runtime Android
        |
        +-> NNAPI primary on broadly supported Android hardware
        +-> XNNPACK fallback/CPU-optimized path
        +-> ORT CPU final fallback
        +-> QNN only when Qualcomm + explicit packaged QNN backend exist
```

Fast/deep cognition remains a separate decision over the same weights:

```
VN97R2ExecutionPolicy
    -> active layer count

VN97AdaptiveExecutionScheduler
    -> execution schedule + ORT provider
```

The two decisions do not select another AI model.

## Why E1 comes before large training

R2-E1 freezes **execution semantics**, not compressed weights.

The same selective-SSM equations already support:

- recurrent token step;
- full associative parallel scan;
- exact state carry between scan chunks.

That makes it possible to prove now that changing execution schedule does not
change VN97 semantics. Dense production training can still proceed later from
the same architecture contract.

QAT, INT4 and ternary production lowering remain blocked until dense training
and fresh validation pass.

## Three execution modes

### Sequential

Selected for:

- realtime token/event streams;
- very short sequences;
- high recurrent-state dependency;
- game/device loops where minimum per-step latency matters.

Schedule:

```
1 -> 1 -> 1 -> ... token
```

Each step consumes and returns the same constant-size VN97 recurrent state.

### Parallel

Selected for:

- long available-at-once prefill;
- comfortable memory;
- low state-dependency workloads;
- acceptable thermal state.

The complete sequence uses the existing associative selective-SSM scan.

### Hybrid

Default mobile prefill path.

A deterministic ramped schedule starts with a small chunk and grows toward the
selected ceiling:

```
8 -> 16 -> 32 -> 64 -> 64 -> ...
```

Each chunk uses the parallel scan internally. The final recurrent state of one
chunk is the exact initial state of the next chunk.

Chunk boundaries therefore change execution schedule only.

## Adaptive inputs

The E1 scheduler uses:

- sequence length;
- deadline;
- realtime flag;
- state-dependency score;
- free memory;
- Android thermal status;
- CPU logical-core count;
- prior provider failures;
- moving latency feedback.

Current deterministic thresholds are intentionally conservative and auditable.
They are seeds for E3/E5 measured tuning, not claims that one fixed threshold
is optimal on every phone.

## S21 FE baseline

The mandatory production route does not assume Qualcomm.

For a Galaxy S21 FE-class Exynos device the intended priority is:

```
NNAPI -> XNNPACK -> ORT CPU
```

QNN is never required.

On a Snapdragon device, QNN can become the first candidate only when:

1. the device fingerprint is Qualcomm/Snapdragon;
2. the app explicitly reports QNN available;
3. an explicit packaged QNN backend path is present.

This prevents an Exynos build from depending on Qualcomm libraries.

## ORT session policy

The Android runtime uses `onnxruntime-android`.

Each session chooses **one primary accelerator provider plus ORT CPU fallback**.
If session creation fails, E1 tries the next provider candidate.

This avoids deliberately stacking many accelerator EPs into one session before
E3 profiling proves that graph partitioning is beneficial.

### NNAPI

E1 registers NNAPI with `CPU_DISABLED`.

Unsupported nodes therefore fall back to ORT's own CPU kernels rather than
being executed by the NNAPI reference CPU device.

FP16 relaxation is **not** enabled in E1. Numerical accuracy is preserved until
measured device profiling and later post-validation precision lowering.

### XNNPACK

E1 disables ORT intra-op spinning, sets ORT intra-op threads to 1, and gives
XNNPACK its own bounded thread pool.

The initial upper bound is four threads and is reduced under thermal pressure.

### CPU

ORT CPU remains the mandatory final fallback.

### QNN

QNN is optional and fail-closed.

E1 will not select it without an explicit backend path. Production packaging
of QNN libraries is later device-specific work; it is not a dependency of the
S21 FE Exynos path.

## Android device probe

`OrtDeviceProbe` derives:

- Android API level;
- available memory;
- logical cores;
- thermal status;
- hardware string;
- SoC manufacturer/model where the API exposes them.

This feeds the same deterministic scheduling contract represented by the Python
reference scheduler.

## Numerical oracle

`execute_reference_schedule()` runs the same PyTorch R2 model through:

- token-by-token recurrent step;
- variable hybrid chunks;
- full parallel scan.

E1 tests compare:

- logits;
- causal-convolution state;
- selective-SSM state;
- active fast/deep layer count.

The tolerance gate is deliberately tight because all paths implement the same
equations.

E2 ONNX export must match this oracle before Android ORT output can be
considered canonical.

## Android ONNX Runtime version

R2-E1 pins:

```
com.microsoft.onnxruntime:onnxruntime-android:1.30.0
```

The runtime module also exports a consumer R8 keep rule for
`ai.onnxruntime.**`.

A custom/minimal ORT build remains a later APK-size optimization. E1 first
establishes correctness and provider routing with the official Android AAR.

## Roadmap after E1

```
R2-E1  ONNX adaptive execution contract
  ->
R2-E2  explicit recurrent-state + chunk ONNX export
  ->
R2-E3  NNAPI/XNNPACK graph coverage + latency/RAM/thermal profiling
  ->
R2-E4  measured dynamic scheduler feedback/autotuning policy
  ->
R2-E5  Galaxy S21 FE on-device benchmark + profile sealing
```

After dense training and fresh validation:

```
QAT / precision sensitivity
-> INT8/INT4/ternary where proven safe
-> custom minimal ORT/operator build
-> final Android production lowering
```

The order intentionally keeps execution-path engineering separate from
unvalidated weight compression.
