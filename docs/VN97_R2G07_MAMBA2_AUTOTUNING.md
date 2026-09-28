# VN97 R2-G0.7 — Mamba-2 provider profiling and device-bound autotuning

R2-G0.7 profiles the canonical G0.6 Android ONNX Runtime path without changing
the inherited Mamba-2 weights or recurrent equations.

## Why G0.7 profiles valid_length

G0.5/G0.6 use one recurrent graph for both prompt prefill and decode:

- valid_length=1: recurrent decode;
- valid_length=8/16/32: progressively larger parallel SSD prefill workloads.

Profiling only by ONNX filename would hide these cost differences because every
workload uses the same recurrent-N.onnx file. G0.7 therefore records provider
measurements per valid_length.

For a max chunk of 32, the required measured workloads are exactly:

    1, 8, 16, 32

For max chunk 16 they are 1, 8, 16; for max chunk 8 they are 1, 8.

## Android profiling receipt

VN97Mamba2OrtProviderProfiler runs the real G0.6 graph on the current Android
device using the existing fail-closed provider session factory.

It attempts available providers from:

    QNN
    NNAPI
    XNNPACK
    CPU

Each accelerator is opened as an isolated provider session with hidden CPU
fallback disabled. CPU is measured directly and is required as a baseline for
every workload.

The profiler records:

- requested and actual provider;
- whether fallback occurred;
- session creation latency;
- steady-state p50/p95 latency;
- available memory before/after;
- thermal status before/after;
- warmup and steady iteration counts.

The signed-by-content receipt schema is VN97M2G07PROFILE1. It binds the G0.6
runtime ID, graph filename, max chunk size and exact device/OS identity.

The receipt explicitly states:

    device_measured=true
    synthetic=false
    same_weights_semantics=true
    production_activation_authorized=false

A synthetic or incomplete receipt is rejected by the tuning compiler.

## Tuning compiler

The host-side command:

    vn97-r2-mamba2-g07 compile --profile profile.json

creates tuning.vn97m2g07.json with schema VN97M2G07TUNE1.

Only direct measurements where requested_provider == actual_provider and
fallback_used=false are eligible for ranking. Providers are ordered by measured
p95 latency, then p50 latency. CPU is always retained as the explicit final
fallback.

The tuning profile is bound to:

- G0.6 runtime ID;
- G0.7 profile receipt ID;
- recurrent graph filename;
- max chunk size;
- Android SDK level;
- logical CPU count;
- hardware/SOC identity.

## Runtime selection

VN97Mamba2OrtExecutor accepts an optional G0.7 tuning file.

If present, the file must match both the current runtime ID and current device
identity. Any mismatch fails closed during open.

For prompt tails that were not directly profiled, the executor uses the next
larger measured workload policy:

    valid_length 2..8   -> profile 8
    valid_length 9..16  -> profile 16
    valid_length 17..32 -> profile 32

Decode valid_length=1 always uses its own measured policy.

The profile also controls XNNPACK thread count for normal/hot/critical thermal
bands. G0.7 does not invent provider rankings from thermal heuristics; rankings
come from device measurements.

If no G0.7 tuning file is supplied, the executor stays on the conservative G0.6
adaptive scheduler. This is a policy fallback only, not an inference-backend
fallback, and it is not treated as device-qualified tuning.

## Memory truth

The dense G0 recurrent state is still float32. Profiling needs separate input
and output state buffers so a failed provider run cannot corrupt the baseline.

This means the profiling harness itself can require hundreds of MB before model
weights and ONNX Runtime allocations. An out-of-memory result on a real device
is valid qualification evidence; G0.7 must not mask it.

## Production boundary

G0.7 builds the measurement and tuning machinery but does not claim S21 FE
numbers until the profiler has actually run on the physical phone.

Production activation remains blocked until:

1. real G0.3 capsule exists;
2. real G0.5 ONNX external-data graph exists;
3. source -> VN97 -> ORT numerical parity passes;
4. physical S21 FE G0.7 receipt is captured;
5. G0.7 tuning is compiled from that receipt;
6. RAM/latency/thermal qualification passes;
7. the cognition bridge is explicitly rebound to the Mamba-2 runtime.
