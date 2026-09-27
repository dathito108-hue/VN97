# VN97 R2-G0.6 — Dynamic-sequence ONNX + Android Mamba-2 executor

R2-G0.6 is the Android integration layer for the Mamba-2-derived VN97 core.
It supersedes the fixed-max G0.5 production idea before that design reaches the
phone.

## Why G0.5 is not the Android contract

G0.5 proved the one-chunk SSD factorization and single-weight semantics, but its
fixed maximum input tensor means a decode call with only one valid token can
still execute operators sized for the full maximum chunk.

That is acceptable as a mathematical parity milestone, but it is not the right
mobile execution contract.

G0.6 therefore keeps batch fixed at one while making the **sequence dimension
truly dynamic**:

- decode input shape: `[1, 1]`;
- prompt tail: `[1, N]`;
- full prefill chunk: `[1, 32]` by default.

One graph and one external-data weight set are still used, but work now scales
with the actual sequence length.

## Dynamic ONNX graph

Schema: `VN97M2G06ONNX1`.

Inputs:

- `input_ids`;
- `conv_state`;
- `ssm_state`.

Outputs:

- FP32 `logits`;
- `next_conv_state` in the inherited state dtype;
- `next_ssm_state` in the inherited state dtype.

The inherited recurrent state may be float32, float16 or bfloat16. Logits are
cast to float32 only at the graph boundary so Android sampling has one stable
representation. The cast does not alter the already-computed source-precision
logit values.

The dynamic graph must match repeated exact G0.4 steps for sequence lengths 1,
prompt tails, and the maximum chunk.

## Android package contract

`Mamba2OrtRuntimePackage` parses the G0.6 manifest and fail-closes on:

- wrong Mamba-2 2.7B geometry;
- wrong batch or dynamic-sequence bounds;
- unexpected graph/input/output contract;
- quantization or disabled same-weight semantics;
- missing/modified ONNX or external-data files;
- recurrent-state shape/byte-budget mismatch;
- any production self-authorization flag.

The graph is opened by file path so ONNX Runtime can resolve external tensor
data beside the protobuf without loading a multi-gigabyte model into a Java
byte array.

## Android recurrent state

`VN97Mamba2OrtExecutor` uses two direct recurrent-state slots. State dtype is
bound to the ONNX manifest and mapped to ONNX Runtime Java tensor types:

- float32 -> 4-byte direct buffer;
- float16 -> 2-byte direct buffer;
- bfloat16 -> 2-byte direct buffer.

The next slot is written by ORT and becomes current only after:

1. the provider call returns successfully;
2. final logits are finite.

A provider/session failure therefore cannot commit a partial recurrent state.

## One session, variable sequence length

The same `recurrent-dynamic.onnx` session is used for:

- `step(token)` with sequence length 1;
- `prefill(tokens)` split into chunks no larger than the manifest maximum.

The Java input/logit buffers are recreated only when sequence length changes.
Repeated decode stays at length 1 and reuses its buffers; repeated full prefill
chunks reuse the full-chunk buffers.

Provider selection still uses the existing Android device/thermal/memory probe
and provider ordering infrastructure. QNN remains optional; NNAPI, XNNPACK and
CPU remain bounded fallbacks according to device support.

## Memory truth boundary

For the locked 2.7B geometry, recurrent state alone contains 43,319,296 values
per batch. Two state slots are intentionally maintained for non-replay commit
semantics. That means the executor can require roughly 173 MB just for double
buffered FP16/BF16 state, or roughly 347 MB at FP32, before weights, activations,
ORT workspace, VN97MEM1, Android/UI and the 3D assistant are counted.

This is why physical S21 FE measurement remains mandatory.

## Production boundary

G0.6 does not activate the new graph in the assistant yet. The next bridge
milestone must bind the signed tokenizer/checkpoint identity to the exact G0.6
manifest/capsule and replace the old F1 executor path without fallback.

The real production chain remains:

real G0.3 capsule -> real source/VN97 parity -> real G0.6 ONNX export/parity ->
Android identity-bound cognition bridge -> S21 FE provider/RAM/latency/thermal
qualification -> release packaging.
