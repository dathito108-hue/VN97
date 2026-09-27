# VN97-R2E2 ONNX Explicit-State Step/Chunk Export

## Purpose

R2-E2 turns one VN97-R2 checkpoint into ONNX Runtime graphs that expose the
recurrent state explicitly.

E2 does not create another AI model.

All graphs are views of the same VN97 checkpoint and use the same weights and
state equations.

The Android execution chain is:

```
VN97 checkpoint
-> E2 ONNX explicit-state bundle
-> E1 adaptive scheduler
-> selected step/chunk graph
-> ONNX Runtime
-> NNAPI / XNNPACK / CPU
```

QNN remains optional only on Qualcomm devices with an explicitly packaged QNN
backend.

## Why explicit recurrent state

The R2 recurrent state is constant with respect to context length.

Each active layer carries:

```
conv_state [batch, d_inner, d_conv - 1]
ssm_state  [batch, d_inner, d_state]
```

E2 stacks these into two ONNX tensors:

```
conv_state [active_layers, batch, d_inner, d_conv - 1]
ssm_state  [active_layers, batch, d_inner, d_state]
```

Every graph consumes the previous state and returns:

```
next_conv_state
next_ssm_state
```

The Android runtime therefore does not need to replay the full prompt for each
new token.

## Graph types

Every bundle contains:

```
step.onnx
```

for one recurrent token.

A bundle also contains a selected subset of fixed chunk graphs:

```
chunk-8.onnx
chunk-16.onnx
chunk-32.onnx
chunk-64.onnx
chunk-128.onnx
```

The supported fixed sizes are:

```
8, 16, 32, 64, 128
```

Only the sizes explicitly selected during export are materialized.

This is deliberate.

Each ordinary ONNX file currently contains its own initializers, so blindly
shipping every chunk graph would duplicate checkpoint weights and enlarge the
APK. E3 profiling will determine the smallest useful graph set for the target
device.

## Adaptive scheduler compatibility

E1 is allowed to select arbitrary effective chunk boundaries.

E2 maps those boundaries deterministically onto the graphs actually present in
the bundle.

Example with only `chunk-32.onnx`:

```
scheduled 512
-> 16 x chunk-32.onnx
```

Example with `chunk-8.onnx`:

```
scheduled 12
-> chunk-8.onnx
-> step.onnx
-> step.onnx
-> step.onnx
-> step.onnx
```

This means VN97 can retain variable sequential/parallel scheduling without
requiring one physical ONNX model file for every possible length.

## Fixed sequence dimensions

Chunk graph sequence length is intentionally fixed.

Only the batch axis is dynamic.

This gives Android execution providers a much simpler shape contract than one
large fully dynamic graph and is intended to improve NNAPI/XNNPACK compilation
and caching behavior.

E3 will measure whether the chosen fixed chunk strategy is actually faster on
the S21 FE before it becomes the final APK policy.

## Step I/O

```
input_ids       int64 [batch]
conv_state    float32 [layers, batch, inner, conv_width]
ssm_state     float32 [layers, batch, inner, state]
```

Outputs:

```
logits          float32 [batch, vocab]
next_conv_state float32 [layers, batch, inner, conv_width]
next_ssm_state  float32 [layers, batch, inner, state]
```

## Chunk I/O

For `chunk-32.onnx`:

```
input_ids       int64 [batch, 32]
conv_state    float32 [layers, batch, inner, conv_width]
ssm_state     float32 [layers, batch, inner, state]
```

Outputs:

```
logits          float32 [batch, 32, vocab]
next_conv_state float32 [layers, batch, inner, conv_width]
next_ssm_state  float32 [layers, batch, inner, state]
```

All chunk graphs use the exact same state layout.

## Fast and deep profiles

E2 can export either:

```
--profile fast
--profile deep
```

Both are views of the same checkpoint weights.

The profile only locks the active layer count into that graph bundle.

Switching fast/deep inside an already active recurrent state remains forbidden,
because state from one active depth is not silently reinterpreted as another
depth.

Start a new state when changing profile.

## Manifest

Each bundle contains:

```
manifest.vn97onnx1.json
```

with schema:

```
VN97R2ONNX1
```

The bundle ID binds:

- architecture ID and fingerprint;
- full R2 config;
- fast/deep profile;
- active layer count;
- ONNX opset;
- checkpoint SHA-256 and stage when exported from a checkpoint;
- explicit state shape contract;
- selected fixed chunk sizes;
- every ONNX graph filename, SHA-256 and byte count;
- graph input/output names;
- the no-quantization/no-production-lowering state.

Graph tampering or manifest-contract tampering fails verification.

## Export from a checkpoint

Install the ONNX validation extras:

```bash
python -m pip install -e ".[onnx]"
```

Then:

```bash
vn97-r2-onnx-export export \
  --checkpoint /data/model.r2.pt \
  --output-dir /data/model-onnx \
  --profile deep \
  --chunks 32 \
  --example-batch-size 1
```

For a device experiment that needs two chunk sizes:

```bash
--chunks 16,64
```

## Verify bundle identity

```bash
vn97-r2-onnx-export verify \
  --bundle-dir /data/model-onnx
```

## CPU ORT parity validation

Step graph:

```bash
vn97-r2-onnx-export parity \
  --checkpoint /data/model.r2.pt \
  --bundle-dir /data/model-onnx \
  --profile deep \
  --batch-size 1
```

Chunk graph:

```bash
vn97-r2-onnx-export parity \
  --checkpoint /data/model.r2.pt \
  --bundle-dir /data/model-onnx \
  --profile deep \
  --chunk-size 32 \
  --batch-size 1
```

The parity gate compares:

- logits;
- next convolution state;
- next SSM state.

The state comparison is important: a graph that produces close logits but
incorrect recurrent state is not valid for streaming inference.

## Numerical oracle

PyTorch VN97-R2 remains the numerical oracle in E2.

E2 CI exports a smoke checkpoint and executes the exported ONNX models through
ONNX Runtime CPU.

It validates:

```
PyTorch chunk
~= ORT chunk

PyTorch state after chunk
~= ORT state after chunk

PyTorch next step
~= ORT next step using the ORT-returned chunk state
```

This proves graph-to-graph recurrent state carry, not only independent graph
outputs.

## Production checkpoint truth

No production 1B ONNX bundle exists yet.

The current project has not completed the production dense 1B training run.

E2 therefore validates the exporter and graph contract on the small R2 smoke
model. The production ONNX bundle must later be exported from the actual
validated production checkpoint, and its manifest must bind that checkpoint
SHA-256.

## Current precision

E2 exports FP32 dense graphs.

It does not perform:

- QAT;
- INT4;
- ternary lowering;
- FP16 weight conversion;
- operator approximation.

Those remain blocked until dense training and fresh model validation succeed.

Provider-specific precision optimization belongs to later mobile lowering.

## S21 FE relevance

For the S21 FE path, E2 is designed to make E3 profiling practical:

- `step.onnx` measures true one-token recurrent latency;
- selected fixed chunk graphs measure prompt/prefill throughput;
- state tensors make context replay unnecessary;
- graph count can be minimized to control APK/storage cost;
- fixed shapes let NNAPI and XNNPACK compile/cache simpler graphs.

E3 will measure real provider partitioning and latency before a final chunk set
is selected.

## Next milestone

R2-E3 will add:

- Android ONNX graph loading from an E2 manifest;
- state-buffer reuse instead of reallocating state every call;
- NNAPI/XNNPACK provider-specific timing;
- warmup vs steady-state latency;
- step latency;
- chunk throughput;
- provider fallback evidence;
- device/thermal/memory evidence;
- an S21 FE profiling receipt.

Only measured device evidence will decide the final S21 FE chunk graph set.
