# VN97 R2-G0.4 — Mamba-2 explicit-state ONNX lowering

R2-G0.4 defines the ONNX execution contract for the exact G0 Mamba-2 2.7B
intelligence seed.

It does not reuse the older VN97-R2 recurrent state layout. Mamba-2 has a
different state topology and must be lowered faithfully.

## Locked production state

For the official 2.7B source:

- layers: 64
- convolution state per layer: [batch, 5376, 4]
- SSD state per layer: [batch, 80, 64, 128]
- token dtype: int64
- recurrent state dtype: float32

The stacked graph interface is therefore:

    input_ids
    conv_state [64, batch, 5376, 4]
    ssm_state  [64, batch, 80, 64, 128]

and returns:

    logits
    next_conv_state
    next_ssm_state

State is explicit and is carried exactly between graph invocations.

## Same-weight semantics

VN97M2ONNX1 requires:

- source weight SHA bound to the pinned Mamba-2 2.7B payload;
- capsule id + capsule manifest SHA;
- one set of weights shared by step/chunk semantics;
- no quantization at G0;
- no augmentation effect at G0;
- no source Mamba runtime dependency;
- production activation blocked until parity evidence passes.

Multi-Timescale State, State Highway and VN97MEM1 injection remain zero-impact
at G0. Their trained G1 graph integration is a later stage and must not alter
the G0 parity baseline.

## Step and chunk graphs

The canonical graph family is:

    step.onnx
    chunk-8.onnx
    chunk-16.onnx
    chunk-32.onnx
    chunk-64.onnx

Only the chunk sizes actually exported are listed in a bundle manifest.

The final production bundle must contain at least one step graph and at least
one chunk graph.

## Tiny oracle export

CI must not allocate a 2.7B model. R2-G0.4 therefore contains a shape-reduced
Mamba-2 mixer adapter that executes the same recurrent equations and explicit
conv/SSD state topology.

The tiny graph is used only to prove:

- PyTorch -> ONNX export mechanics;
- explicit recurrent inputs/outputs;
- dynamic batch axis;
- state carry;
- ONNX Runtime numerical parity.

It is not a production model and is never promoted as intelligence.

## Real lowering path

When a real G0.3 capsule exists, the production sequence is:

    load G0.3 capsule
      -> verify full 5.4 GB source weight SHA
      -> mmap exact inherited tensors
      -> construct VN97-native Mamba-2 graph
      -> export step + selected chunk graphs
      -> bind VN97M2ONNX1 manifest to capsule identity
      -> run source vs VN97 vs ORT parity
      -> only then allow Android profiling

The graph exporter must not silently switch to the old VN97-R2 SSM equations.

## Mobile boundary

A dense 2.7B ONNX graph is not assumed to fit the Galaxy S21 FE. G0.4 is about
semantic correctness first.

Only after dense source/VN97/ORT parity succeeds may the project investigate:

- graph partitioning/external-data layout;
- provider compatibility;
- INT8/INT4 or other quantization;
- RAM reductions;
- S21 FE thermal and latency behavior.

No compression method is allowed to redefine the baseline before exact dense
parity is proven.
