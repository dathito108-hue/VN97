# VN97 R2-G0.4 — Explicit-state Mamba-2 ONNX lowering

R2-G0.4 lowers the preserved Mamba-2/VN97 G0 core into a mobile-oriented ONNX
**step graph** while keeping recurrent state explicit and externally owned by the
runtime.

This milestone is intentionally correctness-first. It does not yet claim the
parallel SSD prefill optimization planned for G0.5.

## Graph contract

The step graph has exactly three inputs:

- `input_ids` — one token per batch;
- `conv_state` — causal convolution state;
- `ssm_state` — Mamba-2 SSD recurrent state.

It returns:

- `logits`;
- `next_conv_state`;
- `next_ssm_state`.

For the locked Mamba-2 2.7B source the state geometry is:

- convolution state: `[64, batch, 5376, 4]`;
- SSD state: `[64, batch, 80, 64, 128]`.

The graph uses the same preserved weights and does not quantize, blend, resize,
or reinitialize the inherited core.

## Recurrent-state mobile budget

For batch 1 the explicit state contains:

- conv state: 1,376,256 elements;
- SSD state: 41,943,040 elements;
- total: 43,319,296 elements.

That is approximately:

- 86,638,592 bytes at FP16/BF16;
- 173,277,184 bytes at FP32.

This is only recurrent state. It excludes model weights, ORT graph metadata,
activations, temporary buffers, VN97MEM1, Android runtime memory and the 3D/UI
stack. Therefore S21 FE viability must be measured rather than inferred from
parameter count alone.

## External ONNX data

A 2.7B graph cannot safely be treated as one ordinary protobuf file. G0.4 uses
PyTorch ONNX external-data export when materializing the real graph. The bundle
manifest hashes every graph-related file, including external data files.

The graph is bound back to the exact G0.3 capsule by:

- capsule ID;
- capsule manifest SHA-256;
- inherited source weight SHA-256;
- architecture/state contract;
- ONNX graph/external-data file SHA-256 values.

## Numerical path

The ONNX module follows the same recurrent equations already locked in G0.2:

`embedding -> residual/RMSNorm -> in_proj -> causal conv state -> SSD update -> D skip -> gated RMSNorm -> out_proj -> final RMSNorm -> tied LM head`.

The state is carried between graph invocations rather than hidden inside an
ONNX session.

## Why step first

G0.4 exports the exact recurrent step before parallelizing prefill. This gives a
small semantic surface for parity:

`VN97 native step == ONNX Runtime step`

Only after that invariant is proven do we optimize long prompt processing.
G0.5 will add chunk/parallel SSD prefill while requiring the same final recurrent
state and logits as repeated G0.4 steps.

A fixed-size exact sequential-unroll adapter exists for development/reference,
but G0.4 does not label it parallel prefill and does not package it as the
production graph.

## Production gates

A G0.4 bundle explicitly carries:

- `same_weights_semantics=true`;
- `quantization_used=false`;
- `parallel_prefill_ready=false`;
- `production_activation_authorized=false`.

Android production switching remains blocked until:

1. real G0.3 capsule materialization;
2. real official Mamba-2 -> VN97 parity;
3. real 2.7B G0.4 ONNX export;
4. VN97-native -> ORT logits/state parity on the real graph;
5. G0.5 parallel/chunk prefill parity;
6. physical Galaxy S21 FE RAM, latency and thermal qualification.

## CLI

Export from a verified real G0.3 capsule:

    vn97-r2-mamba2-g04 export-step \
      --capsule-root /path/to/vn97-g03-capsule \
      --output-dir /path/to/vn97-g04-onnx

Verify graph files and manifest binding:

    vn97-r2-mamba2-g04 verify \
      --bundle-dir /path/to/vn97-g04-onnx

Run CPU ONNX Runtime step parity against the same VN97 capsule:

    vn97-r2-mamba2-g04 parity \
      --capsule-root /path/to/vn97-g03-capsule \
      --bundle-dir /path/to/vn97-g04-onnx

The real 2.7B export is not run automatically by ordinary push CI.
