# VN97 R2-G0.2 — Real Mamba-2 2.7B transfer and exact parity

R2-G0.2 turns the G0 architecture contract into a real-source ingestion and
parity path. It still does not promote a 2.7B checkpoint into Android
production by itself.

## Pinned source

Model:

- repository: `state-spaces/mamba2-2.7b`
- model-release revision:
  `99b226cc377d131cccc610ed4346db564f381f1e`
- official `pytorch_model.bin` SHA-256:
  `254d89bfef6dd9f44c8e3f54e0730af3f10c482969117cb37fb33486348b00be`
- official weight size: `5,405,424,282` bytes
- license: Apache-2.0

Tokenizer lineage:

- `EleutherAI/gpt-neox-20b`
- tokenizer revision:
  `364ae95407723fadd1d47b023c1efb92a4d891c3`

Parity oracle implementation:

- `state-spaces/mamba`
- pinned oracle commit:
  `e9594ce1c732d97440f0332fdc43170a2294dbfa`

The oracle commit is evidence provenance. VN97 Android does not depend on that
repository at runtime.

## Source acquisition

The resumable helper:

    bash tools/r2_g02_fetch_mamba2.sh /path/to/mamba2-source

downloads the exact model revision plus the pinned GPT-NeoX tokenizer files,
checks the 5.4 GB weight size and SHA-256, and writes a local SHA256SUMS record.

The helper requires at least 12 GiB free space before source download. A full
conversion additionally needs space for the VN97 G0 checkpoint, so production
conversion should run on storage with substantially more headroom than the
minimum fetch gate.

No model download occurs in GitHub CI.

## Memory-aware source loading

The official source is one monolithic PyTorch checkpoint rather than a shard
set. R2-G0.2 therefore uses:

    torch.load(..., map_location="cpu", weights_only=True, mmap=True)

The mapped tensors remain CPU file-backed while the converter validates tensor
names/shapes and constructs the VN97 namespace mapping. The converter does not
clone or quantize the inherited tensors before checkpoint serialization.

The source receipt schema `VN97M2SOURCE1` binds:

- exact source revision;
- config SHA-256;
- exact weight SHA-256 and byte size;
- tensor count;
- 2,702,599,680 unique core parameters;
- tokenizer model/revision;
- mmap transport evidence.

## Native SSD reference

`mamba2_ssd_reference.py` is an auditable CPU implementation of the official
unfused one-token Mamba2 path:

    in_proj
      -> [z, xBC, dt]
      -> depthwise causal conv state update
      -> SiLU
      -> [x, B, C]
      -> dt softplus + diagonal A update
      -> SSD recurrent state update
      -> C readout + D skip
      -> gated group RMSNorm
      -> out_proj

For the locked 2.7B source:

- recurrent convolution state per layer: `[5376, 4]`;
- recurrent SSD state per layer: `[80, 64, 128]`.

The reference preserves the official `residual_in_fp32=true` block behavior
and final RMSNorm semantics.

This reference is the numerical specification for VN97-native lowering. It is
not an alternate production model.

## Exact parity gate

Production switching requires `VN97M2G02PARITY1` evidence.

At minimum source and VN97 traces must compare:

- `layer0.hidden`;
- `layer0.conv_state`;
- `layer0.ssm_state`;
- `final.logits`.

The parity CLI also hashes generated token sequences. A tensor-only match with
different greedy generation is rejected.

Example:

    vn97-r2-mamba2-parity \
      --source-trace source-trace.pt \
      --vn97-trace vn97-trace.pt \
      --token-probe probe.json \
      --source-generated source-generated.json \
      --vn97-generated vn97-generated.json \
      --source-weight-sha256 254d89bf... \
      --vn97-checkpoint-sha256 <sha256> \
      --output parity.vn97m2g02.json

The production gate is fail-closed: any missing required trace, non-finite
tensor, tolerance failure, or generated-token hash mismatch blocks parity.

## Commands

Config-only structural assessment, no 2.7B allocation:

    vn97-r2-mamba2-transfer assess \
      --source-root /path/to/mamba2-source

Full real-source integrity verification:

    vn97-r2-mamba2-transfer verify-source \
      --source-root /path/to/mamba2-source \
      --receipt-output source.vn97m2source1.json

Real G0 conversion:

    vn97-r2-mamba2-transfer convert \
      --source-root /path/to/mamba2-source \
      --output vn97-g0-mamba2-2.7b.pt

The conversion output remains non-promotable until exact parity evidence exists.

## Still open after G0.2

G0.2 does not claim:

1. the real 5.4 GB checkpoint has been downloaded in this repository;
2. a real 2.7B VN97 checkpoint artifact has already been created;
3. source/VN97 parity has passed on the real weights;
4. ONNX export of Mamba-2 SSD states has passed;
5. the Galaxy S21 FE can run the dense 2.7B model within acceptable RAM,
   latency or thermal limits.

Those remain evidence gates, not assumptions.
