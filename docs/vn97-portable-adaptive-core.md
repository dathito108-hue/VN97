# VN97 adaptive core for PC and Android

This develops the existing fast/slow candidate, not another production model.
Current architecture identity/checkpoints stay unchanged. The R2 Android
assistant remains the active baseline until explicit promotion; the new core
has no automatic fallback, inventory activation or training in this change.

## One learned architecture, two kinds of adaptation

Each residual block has RMSNorm, learned input projection into drive/duration/
write/read, and two ordered positive learned rates. Fast state responds on Δ;
slow state responds on Δ × write. Exact-ZOH updates maintain stability for
positive rates. A learned read gate mixes the two states; a learned output
projection and residual return to the shared model width. Embedding, final
RMSNorm and tied language head remain from the existing candidate backbone.
Selectors depend on layer input, not that cell's previous hidden state, so the
update remains affine and associative scan is valid.

This combines established building blocks; neither originality, AGI quality nor
a speed/accuracy gain over R2/Mamba is established by implementation alone.
Gates are continuous values. A small gate does not mean the hardware skips work.

Execution adapts independently: the same weights and ALL layers can process
1/8/32 tokens per call. PC may use 32-token chunks; mobile may use 8 or 1. These
are policies, not device benchmarks or mandatory values. The caller can change
chunk size at boundaries and keep the exact same state layout. No layers are
silently omitted and no weights are switched when changing device budgets.
Parallel scan and recurrence are mathematically equivalent, but FP32 reduction
orders are checked with tolerances rather than claimed bit-identical.

## Portable contract VN97ASCORT1

- Inputs: `input_ids` INT64 [1,T], `fast_state` and `slow_state` FP32 [L,1,D].
- Outputs: `logits` FP32 [1,T,256] and next fast/slow states of identical shape.
- Fixed graph lengths T=1,8,32; opset18, standard ONNX operators.
- One deduplicated `weights.bin` shared by the three graph files.
- Manifest binds architecture/config, learned weight identity, graph inventory,
  file lengths/hashes, fixed state shape and research-only activation=false.
- Caller pins the manifest SHA-256. A checksum is identity, NOT publisher trust.
  These files are not accepted by the production signed model import flow.
- State binds that manifest SHA-256 and a token position. Foreign/invalid states
  are rejected. State migration across devices must preserve FP32 bytes and the
  same bundle. Persistent state serialization is not implemented here.

ONNX has Exp but not standard Expm1. The exporter lowers ZOH with a fourth-order
small-argument expansion below p=1e-3; otherwise it uses `(1-exp(-p))/rate`.
Tests compare against the original PyTorch cell, including near-zero duration.

## Implementations

`research/vn97_native/portable_core.py` exports existing candidate checkpoints:

```sh
python research/vn97_native/portable_core.py --checkpoint PATH_TO_CHECKPOINT --output NEW_BUNDLE_DIRECTORY
```

It never trains or changes the candidate's weights. Existing checkpoint loader
integrity checks still apply. `portable_runtime.py` runs these graphs on PC CPU
with ONNX Runtime and a one-session cache. `AdaptiveStateOrtCore.kt` is the
Android research executor for the same files. It verifies pinned manifest and
file identities and accepts caller-selected chunk sizes with bounded state.
It is not connected to the main chat, planner or activation menu yet.

For the existing 9,979,914-parameter candidate (5 layers, width512, state width768),
two FP32 state banks contain 7,680 values = 30,720 bytes at batch1. This excludes
weights, graph sessions, duplicated buffers, activations, outputs and application
memory. The weights alone are approximately38.1 MiB in FP32 before packaging;
export size must be measured. Do not call this the whole-app RAM footprint.

## Evidence and next gate

CI exports a SMALL randomly initialized candidate and runs real ONNX Runtime
CPU inference, comparing logits and terminal states against the original
PyTorch model for token/mobile/PC chunks and mixed-device budgets. This proves
numerical lowering and continuity on those cases, not trained language quality
or 10M-checkpoint parity. Android compilation is checked separately. No physical
S21 FE or ARM64 performance/thermal measurement is implied.

The next gate is export and held-out inference evaluation of the preserved10M
checkpoint (no training needed), then actual Android execution/parity, measured
latency/RAM/thermal behavior, and controlled promotion into the one canonical
assistant. Quantization and native acceleration must retain separate parity
and quality gates. Training remains paused until the user reauthorizes it.
