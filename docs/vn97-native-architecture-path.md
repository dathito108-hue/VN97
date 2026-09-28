# VN97: a measured path to its own mobile architecture

Status: experimental foundation, not a production replacement or a novelty claim.
Continue Code 1 → hardware-aware Code 2 → VN97. Keep one active foundation,
one canonical planner, sovereign memory and M6 authority. The current ONNX
lineage remains the baseline until a candidate passes promotion gates.

## Distinguish the changes

| Change | What it establishes | What it does not establish |
| --- | --- | --- |
| New packaging or model name | Inventory and distribution identity | New architecture or new intelligence |
| K2T residual ternary planes | A candidate weight representation | Different recurrent computation or preserved task quality |
| Adaptive-state experiment here | An explicit alternative recurrent update with an executable reference | A trained language model, unique invention, faster phone execution |
| Trained and evaluated replacement | Evidence for VN97-specific computation and learned behavior | Erasure of training/weight provenance or automatic AGI |

SSM, exact ZOH, gating, affine scans and multiple timescales are building blocks;
using them does not by itself establish originality. Preserve attribution and
license records for inherited weights/code. Mamba-derived weights must remain
identified as such even if transformed or used in a later learning stage.

## First candidate: input-gated fast and slow state in one cell

Implementation: `research/vn97_native/adaptive_state.py`.

For each state channel, the same drive u enters a fast state f and slow state s.
The input supplies duration Δ, slow-write gate w and slow-read gate r. Require
0 ≤ Δ ≤ Δmax, 0 ≤ w,r ≤ 1 and 0 < λslow < λfast.

```
fast elapsed = Δ
slow elapsed = Δ × w
for each bank with rate λ and elapsed τ:
    a = exp(-λτ)
    b = -expm1(-λτ) / λ × u
    h_next = a × h_previous + b
output = (1-r) × f_next + r × s_next
```

At w=0 the slow bank holds exactly. There is no periodic token counter, so the
same input stream has the same result across chunk boundaries. Rates are fixed
configuration values in this reference. A trainable implementation must learn
input projections for drive, duration and gates, and positive ordered rates;
none of those trained projections exist in this checkpoint.

Crucial constraint: selectors must depend on the current input (or a separately
scan-compatible causal input projection), not on this cell's previous hidden
state. Otherwise the precomputed affine scan below is invalid. Do not add
hidden-state routing while continuing to claim these parallel semantics.

Composition is `(a2,b2) ∘ (a1,b1) = (a2*a1, a2*b1+b2)`. The reference implements
both token recurrence and a tree-depth prefix scan. Its Python scan performs
O(T log T) work and is a mathematical oracle, not a mobile optimization. A
production kernel needs an efficient scan and bounded chunk workspace.

Two width-D banks require 2D recurrent values per cell. This **doubles** storage
relative to one bank of width D; any comparison must hold total state budget
fixed, such as splitting an existing budget across the two banks. The 512-byte
FP16 example at D=128 excludes all projections, weights, activations, scratch
buffers, layers, batch replication and runtime overhead. Closed gates do not
prove skipped hardware work or reduced energy.

## Development gates

1. **Reference completed here:** exact-ZOH oracle, sequential/scan parity from
   nonzero state, chunk continuation, closed-gate hold, zero/tiny duration,
   configuration-bound state and malformed-input rejection. Nine stdlib tests;
   no model download, GPU or training required.
2. **Trainable candidate:** tensor implementation and gradients checked against
   this oracle; integrate as a research cell in the existing VN97 backbone,
   preserving tokenizer, outer residual/norm path and planner. Establish fresh
   checkpoint schema and explicit architecture fingerprint. Existing Mamba
   recurrent state/checkpoints are not directly interchangeable.
3. **Controlled training:** establish a reproducible small baseline, fixed
   train/validation/test separation and equal parameter/state budgets. Compare
   one-bank vs two-bank, fixed vs input-gated slow updates, equal vs separated
   timescales. Train projections/cell before considering inherited-weight
   transfer or distillation. Do not assume transferring tensors transfers skills.
   Freeze evaluation rules before viewing test results.
4. **Behavior before compression:** evaluate held-out Vietnamese language,
   instruction following, structured tool plans, long-span retrieval and actual
   service tasks. Pin baseline commit, candidate fingerprint, dataset hashes,
   seeds and raw outputs. Do not promote based only on reconstruction error or
   matching a weak baseline. Set both absolute task requirements and regression
   tolerances from the deployment use cases before the experiment.
5. **Mobile lowering:** export the trained cell to ONNX, compare logits and
   recurrent states over multiple carried chunks, then evaluate ternary
   projections as a separate ablation. A fallback that silently executes another
   model is not allowed. Do not change the current production G0.6 geometry to
   make this research candidate pass its reader.
6. **Physical promotion:** measure first-token and p50/p95 token latency, peak
   PSS, sustained thermals, memory failures and task quality on S21 FE with matched
   conditions. Promote through existing controlled evaluation/rollback and M6
   paths only after all gates; keep source attribution and evidence intact.

This research track supports the mobile/revenue objective but is not revenue
itself. Existing service delivery and receipt verification should continue
without waiting for architecture research to finish. No prediction of an
8-hour trained-model completion or guaranteed revenue follows from this work.

Run: `python -m unittest discover -s research/vn97_native -p 'test_*.py' -v`.

## Trainable-cell checkpoint

`research/vn97_native/trainable_state.py` adds a differentiable tensor candidate:
input projection produces drive, bounded duration and read/write gates; softplus
parameterization learns ordered positive scalar fast/slow rates. The initial
rates have a small numerical floor. Per-channel rates, residual/norm integration
and language-model training remain later work.

The tensor paths support batched token execution, sequential execution and the
same tree-depth affine scan. State continuation retains gradients across chunks;
truncated training must explicitly call `state.detach()`. FP32/FP64 are supported
for this research implementation; FP16 lowering is not validated. Python finite
checks and O(T log T) scan workspace are unsuitable as a claimed mobile kernel.

Seven tensor tests cover an independent scalar oracle, scan/serial parameter
and input gradients, numerical gradcheck of every parameter, chunked gradients,
checkpoint tensor round-trip/token continuation, invalid inputs/state, and 120
optimizer steps on one fixed synthetic smoothing batch. The optimizer test is
not held-out evaluation, model training completion or language proficiency.

Run only the dependency-free reference tests with:
`python -m unittest discover -s research/vn97_native -p 'test_adaptive_state.py' -v`.
After installing CPU Torch, run tensor tests with:
`python -m unittest discover -s research/vn97_native -p 'test_trainable_state.py' -v`.
CI pins Torch 2.5.1 for reproducible CPU testing; no model weights or GPU required.

The state architecture tag covers input width and configuration, not parameter
values. Standard `state_dict` round-trip is a research test, not an authenticated
checkpoint format: never reuse live state after a weight update, and do not
connect this candidate to production loaders until checkpoint/version binding
and controlled promotion are implemented. The current mobile APK is unchanged.

## Small language-backbone integration

`language_candidate.py` wraps the actual repository `VN97R2Model` and replaces
its recurrent blocks with the adaptive-state candidate. It retains the existing
embedding, tied output, final normalization, layer loop and state carrier; each
candidate block keeps an RMSNorm/residual path. The experiment uses d_conv=1
(no convolution history) and two state values per inner channel. This is a
research adaptation, not weight-equivalent to the current SSM or Mamba lineage.
The wrapper has a distinct `VN97ASC-LM1` configuration and `VN97ASC-LMCP1`
checkpoint. Its byte vocabulary is experimental, not a replacement VN97TK1.

Live streams bind to the wrapper instance, architecture and ordinary PyTorch
parameter version counters. Optimizer/in-place parameter updates and loading
weights invalidate them; start a fresh stream. Unsupported `.data` manipulation
is not guarded. This guard is not a production cryptographic authority boundary.
Checkpoint files bind config/fingerprint and tensor SHA-256 and forbid production
activation. This is integrity checking, not publisher authentication. Use only
trusted research artifacts; weights are FP32 and no optimizer/resume state is
saved. The production capsule loader does not consume these checkpoints.

Four additional tests verify whole/token/chunk logits, language-loss gradients,
stale/foreign streams, checkpoint round-trip, tamper and activation rejection.
The existing broad training-package facade is bypassed by a private namespace
that imports the real repository modules, not mocks.

`evaluate_language_candidate.py` fixes seed 97, 80 Adam steps and all settings
before evaluation. It trains on 48 original synthetic Vietnamese CSV-task
records, evaluates eight validation and eight test records, and saves weights,
dataset, hashes and all split metrics. Full records are disjoint, **templates and
many byte substrings are shared**. Test metrics are computed after training;
they never choose a checkpoint or hyperparameter. An untrained copy is measured
on the same splits after the candidate has finished.

This is an end-to-end training/evaluation pipeline probe. Low next-byte loss on
this template cannot establish Vietnamese understanding, task execution,
architecture superiority, source-weight independence of a future larger model,
or revenue ability. Equal-budget ablations, multiple seeds and a genuinely
held-out task corpus are still required. A fixed candidate failing those later
gates must remain non-production regardless of this probe's loss.

Run `python research/vn97_native/evaluate_language_candidate.py --output NEW_DIR`.
CI preserves the tiny research checkpoint and evidence for seven days; exact
results must be read from that run's report rather than assumed from this plan.

## User-requested 10M CPU pilot

The bounded research configuration now admits d_model=512, d_inner=768 and five
layers: **9,979,914 trainable parameters**. `CandidateConfig.parameter_count()`
is tested against instantiated small models and the pilot checks the actual
10M count before optimization. The hard ceiling is 10.5M, not unlimited scaling.
Production geometry/loaders remain untouched.

`pilot_10m.py` uses 512 original synthetic training records over copy-ID, sum,
deduplication and request classification. Validation/test each contain 64 records
with different prompt templates. Full prompts are disjoint; common task rules,
substrings and some answers are shared. Byte vocabulary remains experimental.
Only answer bytes contribute to loss. Fixed seed97, batch4, AdamW 0.0003, up to
160 steps or 480 training seconds; no validation/test checkpoint selection.
Preflight requires 2.5GB available memory and 1GB free disk. The time guard is
checked after each completed step and excludes setup, evaluation and saving.

Evaluation includes answer-byte loss/accuracy per task and eight bounded greedy
free-generation examples with exact answer matching. Accuracy with provided
previous correct bytes must not be confused with successful autonomous answers.
Inspect raw generations, even if byte loss improves. This is a short learning
pilot, not completed pretraining, AGI, architecture superiority or mobile speed.

The workflow runs this heavier stage only on `vn97-10m-cpu-pilot` PR events;
ordinary future research PRs keep lightweight checks. Job timeout20min is a
ceiling, not a request to consume all that time. Do not manually rerun unchanged
training just to improve a score. Report completed steps and any time-budget stop.

Exact FP32 weights and provenance are saved and hash-checked through a reload.
The ZIP is split into 20MiB pieces solely for transport; concatenate in numbered
order and compare SHA256 with `bundle-sha256.json`. This is not quantization.
Optimizer/RNG are not saved: later learning may start from these weights but is
not a bit-exact resume of this optimizer run. Keep production activation false.
