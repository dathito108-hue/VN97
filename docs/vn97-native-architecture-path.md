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
