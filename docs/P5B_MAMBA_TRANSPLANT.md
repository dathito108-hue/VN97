# P5B — Mamba-130M → Native VN97 Intelligence Transplant

P5B tests whether a proven selective-SSM checkpoint can seed a larger native
VN97 model so that VN97 does not need to learn all low-level sequence dynamics
from scratch.

The pinned source is **state-spaces/mamba-130m-hf**, revision
`5708daa364c50b880e7bd92eab456e0d34492ee9`.

The source model is Apache-2.0 and is used only during conversion. The final
runtime remains one native VN97 model.

## Why this source is structurally useful

The pinned Mamba checkpoint has:

- hidden width 768;
- 24 selective-SSM layers;
- state width 16;
- input-dependent dt/B/C;
- RMSNorm;
- SiLU gating;
- diagonal A dynamics.

Those properties line up closely with the native VN97 selective SSM. The main
differences are Mamba's 2x inner expansion, depthwise causal convolution, D
skip, source tokenizer, and dense floating-point projections.

Therefore this is **not a lossless format conversion**. P5B is an intelligence
transplant followed by a short alignment/calibration stage.

## Frozen target

P5B constructs a native VN97 target with:

- d_model 768;
- 24 VN97 layers;
- d_state 16;
- rank-384 factorized VN97 embedding/head;
- native VN97 tokenizer;
- native VN97 ternary projection layers;
- normal VN97 deployment checkpoint/model-image formats.

This target is roughly 59M trainable parameters, but deployment projections are
packed ternary by the existing VN97 toolchain.

## Weight transplant

For each source layer, P5B selects 768 of Mamba's 1536 inner channels using a
deterministic energy score computed from in/out/x/dt/conv/D tensors.

Transferred or derived components:

- RMSNorm scales: direct;
- A_log: selected rows;
- in_proj signal/gate: selected rows;
- out_proj: selected columns;
- B/C: selected source x_proj columns;
- dt: composed `dt_proj @ x_proj_dt`, then selected to 768x768;
- dt bias: selected rows.

All mapped VN97 projection weights are snapped into VN97's native ternary
parameter representation before checkpoint export.

Not transferred directly:

- Mamba depthwise conv1d;
- Mamba D skip.

Those missing mechanisms are the reason post-transplant alignment is required.

## Tokenizer transplant

Mamba's 50k-token embedding cannot be copied directly into the VN97 tokenizer.
P5B instead:

1. decodes every VN97 byte/BPE token;
2. re-encodes that text with the source tokenizer;
3. averages the corresponding source embeddings;
4. constructs a VN97 vocabulary embedding matrix;
5. factorizes it to the canonical rank-384 tied VN97 embedding/head.

This lets VN97 keep its own tokenizer while inheriting a lexical projection
from the source model.

## Decision rule

P5B has two separate milestones:

1. `STRUCTURALLY_FEASIBLE` — architecture/config bridge is valid.
2. `TRANSPLANT_READY_FOR_ALIGNMENT` — the real 517 MB source checkpoint has
   been converted to a finite native VN97 checkpoint.

Only after milestone 2 do we replace the from-scratch P5A path as the primary
route. P5A remains a fallback until the transplant checkpoint passes alignment
and held-out validation.

## Kaggle structural assessment

```bash
bash tools/kaggle_p5b_mamba_transplant.sh assess \
  /kaggle/working/p4e-k-final/tokenizer.vn97tk1
```

Expected:

```text
VN97 P5B COMPATIBILITY feasible=true lossless=false alignment_required=true ...
VN97P5B1 status=STRUCTURALLY_FEASIBLE lossless=false alignment_required=true
```

## Kaggle real transplant

```bash
bash tools/kaggle_p5b_mamba_transplant.sh convert \
  /kaggle/working/p4e-k-final/tokenizer.vn97tk1
```

The conversion downloads the pinned public Mamba source checkpoint, verifies
its SHA-256, converts it on CPU, writes a native VN97CK1/VN97MI1 artifact, and
runs a finite-logit smoke test.

Expected final marker:

```text
VN97P5B1 status=TRANSPLANT_READY_FOR_ALIGNMENT ...
```

P5B does not promote the checkpoint automatically.
