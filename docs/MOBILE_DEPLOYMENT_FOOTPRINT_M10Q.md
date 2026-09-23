# M10Q — Mobile Deployment Footprint Gate

M10Q extends the M10P production training campaign with an exact pre-allocation
mobile deployment footprint gate.

The model architecture does not change. Candidates are still the canonical
VN97LanguageCore only, with the existing Code-1 -> Code-2 packed ternary path.

## Why parameter count is not enough

M10P already rejects candidates above an explicit parameter budget before real
weights are allocated. A parameter count alone does not describe the mobile
runtime footprint because VN97 deployment contains a mixture of:

- tied float32 embedding/factor tensors;
- per-layer RMSNorm and selective-state tensors;
- VN97T2 packed ternary projections with tile padding, row scales and headers;
- optional VN97TK1 tokenizer bytes;
- fixed VN97MI1 section-table/header overhead;
- recurrent runtime state `[layers, batch, d_model, d_state]`.

M10Q therefore computes the exact geometry-derived VN97MI1 byte length before
training. Learned values are not needed for the calculation.

## Exact estimator contract

`estimate_vn97_mobile_footprint()` mirrors the existing
`build_model_image()` layout:

- identical VN97MI1 header and section-table sizes;
- identical section ordering;
- identical 4-byte section alignment;
- exact full/factorized tied-embedding storage;
- exact per-layer float32 sections;
- exact VN97T2 2-bit tile padding;
- exact VN97T2 row-scale and header cost;
- exact tokenizer byte count when the shared VN97TK1 package is included.

Regression tests compare the estimator directly with real VN97MI1 output for
both full and factorized tied embeddings.

## Campaign admission

`vn97-campaign` gains bounded deployment controls:

- `--max-model-image-bytes`;
- `--max-recurrent-state-bytes`;
- `--deployment-tile-rows`;
- `--deployment-tile-cols`.

The defaults preserve the existing 512 MiB runtime format/state ceilings.
Production campaigns can set smaller device-class limits.

Candidate rejection remains deterministic:

1. parameter budget;
2. model-image byte budget;
3. recurrent-state byte budget;
4. M10N held-out quality gate after training.

A mobile-budget rejection happens before real candidate weights are allocated
or training begins.

Each campaign row records the estimated mobile footprint, and the campaign
criteria record the deployment tile geometry and byte ceilings.

## Architecture boundary

M10Q adds no alternate model, Transformer/LLaMA backend, cloud inference path,
or vendor-specific NPU model.

It only makes the existing VN97 candidate-selection path aware of the exact
mobile deployment geometry already consumed by VN97MI1/VN97T2 and the native
runtime.
