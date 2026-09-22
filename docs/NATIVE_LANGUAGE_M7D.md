# M7D — Native VN97 Language Inference Execution

M7D turns the native runtime from a state/checkpoint owner into a real recurrent language executor.

## Canonical step

For each input token and batch element:

1. resolve the tied token embedding;
2. for each VN97 layer:
   - RMSNorm;
   - VN97T2 `in_proj` producing signal + gate;
   - VN97T2 `dt_proj` + F32 bias;
   - VN97T2 `b_proj`;
   - VN97T2 `c_proj`;
   - existing fused selective exact-ZOH recurrent step;
   - VN97T2 `out_proj`;
   - residual add;
3. final RMSNorm;
4. tied LM-head logits.

The selective kernel receives precomputed stable `a = -exp(a_log)`. A trusted loader derives
that array from canonical model parameters once; M7D does not recompute model packaging.

## Embedding/head modes

`LanguageEmbeddingKind::kFullF32`

- embedding: `[vocab, d_model]`;
- output logits: hidden × embeddingᵀ.

`LanguageEmbeddingKind::kFactorizedF32`

- token factors: `[vocab, rank]`;
- projection: `[rank, d_model]`;
- input: token factors × projection;
- output: hidden × projectionᵀ × token factorsᵀ.

This is the exact M3A tied factorization.

## Model view

`LanguageModelView` contains immutable/non-owning pointers to:

- model identity and geometry;
- full or factorized embedding data;
- per-layer RMSNorm weights;
- parsed VN97T2 projection views;
- dt bias;
- stable A coefficients;
- final RMSNorm weights.

It does not own or free those buffers. The owner must keep them alive for the call.

M7D deliberately does not define a new model file format. The intended future path is:

`M9 verified/activated artifact → trusted Android/native loader → LanguageModelView → RuntimeSession`.

## Workspace

`LanguageWorkspaceFloats()` computes the exact F32 scratch requirement for the requested batch.
The recurrent state remains external to the language primitive, allowing RuntimeSession to keep
the canonical layout:

`[layer, batch, d_model, d_state]`.

No learned weight buffer is copied into runtime scratch.

## RuntimeSession binding

`RuntimeSession::InferStep()`:

- requires ACTIVE lifecycle;
- requires exact model/session layer/model/state geometry;
- validates the model on first binding;
- uses the session's resolved recurrent and packed backends;
- executes one real recurrent token per batch item;
- increments sequence position only after successful inference;
- binds the session to `model_id` only after that success.

A bound session rejects a different model identity. It also rejects the legacy metadata-only
`Advance()` operation.

## VN97RUN2

VN97RUN1 remains unchanged for unbound sessions.

A bound session uses VN97RUN2:

- bytes 0..7: `VN97RUN2`;
- version 2;
- header size 100;
- existing shape/backend/lifecycle/sequence/state-count fields;
- payload CRC32;
- model-bound flags;
- 32-byte model identity;
- header CRC32 over bytes 0..95;
- unchanged F32 recurrent-state payload after the header.

Restore accepts both VN97RUN1 and VN97RUN2. VN97RUN2 verifies flags/model-ID consistency and CRCs
before hydrating state.

Because no real language inference existed before M7D, a legacy VN97RUN1 checkpoint with nonzero
sequence position or nonzero state is not implicitly rebound to a model. That fails closed.

## Verification

The M7D test surface contains:

- a tiny real VN97T2 model whose matrices are serialized and parsed by `ParsePackedTernary()`;
- independent dense reference math for full tied embedding → selective recurrence → tied logits;
- batch recurrent-state/logit comparison;
- factorized tied embedding/head execution;
- invalid-token/output-bound/model-ID checks;
- RuntimeSession first-inference model binding;
- rejection of metadata-only Advance after binding;
- VN97RUN2 checkpoint/restore identity persistence;
- wrong-model rejection without recurrent-state mutation;
- VN97RUN2 header/model-ID tamper rejection;
- VN97RUN1 backward restore plus fail-closed legacy rebinding.

A standalone isolated prototype of the language core passed C++17 `-Werror` with ASan/UBSan.
Full repository/device verification is still required for the final branch because the local
container currently cannot resolve github.com for checkout.
