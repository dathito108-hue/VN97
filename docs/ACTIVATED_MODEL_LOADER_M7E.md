# M7E — Activated Model Loader + JNI Inference Wiring

M7E closes the deployment gap deliberately left by M7D. It does **not** introduce a second model architecture, a Transformer/LLaMA backend, or an alternate capability system.

Canonical path:

`VN97LanguageCore (Code 1 + Code 2 hardware-aware weights) → VN97MI1 runtime image → VN97CAP1 / M9 verify + activate → trusted activated artifact FD + artifact SHA-256 → mmap loader → LanguageModelView → RuntimeSession → JNI`

## VN97MI1 runtime image

`VN97MI1` is a read-only serialization of the existing native `LanguageModelView`. It is an activation artifact, not a competing model/package architecture. `VN97CAP1` remains the capability/trust envelope and M9 remains the transactional activation authority.

The image contains, in canonical order:

- runtime config: vocab, `d_model`, layer/state counts, `dt_min`, `dt_max`, `rms_eps`;
- one full tied F32 embedding, or factorized tied token factors + projection;
- for every layer: RMSNorm, VN97T2 `in/dt/B/C/out` projections, F32 `dt_bias`, and F32 `a_log`;
- final RMSNorm;
- optional VN97TK1 tokenizer bytes.

Every section is bounded, 4-byte aligned, ordered exactly, has zero flags/reserved fields, and must consume the artifact without hidden/trailing bytes. Runtime image size is capped at 512 MiB, matching the existing M9 package bound.

## Trusted loading and identity

Android receives a `TrustedActivatedModelArtifact` only through the internal M9 bridge. It carries an already-activated artifact FD/range plus the exact 32-byte M9 `artifact_sha256`.

Native loading:

1. validates regular-file FD/range and configured size bound;
2. duplicates the FD and maps the exact range read-only/private;
3. computes SHA-256 over the exact mapped artifact range;
4. rejects the artifact unless the digest exactly equals the M9 identity;
5. validates all image sections and each embedded VN97T2/VN97TK1 object;
6. maps F32 and packed weights zero-copy;
7. derives stable `A = -exp(a_log)` once into a bounded owned array;
8. constructs and validates the existing `LanguageModelView`.

The artifact SHA-256 becomes `LanguageModelView.model_id`. `RuntimeSession::InferStep` therefore binds the exact activated artifact identity. VN97RUN2 checkpoints preserve that identity; VN97RUN1 restore remains supported and remains unbound until a valid model-bound inference path is established according to the existing M7D rules.

## JNI inference surface

The Android runtime now exposes:

- model open/info/close from a trusted activated artifact;
- VN97TK1 UTF-8 encode/decode when the image contains its tokenizer;
- native recurrent inference step;
- multi-step native prefill with `[steps, batch]` token layout;
- deterministic greedy native generation for batch 1;
- runtime model-binding inspection.

JNI never receives arbitrary model pointers. The model registry keeps the mapped image alive across each runtime call, and the runtime still performs geometry/model-identity validation before state advancement.

M7E greedy generation is the deterministic generation plumbing gate, not the final sampling policy. The next end-to-end block can attach the canonical sampler/streaming layer without changing the loader or VN97 recurrence.

## Authority boundaries

M7E does not download, trust, activate, approve, or grant capabilities. M6 remains the deny-by-default authority boundary for external actions. M9 remains responsible for publisher trust, package verification, compatibility planning, transactional activation and rollback. The M7E loader only consumes an artifact that the host supplies through the trusted post-activation bridge and rechecks its exact digest before mapping it.

## Verification targets

The M7E regression surface covers:

- deterministic full/factorized VN97MI1 export;
- canonical section ordering and embedded VN97T2/VN97TK1 identities;
- tokenizer/model vocabulary binding and finite parameter checks;
- read-only FD mapping and exact SHA-256 rejection;
- stable-A derivation and `LanguageModelView` validation;
- model-registry → RuntimeSession inference/prefill/generation;
- exact RuntimeSession model binding and VN97RUN2 checkpoint production;
- JNI/Kotlin compile wiring while preserving existing M7A/B lifecycle and VN97RUN1 restore behavior.
