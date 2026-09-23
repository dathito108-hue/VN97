# M7I — Native Cognition Inference + Hidden Embeddings

M7I connects the existing M5C `VN97COG1` inference contract to the activated native Android
runtime. It does not introduce a second model, an alternate backend, or a replacement cognition
architecture.

The architecture remains:

`Code 1 → Code 2 hardware/mobile-aware upgrade → VN97 production`

and cognition uses the same trusted M9-activated VN97MI1 model image already consumed by M7E/M7G.

## M5C-compatible cognition inference

`NativeCognitionInferenceEngine` preserves the M5C inference boundary:

- protocol: `VN97COG1`;
- operation names: `plan`, `memory_query`, `step`, `verify`, `external_intent`;
- the same operation schemas and generation budgets as `src/vn97/cognition_adapter.py`;
- prompt token limit: 4096;
- output UTF-8 limit: 64 KiB;
- adapter prompt UTF-8 limit: 128 KiB;
- BOS + VN97TK1 text control on every cognition request;
- deterministic greedy structured generation;
- EOS control token;
- strict UTF-8 transport.

Each cognition generation or embedding request creates a fresh `NativeRuntimeSession`, activates
it, performs the request against the same `NativeActivatedModel`, and closes it. This matches the
reference M5C inference semantics and prevents persistent chat recurrent state from contaminating
internal planning/retrieval/verification operations.

A fresh cognition session is an ephemeral recurrent state for the same model/runtime
implementation. It is not a parallel model or inference backend.

## VN97COG1 prompt envelope

The Android/native prompt builder emits:

```text
VN97COG1
operation=<plan|memory_query|step|verify|external_intent>
Return exactly one UTF-8 JSON object. No markdown fences. No prose outside JSON.
schema=<operation schema>
request=<canonical JSON>
```

M7I accepts only a single bounded request-object line at this low-level boundary. Canonical sorted
JSON serialization plus strict duplicate-key/type/schema parsing is intentionally the next typed
adapter milestone; M7I does not claim the complete M5B cognition loop.

## Hidden-only native inference

M5C memory retrieval uses the final normalized language hidden state as its embedding. Before M7I
the native core exposed only vocabulary logits.

M7I refactors the native language step into one shared internal recurrence path that can expose
either:

- final logits; or
- final RMS-normalized hidden state.

`LanguageStepHiddenF32WithBackends` therefore reuses exactly the same embedding, selective SSM,
Code-2 packed ternary kernels, residual path, final normalization, backend selection, model
validation and recurrent-state mutation as normal language inference. It simply skips the tied
LM-head vocabulary projection when logits are unnecessary.

This hidden path is propagated through:

`LanguageModelView → RuntimeSession → activated model registry → JNI → NativeRuntimeSession.prefillHidden()`

The ordinary logits path remains unchanged at its public contract.

## Retrieval embedding

`NativeCognitionInferenceEngine.embedText(text, vectorDim)`:

1. requires `vectorDim == model.dModel`;
2. encodes with VN97TK1 BOS + text control;
3. enforces the 4096-token inference limit;
4. runs hidden prefill in a fresh model-bound session;
5. requires exactly `dModel` finite values;
6. rejects a zero or non-finite vector norm;
7. returns the raw final normalized hidden vector.

The vector is not normalized again, matching the reference M5C `TorchVN97InferenceEngine`.

## Model identity and trust

All native cognition inference consumes an already opened `NativeActivatedModel`. Opening that
model remains the M7E path from an M9 committed artifact and the native loader still hashes the
exact artifact range before exposing a model handle.

Every fresh cognition RuntimeSession binds the same 32-byte model identity on first inference.
M7I cannot activate packages, mint approvals, or execute external capabilities. External-intent
model output is only cognition data; execution still requires the existing M6 authority path.

## Tokenizer decode

M7I centralizes raw VN97TK1 decoded bytes in `NativeActivatedModel.decodeBytes()`.

- M7G streaming reuses the same byte source for incremental UTF-8 assembly.
- M7I structured cognition uses a strict UTF-8 decoder with malformed/unmappable input set to
  REPORT, matching the reference M5C transport contract.

## Verification surface

Repository regression wiring covers:

- hidden-only native output against the reference final RMSNorm output;
- hidden-only recurrence state equivalence with the ordinary language step;
- activated-model hidden prefill;
- exact model binding and sequence-position accounting;
- exact VN97COG1 operation/schema framing;
- rejection of envelope-breaking request text;
- strict UTF-8 rejection;
- canonical M5C operation generation budgets.

The Android host gate continues to run M7B, M7G and M7H tests before the M7I cognition contract
test.

## Deliberate next boundary

M7I provides the production native inference engine required by M5C, but it does not duplicate the
Python M5B planner in Kotlin yet.

The next milestone should add the strict typed Android cognition adapter:

`canonical JSON serializer → strict bounded parser → typed plan/memory-query/step/verify/external-intent objects → M5 planner bridge`

That layer must preserve evidence provenance and keep all external actions behind M6.
