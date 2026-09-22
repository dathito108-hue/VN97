# M7G — Canonical Sampling + Streaming Generation

M7G completes the first production end-to-end text generation path on top of M7E. It does not add a second model, a Transformer/LLaMA backend, or a parallel runtime.

Canonical path:

`VN97TK1 → activated VN97MI1 → native prefill → recurrent native step → native sampler → VN97TK1 byte decode → UTF-8 streaming response`

## Native sampler

The sampler consumes the logits produced by the existing VN97 language core and supports:

- temperature, with `temperature=0` as stable greedy decoding;
- top-k, with `0` meaning disabled;
- top-p nucleus truncation;
- deterministic seed + absolute RuntimeSession sequence-position counter;
- sampler workspace is allocated once when a generation cursor opens; the token loop does not allocate sampler heap memory;
- stable lowest-token-ID tie breaking for greedy decoding;
- rejection of non-finite logits/configuration.

The sampler has no hidden mutable RNG state. A VN97RUN2 checkpoint already preserves the model identity, recurrent state, and absolute sequence position, so sampling can resume without introducing a second checkpoint format.

## Native generation cursor

The generation cursor keeps the full vocabulary logits in native C++ memory across generated tokens. Android receives only the sampled token ID each step, avoiding a vocabulary-sized JNI transfer for every token.

Opening a cursor:

1. verifies model/runtime handles and batch=1 streaming geometry;
2. bounds combined logits + reusable sampler workspace to 64 MiB;
3. registers the cursor before mutating runtime state so registry allocation failure cannot consume the prompt;
4. prefills the prompt through the M7E model-bound native path;
5. records the resulting RuntimeSession sequence position.

Every `next` call checks the live RuntimeSession sequence position before sampling and after the recurrent step. External/interleaved mutation is therefore detected as a sequence mismatch instead of silently continuing from an unexpected context.

The sampled token is committed through `vn97_model_runtime_infer_step` before it is surfaced to Android. This means a persisted VN97RUN2 checkpoint always corresponds to tokens already exposed by the generation stream.

## Android streaming

`NativeRuntimeSession.generateStreaming(...)`:

- accepts only a trusted `NativeActivatedModel` that already passed M9 activation/M7E loading;
- automatically adds BOS only for a fresh runtime sequence;
- keeps the model/runtime JVM locks for the lifetime of the native cursor;
- emits one `NativeGenerationChunk` per sampled token;
- supports cooperative cancellation when the callback returns `false`;
- reports EOS, token-limit, or cancellation stop reasons;
- preserves byte-level tokenizer correctness with an incremental UTF-8 decoder.

The UTF-8 accumulator holds incomplete multi-byte suffixes between tokenizer tokens. This prevents a byte-level VN97TK1 token split from creating replacement characters merely because a Unicode code point spans multiple generated tokens.

## Authority and trust

M7G does not download, trust, activate, approve, or execute external capabilities. M9 remains the publisher/package/activation boundary and M6 remains the deny-by-default external action authority. Generation consumes only the model handle created by the M7E trusted activated-artifact loader.

## Verification surface

M7G adds regression coverage for:

- greedy tie behavior;
- top-k and top-p filtering;
- deterministic seed/counter sampling;
- invalid/non-finite sampler inputs;
- generation cursor prefill/step sequence accounting;
- stale/invalid cursor handles;
- JNI sampler and generation bridge compilation;
- split multi-byte UTF-8 streaming decode;
- existing M7B host lifecycle test remains in the host gate.
