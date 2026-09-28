# VN97 R2-G0.9 — Identity-bound Mamba-2 cognition candidate

G0.9 joins the correctness components built in G0.6, G0.7 and G0.8 into one
candidate-only cognition path.

It deliberately does not replace the canonical app cognition bridge yet.

## Bound identities

The bridge schema is:

    VN97M2G09BRIDGE1

It binds:

- G0.6 runtime ID;
- G0.5 ONNX manifest ID;
- G0.3 capsule ID;
- inherited source-weight SHA-256;
- G0.8 GPT-NeoX tokenizer ID;
- G0.7 device-specific tuning ID;
- G0.7 physical profile receipt ID;
- source tokenizer token-ID space;
- runtime logits width;
- exact EOS token ID.

The binding also requires:

    same_weights_semantics=true
    same_token_ids_required=true
    padded_logits_must_be_masked=true
    device_measured_tuning_required=true
    candidate_validation_only=true
    production_activation_authorized=false

Runtime, tokenizer and tuning must all resolve to the same identities before
the candidate opens.

## Candidate inference chain

The candidate path is:

    UTF-8 prompt
      -> VN97GptNeoXTokenizer
      -> exact source token IDs
      -> VN97Mamba2OrtExecutor.prefill
      -> G0.5/G0.6 recurrent ONNX graph
      -> mask padded logits [50277,50288)
      -> source-vocabulary sampler
      -> valid_length=1 recurrent decode
      -> GPT-NeoX byte decode
      -> strict UTF-8 output

There is no fallback to:

- VN97TK1;
- the old F1 ONNX executor;
- NativeRuntimeSession;
- Mamba/Triton/CUDA;
- a second AI model.

## Sampling

The sampler rejects NaN and positive infinity. Negative infinity is accepted
only as a masked/non-candidate logit.

Every one of the 11 padded embedding rows must be negative infinity before
sampling. Greedy and temperature/top-k/top-p modes only operate over the first
50,277 source-tokenizer IDs.

EOS is read from the G0.8 tokenizer descriptor rather than hardcoded.

## VN97MEM1 retrieval geometry

G0.9 implements NativeCognitionInference so existing planner/memory plumbing can
be validated later without introducing a second embedding model.

For retrieval only, GPT-NeoX token IDs are projected deterministically into the
2,560-dimensional model space and L2-normalized.

This projection is not a language model and does not generate logits. A later
memory-migration milestone may rebuild stored retrieval vectors when the
Mamba-2 path becomes canonical.

## Why candidate-only

G0.9 proves that the pieces can be wired together with identity checks, but the
project still lacks the real evidence required to authorize production:

1. real 5.4 GB G0.3 capsule;
2. real G0.5 external-data ONNX graph;
3. source -> VN97 -> ONNX Runtime numerical/generation parity;
4. real official GPT-NeoX -> G0.8 token-ID parity;
5. physical S21 FE G0.7 provider profile and tuning;
6. physical S21 FE memory/latency/thermal qualification.

Therefore VN97Mamba2CognitionCandidate is not referenced by
AndroidPlatformRuntime. Production rebinding belongs to the evidence-promotion
milestone after these gates pass.
