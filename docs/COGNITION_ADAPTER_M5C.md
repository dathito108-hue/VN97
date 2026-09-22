# VN97 M5C native cognition adapter

M5C binds the typed M5B CognitionBackend contract to VN97-owned tokenizer/model inference.
It does not introduce a third-party LLM/backend and it does not weaken the M5A/M5B lifecycle,
memory provenance or M6 external-authority boundaries.

## Inference boundary

VN97InferenceEngine is the stable cognition-facing inference contract:

- generate_text(prompt, max_new_tokens) -> UTF-8 text
- embed_text(text, vector_dim) -> finite float vector

VN97CognitionAdapter depends only on this interface. The reference implementation,
TorchVN97InferenceEngine, directly uses VN97TK1 and VN97LanguageCore.

A future C++/Android full-model runtime can implement the same interface without changing
planner/cognition schemas.

## Reference recurrent inference

TorchVN97InferenceEngine requires model.config.vocab_size to exactly match the active VN97TK1
vocabulary.

Generation:

1. encodes the structured cognition prompt with BOS + TEXT control tags;
2. enforces a bounded prompt-token budget;
3. prefills VN97LanguageCore once;
4. continues token-by-token using caller-visible recurrent states;
5. stops on EOS or max_new_tokens;
6. decodes only through VN97TK1;
7. rejects invalid UTF-8 or output beyond the configured byte budget.

This is reference execution over the VN97 model. M5C does not claim that random/untrained
reference weights produce useful plans or reasoning.

## Native retrieval embedding

VN97LanguageCore now exposes forward_hidden() / forward_hidden_embeddings().

The normal logits path is:

    hidden = final_norm(recurrent_core(...))
    logits = lm_head(hidden)

Existing forward()/forward_embeddings() use the same hidden path, so the public language-model
semantics are unchanged.

The reference inference engine uses the final normalized hidden state of the encoded retrieval
text as the retrieval vector. It requires:

    VN97MEM1.vector_dim == model.d_model

The language model therefore selects retrieval query text, but it does not emit an arbitrary
float array in JSON. Vector production remains local VN97 model computation.

A future trained retrieval head can implement VN97InferenceEngine.embed_text without changing
the M5B MemoryQuery contract.

## VN97COG1 prompt envelope

Every cognition operation is serialized as a bounded UTF-8 prompt:

    VN97COG1
    operation=<plan|memory_query|step|verify>
    Return exactly one UTF-8 JSON object...
    schema=<operation schema>
    request=<canonical JSON>

Request JSON is deterministic:

- UTF-8;
- sorted keys;
- compact separators;
- NaN/Infinity prohibited.

User goals, dependency text and memory content are JSON data fields rather than interpolated
schema fragments.

## Strict structured output

The adapter accepts exactly one JSON object. It rejects:

- markdown fences or surrounding prose;
- duplicate JSON keys;
- NaN/Infinity;
- wrong root type;
- missing keys;
- extra keys;
- wrong scalar/list types;
- unknown step or memory kinds;
- invalid confidence/weights;
- invalid dependency IDs;
- empty required strings.

Typed objects are then passed back through M5B, which still enforces plan size, context bounds,
retry budgets and external-step limits.

## Plan schema

Plan output is:

    {
      "steps": [
        {
          "kind": "REASON|RETRIEVE|VERIFY|RESPOND|EXTERNAL",
          "objective": "...",
          "dependencies": [1],
          "requires_verification": false,
          "min_confidence": 0.0
        }
      ]
    }

M5B remains responsible for final-response requirements and immutable graph validation.

## Memory-query schema

The model returns retrieval intent, not a raw embedding:

    {
      "query": "...",
      "top_k": 5,
      "kinds": ["EPISODIC", "SEMANTIC"] | null,
      "semantic_weight": 1.0,
      "recency_weight": 0.0,
      "importance_weight": 0.0,
      "recency_half_life_ns": 86400000000000
    }

VN97CognitionAdapter calls engine.embed_text(query, vector_dim) locally and constructs the typed
M5B MemoryQuery.

The model cannot inject now_ns and cannot inject evidence record IDs.

## Step / verification schemas

Step proposal:

    {"result": "...", "confidence": 0.0}

Verification:

    {"passed": true, "note": "..."}

Trusted evidence IDs remain attached by M5B from actual VN97MEM1 retrieval/dependency provenance,
never from model output.

## Failure semantics

Inference runtime failures remain backend failures and consume the bounded M5B retry policy where
that policy applies.

Inference-contract or structured-output violations are CognitionContractError failures and
therefore fail closed nonretryably inside M5B.

Prompt/output/token limits are explicit and testable.

## M5 completion boundary

After M5C, M5 is complete at the cognition-adapter contract:

- M5A: deterministic plan lifecycle/checkpoint contract;
- M5B: bounded cognition/retrieval/verification orchestration;
- M5C: VN97-owned tokenizer/model inference adapter.

This does not mean VN97 has already learned high-quality AGI reasoning. Model capability still
depends on trained/imported VN97-native intelligence weights. Full Android C++ inference,
hardware scheduling and JNI remain M7 concerns; tool/device authority remains M6.
