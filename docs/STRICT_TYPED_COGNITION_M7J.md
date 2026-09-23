# M7J — Strict Typed Cognition Adapter

M7J adds the strict typed Android adapter above M7I's native VN97COG1 inference engine.

The architecture remains:

`Code 1 → Code 2 hardware/mobile-aware upgrade → VN97 production`

No second planner, model, inference backend, memory engine, authority gate, or external-action executor is introduced.

## Purpose

M7I already provides:

`VN97COG1 → VN97TK1 → fresh model-bound RuntimeSession → native generation / final-hidden embedding`

M7J converts the generated JSON into bounded typed objects before any planner-facing code may trust it.

The new path is:

`typed request → canonical JSON → M7I inference → strict JSON parser → exact schema validation → typed result`

## Strict JSON core

`VnStrictJson` is dependency-free and fail-closed.

It enforces:

- UTF-8 input byte bounds;
- bounded nesting depth and node count;
- bounded decoded string bytes;
- duplicate-key rejection;
- exact JSON number grammar;
- finite floating-point values only;
- no trailing prose;
- control-character rejection;
- strict escape decoding;
- surrogate-pair validation;
- deterministic canonical object-key ordering.

Canonical request serialization means model-facing M5C requests do not depend on JVM map iteration order.

## Typed M5C surface

`NativeTypedCognitionAdapter` mirrors the five M5C operations:

- plan;
- memory_query;
- step;
- verify;
- external_intent.

Typed structures cover:

- plan-step kind/objective/dependencies/verification/confidence;
- dependency results and evidence record IDs;
- memory query kinds, retrieval weights and half-life;
- memory context with provenance scores;
- step proposals;
- verification decisions;
- trusted capability catalog views;
- external intent scope and canonical payload JSON.

Every generated object must contain exactly the expected keys. Missing or extra keys fail closed.

## Numeric contracts

M7J distinguishes integers from numeric floats.

Examples:

- `top_k: 2` is valid;
- `top_k: 2.0` is rejected;
- plan dependencies must be positive integer IDs;
- confidence must be finite and in `[0, 1]`;
- retrieval weights must be finite and non-negative with positive total weight;
- recency half-life must be a positive integer.

This matches the M5C reference adapter instead of relying on permissive JVM number coercion.

## Retrieval embedding

For `memory_query`, M7J first validates the generated retrieval request, then invokes M7I's
`embedText(query, vectorDim)`.

The returned vector must:

- exactly match `vectorDim`;
- contain only finite values;
- have non-zero finite norm.

The adapter then returns a typed memory query ready for the existing sovereign memory retrieval contract.

## External authority boundary

`external_intent` remains a proposal only.

M7J:

- parses `capability_id`;
- parses string-only scope;
- canonicalizes the payload object;
- returns `NativeExternalIntent`.

It does **not**:

- select an unadvertised capability;
- mint approval;
- execute a capability;
- bypass M6 policy/authority;
- activate a package.

A later binder/controller must re-check the intent against the trusted capability catalog and M6 authority before any effect occurs, preserving the existing Python M6/M5 design.

## No planner duplication

M7J deliberately does not clone the M5 planner state machine into Android.

It supplies typed native cognition results that a planner bridge can consume. Plan lifecycle,
budgets, retries, evidence propagation, external waiting states and terminal semantics remain the
canonical M5 contracts until the bridge milestone explicitly maps them.

## Verification surface

The M7J host regression covers:

- all five VN97COG1 operations;
- canonical request ordering;
- plan-step typing;
- memory-query embedding invocation;
- step and verification typing;
- canonical external payload;
- duplicate JSON keys;
- non-standard/non-finite JSON numbers;
- trailing prose;
- unpaired surrogate escapes;
- integer-vs-float contract rejection;
- zero-norm retrieval embedding;
- exact-key enforcement.

The host test prints `M7J_TYPED_COGNITION_PASS` on success.
