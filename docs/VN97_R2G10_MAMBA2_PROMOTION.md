# VN97 R2-G0.10 — real-evidence promotion gate

R2-G0.10 is the first stage allowed to emit a production-authorizing receipt
for the Mamba-2-derived VN97 cognition path.

It does not manufacture evidence. It only composes already measured evidence
from the exact G0.9 lineage.

## Required evidence

A promotion requires all three receipts:

1. VN97M2G10MODELPARITY1
   - exact G0.9 bridge/runtime/capsule/source-weight lineage;
   - full source weights verified;
   - real source -> VN97 parity PASS;
   - VN97 -> ONNX Runtime parity PASS;
   - recurrent-state parity PASS;
   - synthetic=false.

2. VN97M2G10TOKENPARITY1
   - exact G0.9 bridge/tokenizer lineage;
   - official GPT-NeoX reference implementation;
   - at least 100 reference cases;
   - token IDs exact;
   - decoded bytes exact;
   - synthetic=false.

3. VN97M2G10MOBILEQUAL1
   - exact G0.9 bridge/runtime/tuning/profile lineage;
   - target family Samsung Galaxy S21 FE;
   - physical device measurement;
   - zero execution failures;
   - memory PASS;
   - latency PASS;
   - thermal recovery PASS;
   - synthetic=false.

Every receipt is content-addressed. Any modified payload with a stale receipt ID
is rejected before semantic checks.

## Promotion receipt

If and only if all evidence passes, the host compiler emits:

    VN97M2G10PROMOTE1

The promotion binds the G0.9 bridge, runtime, capsule, source-weight hash,
tokenizer, G0.7 tuning/profile receipt and all three G0.10 evidence receipt IDs.

The parity error summaries are stored as integer values scaled by 1e12. This
avoids cross-language floating-point JSON rendering differences and keeps the
promotion ID identical between Python and Kotlin.

A valid promotion contains:

    real_evidence_required=true
    rollback_required=true
    production_activation_authorized=true

This boolean is not a model-generated authority decision. It is the output of
the deterministic evidence compiler over externally produced measurements.

## Android verification

Mamba2ProductionPromotion loads the promotion on Android, recomputes the exact
content identity, and requires compatibility with the active G0.9 binding,
G0.6 runtime, G0.8 tokenizer and G0.7 tuning profile.

A promotion for another runtime, tokenizer, capsule, source weights, tuning or
device-profile receipt fails closed.

## Activation boundary

G0.10 still does not silently replace AndroidPlatformRuntime. The production
selector/rollback wiring must consume a valid promotion receipt explicitly.

Until the three real evidence receipts exist, no real G0.10 promotion receipt
can truthfully be generated.
