# VN97 Architecture Contract v0

VN97 is a mobile-first sovereign assistant architecture. The language model is one subsystem,
not the definition of AGI by itself.

## Core retained from the prototypes

- recurrent selective state-space dynamics instead of global softmax attention;
- constant-size per-layer recurrent state for token generation;
- input-dependent B, C and timestep selection;
- gated residual blocks with RMSNorm;
- tied input embedding / output head;
- ternary projection weights as the target execution representation;
- tile-aware physical weight layout as a backend optimization boundary;
- modality identity as an explicit signal rather than silently mixing sources.

## Corrections locked into the canonical design

1. Per-output-channel ternary scale replaces one scale for an entire matrix.
2. Bounded positive timestep prevents extreme state updates.
3. Negative diagonal A preserves stable decay dynamics.
4. Exact diagonal zero-order-hold input factor replaces the rough dt times B approximation.
5. Multi-timescale log-spaced dynamics improve memory diversity at initialization.
6. Full-sequence execution must match token-by-token recurrent execution while carrying state.
7. Model recurrent state is separated from long-term episodic and semantic memory.
8. 1.58-bit is not used as a storage claim: the deployable v1 format stores fixed-width
   2-bit symbols, while log2(3) is only the ternary information lower bound.
9. Tile geometry is selected by a backend/device profile. Python loops are not treated as NPU
   acceleration and no universal 16x16/32x32 SRAM geometry is assumed.
10. Raw bytes are a lossless transport/fallback representation. Audio and vision require
    modality-specific frontends before shared semantic reasoning.
11. Conditional compute must be implemented with an exportable/vectorized routing contract;
    Python data-dependent branches are not part of the production graph.
12. Backend selection is explicit and testable. auto may choose an optimized backend only
    when that backend is compiled and available; explicit unavailable requests fail closed.
13. Full-sequence training/reference execution uses associative affine scan semantics.
14. Native deployment recurrence updates caller-owned state in place and preserves M0/M2A
    numerical semantics.
15. Token-dependent exact ZOH discretization is fused in native deployment execution; production
    prefill does not require materialized decay/drive tensors of shape [B,L,D,N].
16. Stable diagonal A [D,N] is cached from -exp(a_log), finite and strictly negative.
17. VN97TK1 tokenization is byte-lossless: every input byte has a canonical token ID even when
    no learned token matches. Learned tokens may only shorten a byte sequence, never make input
    unrepresentable.
18. Token control IDs and byte-token IDs are stable format-level identities. Text/audio/vision
    modality tags are reserved before learned vocabulary IDs so future frontends do not remap
    existing text tokens.
19. Learned-token lookup uses serialized first-byte buckets ordered longest-first; native
    encoding must not scan the complete learned vocabulary at every input position.
20. Full tied embeddings remain the compatibility default. Optional factorized tied embeddings
    use one shared VxR factor matrix and RxD projection for both input lookup and output logits;
    no independent language-head copy is permitted.

## System layers

    Modality frontends / tokenizer / byte fallback
                    |
                    v
          VN97 recurrent language core
                    |
                    v
        Reasoning + planning controller
                    ^
                    |
       Working / episodic / semantic memory
                    ^
                    |
        Tool + internet + device action fabric
                    |
                    v
        Authority / approval / audit boundary
                    |
                    v
       Android native runtime + 3D assistant shell

## Non-negotiable mobile constraints

- generation memory must not grow linearly with context length;
- production ternary weights must be physically packed, not merely simulated as FP tensors;
- optimized kernels must consume packed weights without full-matrix dequantization;
- backend dispatch must retain a portable correctness fallback;
- production recurrence must use native fused execution, not a Python token loop;
- deployment prefill must avoid sequence-expanded recurrent intermediates when compact inputs suffice;
- tokenizer transport must be lossless and independent of external tokenizer services;
- learned tokenizer lookup must use an indexed mobile representation;
- embedding compression must preserve exact input/output weight tying;
- long-term exact recall must live in explicit memory/retrieval instead of overloading recurrent state;
- external side effects must pass through an explicit authority boundary;
- Android background continuity must respect operating-system scheduling and lifecycle limits.

## Roadmap

### M0 - Reference recurrent intelligence core - complete
Established the numerical contract, stable SSM dynamics, ternary training path and recurrent invariant tests.

### M1 - Native packed ternary execution - complete
M1A defined VN97T2 and a portable scalar packed matvec. M1B added explicit backend dispatch and
ARM64 NEON packed execution.

### M2 - Parallel training / fused recurrence - complete at kernel contract
M2A provides the associative PyTorch affine scan. M2B adds native in-place recurrence/readout.
M2C fuses timestep preparation, exact ZOH discretization, recurrent update and readout from
compact projected inputs.

### M3 - Mobile tokenizer, modality adapters and embedding compression - in progress
M3A is complete: VN97TK1 provides stable control IDs, 256 lossless byte tokens, learned
multi-byte vocabulary, serialized first-byte bucket indexing, Python/native equivalence and
C-linkage encode/decode. Optional factorized tied embeddings reduce vocabulary parameters while
the default full embedding/head remains checkpoint-compatible.

M3B will add modality adapter contracts: text identity is already reserved in VN97TK1; audio
and vision must enter through efficient frontends and explicit modality identity before the
shared recurrent core.

### M4 - Sovereign memory
Add bounded working memory plus persistent episodic and semantic memory with retrieval,
provenance and retention policy.

### M5 - Reasoning and planning controller
Add deliberate reasoning, task decomposition, verification, interruption/resume and
uncertainty-aware execution.

### M6 - Tool and authority fabric
Internet, files, apps and device actions through typed capabilities, explicit permissions,
approvals and auditable receipts.

### M7 - Android native runtime
JNI/NDK runtime, hardware-aware scheduling, checkpoint/recovery, background continuation and
battery/thermal governance.

### M8 - Interactive 3D assistant
Low-latency avatar shell, speech/vision hooks and continuity with the sovereign cognition state.

### M9 - Capability acquisition
Controlled import/adaptation pipeline that converts compatible learned capability into
VN97-native packages without silently mutating trusted runtime code.
