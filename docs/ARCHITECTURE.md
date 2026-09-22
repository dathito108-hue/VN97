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

## Canonical corrections and invariants

1. Per-output-channel ternary scale replaces one scale for an entire matrix.
2. Bounded positive timestep prevents extreme state updates.
3. Negative diagonal A preserves stable decay dynamics.
4. Exact diagonal zero-order-hold input factor replaces the rough dt times B approximation.
5. Multi-timescale log-spaced dynamics improve memory diversity at initialization.
6. Full-sequence execution must match token-by-token recurrent execution while carrying state.
7. Model recurrent state is separated from long-term episodic and semantic memory.
8. Deployable ternary storage uses fixed-width 2-bit symbols; log2(3) is only an information
   lower bound.
9. Tile geometry is selected by a backend/device profile.
10. Raw bytes are lossless text transport/fallback, not a substitute for audio/vision frontends.
11. Conditional compute must use an exportable/vectorized routing contract.
12. Backend selection is explicit, testable and fail-closed when unavailable.
13. Full-sequence training/reference execution uses associative affine scan semantics.
14. Native deployment recurrence preserves M0/M2A state semantics.
15. Native selective execution fuses exact ZOH dynamics and avoids materialized [B,L,D,N]
    decay/drive tensors.
16. Stable diagonal A is cached from -exp(a_log), finite and strictly negative.
17. VN97TK1 guarantees a canonical token ID for every byte.
18. Text/audio/vision control IDs are stable format-level identities.
19. Learned tokenizer lookup uses first-byte buckets ordered longest-first.
20. Optional factorized embeddings preserve exact input/output tying.
21. Audio and vision enter through modality-specific preprocessing and ternary projection to
    d_model; raw waveform/pixels never masquerade as text tokens.
22. Modality identity is represented by the same reserved vocabulary embedding used by the
    tokenizer contract, not by an unrelated embedding table.
23. Dense modality embeddings use the same recurrent core and state update contract as text.
24. Native modality preprocessing must numerically match the reference frontend before device
    optimizations are accepted.
25. Working memory has explicit item/byte budgets; persistent episodic/semantic memory is not
    hidden inside the recurrent state or an unbounded process history.
26. VN97MEM1 persistent memory is append-framed, versioned and integrity checked before records
    become visible to cognition.
27. A memory parent_id is valid only when it references an actually present earlier record;
    provenance ancestry must survive retention compaction.
28. Only an incomplete final memory frame may be recovered as a torn tail. CRC failure or a
    malformed complete record fails closed.
29. Persistent record IDs are stable across compaction; the highest ID is retained so append
    cannot silently reuse an earlier identity.
30. Memory retrieval semantics are deterministic and storage/model agnostic. Vector production
    can evolve without changing persistent record meaning.
31. CRC32 and content SHA-256 are integrity/content-identity mechanisms, not authority or
    cryptographic author signatures.
32. The native retrieval index stores journal offsets and precomputed inverse norms instead of
    copying every vector into a second resident vector table.
33. A native MemoryStore holds an exclusive advisory writer lock for its lifetime; concurrent
    writers fail closed instead of racing record IDs or compaction.
34. Native append extends the existing index only from the previously validated byte boundary;
    full rebuild is reserved for open/recovery/compaction boundaries.
35. A reasoning plan has a stable SHA-256 identity over canonical goal, immutable step graph and
    reasoning budget; creation time and runtime progress do not change plan identity.
36. Reasoning is bounded by explicit transition, retry, memory-query and memory-hit budgets.
37. Low-confidence or explicitly protected candidates enter WAITING_VERIFICATION before they can
    satisfy dependent steps.
38. M5 never performs external side effects. EXTERNAL plan steps stop at WAITING_EXTERNAL until
    an authority/tool layer supplies a result.
39. An interrupted RUNNING internal step is requeued before checkpointing; VN97PLN1 never treats
    a partial in-flight computation as a durable result.
40. VN97PLN1 stores canonical JSON with SHA-256 integrity, but its digest is not an authority
    grant or cryptographic author signature.
41. Retrieved memory evidence is referenced by stable VN97MEM1 record IDs and is independently
    bounded by planner retrieval budgets.

## System layers

    Text tokenizer / audio frontend / vision frontend
                    |
                    v
           compact d_model embeddings
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
- production ternary weights must be physically packed;
- optimized kernels must consume packed weights without full-matrix dequantization;
- production recurrence must use native fused execution;
- deployment prefill must avoid unnecessary sequence-expanded recurrent intermediates;
- tokenizer transport must be lossless and independent of external tokenizer services;
- learned tokenizer lookup must use an indexed mobile representation;
- embedding compression must preserve exact input/output weight tying;
- audio/vision frontend output must be bounded, compact and compatible with packed projection;
- modality identity must be explicit before shared recurrent reasoning;
- working memory must have explicit finite budgets;
- long-term memory must use an explicit persistent VN97-native format rather than unbounded
  hidden conversation/process state;
- memory corruption must fail closed except for explicit recovery of an incomplete final frame;
- long-term exact recall belongs in explicit memory/retrieval;
- deliberate reasoning must have finite transition/retry/retrieval budgets;
- plan checkpoints must be written only at durable lifecycle boundaries;
- external side effects must pass through an explicit authority boundary;
- Android background continuity must respect OS scheduling and lifecycle limits.

## Roadmap

### M0 - Reference recurrent intelligence core - complete
Stable SSM dynamics, ternary training path and recurrent invariants.

### M1 - Native packed ternary execution - complete
VN97T2 physical packing, scalar kernel, ARM64 NEON backend and explicit dispatch.

### M2 - Parallel training / fused recurrence - complete
Associative training/reference scan plus native fused recurrence and exact-ZOH selective kernel.

### M3 - Mobile tokenizer, modality adapters and embedding compression - complete at frontend contract
M3A adds VN97TK1 and optional exactly tied factorized embeddings. M3B adds audio frame and
vision patch frontends, native preprocessing equivalence, explicit reserved modality identity
and dense embedding ingress into the same recurrent core. Speech/vision capability quality is a
training or capability-package concern and does not require changing the runtime contract.

### M4 - Sovereign memory - complete at native engine contract
M4A establishes bounded working memory, VN97MEM1 append-only episodic/semantic storage,
provenance, deterministic retrieval, retention/atomic compaction and torn-tail recovery.
M4B adds the native file-backed engine: exclusive writer locking, mmap record access,
incremental suffix indexing without vector duplication, native SHA-256 append, exact retrieval,
record views and atomic provenance-preserving compaction. VN97MEM1 remains unchanged.

### M5 - Reasoning and planning controller - in progress
M5A adds a deterministic bounded plan state machine, dependency ordering, confidence-driven
verification, bounded VN97MEM1 retrieval, interruption/resume semantics, an explicit
WAITING_EXTERNAL boundary and atomic VN97PLN1 safe-point checkpoints.

M5B will connect the state machine to a cognition backend contract for bounded proposal,
reflection/verification and plan refinement without weakening M5A lifecycle invariants.

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
