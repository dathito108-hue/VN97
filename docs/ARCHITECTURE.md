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
42. Cognition backend outputs are typed proposals, queries and verification decisions; the
    backend never mutates planner lifecycle state directly.
43. A retrieval query vector must exactly match the active VN97MEM1 vector dimension before
    retrieval can run.
44. Dependency and memory text passed to cognition is bounded by explicit UTF-8 byte budgets;
    truncation never changes trusted evidence record IDs.
45. Backend-proposed evidence IDs are not accepted. Evidence is attached only from actual
    VN97MEM1 retrieval and inherited dependency provenance.
46. Plan refinement never mutates a started plan graph. Only an unstarted READY plan may be
    replaced by a new immutable plan with a new plan ID.
47. Backend runtime exceptions consume bounded retry policy, while schema/shape/size contract
    violations fail closed without retry.
48. An EXTERNAL step is never sent through internal cognition execution and remains stopped at
    WAITING_EXTERNAL until M6 supplies an authorized result.
49. Per-run cognition yielding occurs only after a safe step/verification boundary; no RUNNING
    internal step is returned to the caller.
50. Cognition model execution enters M5 only through VN97InferenceEngine; M5C does not introduce
    a third-party LLM/backend dependency.
51. VN97TK1 vocabulary size must exactly match the reference VN97LanguageCore vocabulary before
    reference cognition generation is allowed.
52. Structured cognition output is strict JSON: duplicate keys, wrappers/prose, missing/extra
    fields, non-finite numbers and unknown enums fail closed.
53. The cognition model proposes retrieval text and policy fields, not arbitrary trusted
    embedding arrays or evidence record IDs.
54. Reference retrieval vectors are derived locally from the final normalized VN97 hidden state;
    the active VN97MEM1 vector dimension must equal d_model.
55. Existing language logits are computed from the same exposed normalized hidden path, so
    retrieval-vector access does not create a second recurrent/model semantics.
56. M5C reference execution is not a claim of Android-native full-model runtime; JNI/C++ model
    assembly and hardware scheduling remain M7 responsibilities.
57. An M6 external request is bound to immutable plan ID, step ID and objective plus an exact
    typed capability, canonical scope and canonical payload; changing any bound field changes the
    request digest.
58. The capability registry is sealed before external execution. Unknown, duplicate or
    post-seal capability registration fails closed.
59. Authority is deny-by-default. A policy grant matches only the exact principal, capability ID
    and scope digest; there is no wildcard authority fallback.
60. Approval, when required, is a time-bounded HMAC-SHA256 token bound to the exact request
    digest and principal. Cognition output is never itself an approval.
61. Capability leases are issued only by the authority gate and are bounded by principal,
    capability, exact scope, lifetime and use count; fabricated, expired or exhausted leases
    fail closed.
62. A side effect may run only after registry validation, policy authorization, required approval
    verification and lease consumption. Cognition never receives the handler, policy table,
    approval secret or lease mutation API.
63. M6 writes an immutable success receipt before delivering an external result back to M5.
    Replaying the exact successful request reuses the receipted result instead of invoking the
    side effect twice.
64. M6A receipt SHA-256 chaining provides integrity and idempotency, not an author signature.
    Authorization authenticity is supplied by the trusted approval/policy boundary.
65. Concrete tool implementations register only as typed handlers behind the sealed M6A
    registry; capability code does not gain a second authority path.
66. File access is confined to trusted runtime-configured root IDs plus canonical relative POSIX
    paths. Traversal, symlink components and non-regular targets fail closed.
67. File creation never overwrites an existing target. Replacement requires the caller-bound
    SHA-256 of the current bounded file and uses fsynced temporary data plus atomic publication.
68. Web retrieval is HTTPS GET only on port 443. URL credentials, fragments, redirects and any
    DNS result that is not globally routable are rejected before transport.
69. HTTPS transport connects to a prevalidated resolved IP while TLS SNI/certificate validation
    remains bound to the requested hostname, preventing DNS rebinding between validation and
    connection.
70. M6B web results are bounded strict text/JSON; arbitrary binary downloads are not accepted by
    the base fetch capability.
71. App launch and clipboard mutation are available only through an injected platform adapter;
    M6B does not add shell-command, arbitrary intent or model-direct device execution.
72. M6B capability descriptors require explicit M6A approval by default in addition to exact
    policy grants and bounded leases.
73. Cognition may propose an external capability ID, string-valued scope and JSON payload,
    but it never supplies trusted plan ID, step ID or objective fields for an M6A request.
74. External-intent binding requires a sealed capability registry and a trusted allowlisted
    catalog derived from registered descriptors; cognition cannot enumerate policy grants,
    approval secrets or lease state.
75. The final ExternalActionRequest is rebuilt from the currently WAITING_EXTERNAL planner step
    and revalidated against the selected registered capability before approval is possible.
76. If plan/step/objective/catalog state changes while cognition is proposing an intent, binding
    fails closed rather than rebinding stale intent to a different external step.
77. Approval UX presentation is generated from the exact bound request digest, objective, scope
    and canonical payload by trusted runtime code rather than model-authored prose.
78. Approval prompts are bounded, expiring and one-shot. Denial, expiry, stale/tampered prompts
    and duplicate/concurrent resolution cannot mint an ApprovalToken.
79. Only the trusted approval coordinator may call ApprovalAuthority.approve(); model output,
    notification taps and presentation text are never themselves authority.
80. Android approval UI and Keystore-backed secret storage must preserve the M6C exact-request
    binding and remain outside cognition/model execution.

81. Android/JNI integration uses opaque runtime handles rather than model-owned raw pointers;
    stale handles fail closed and in-flight operations retain session lifetime independently.
82. A native runtime session owns an exact finite recurrent-state shape across layers, batch,
    d_model and d_state; allocation arithmetic is overflow checked and bounded for mobile use.
83. Runtime lifecycle is explicit: CREATED may activate, ACTIVE may suspend, and SUSPENDED may
    resume. Checkpoint creation is valid only at a SUSPENDED safe boundary.
84. Host recurrent-state replacement is prohibited while ACTIVE and requires exact element count
    plus finite float values.
85. Runtime sequence position is monotonic and overflow checked; restore preserves the exact
    checkpointed position.
86. VN97RUN1 is a versioned exact-length little-endian runtime checkpoint with header/payload
    CRC32 integrity, exact state count and requested backend profile.
87. Runtime checkpoint CRC32 is corruption detection only; it is not an author signature,
    capability grant, approval token or substitute for the M6 authority boundary.
88. Restore re-resolves requested recurrent and packed-ternary backends on the current device.
    An explicit unavailable backend fails closed; AUTO may resolve to the local supported backend.
89. Restored runtime sessions always enter SUSPENDED and never resume computation implicitly.
90. M7 native lifecycle/checkpoint code does not bypass M6. External side effects remain reachable
    only through the completed M6 capability/authority/approval/audit path.

91. Android managed code receives only an opaque signed-Long representation of the M7A handle;
    JNI rejects native handles that cannot be represented without sign wrap.
92. JNI validates required array lengths and signed numeric inputs before crossing into the M7A
    C ABI; native RuntimeStatus values remain the canonical lifecycle/checkpoint failure contract.
93. Kotlin serializes session-handle use against close, while native in-flight calls independently
    retain session lifetime through the M7A shared handle registry.
94. The M7B1 Android runtime library declares no permissions and does not implement network,
    app-launch, clipboard or other M6 side-effect capabilities.
95. Runtime checkpoint persistence is scoped to a trusted app-internal root, rejects symlink
    root/target paths, enforces a byte bound, fsyncs temporary data, requires same-filesystem
    atomic replace and fsyncs the parent directory.
96. NativeRuntimeOwner never silently discards a present checkpoint. Restore corruption fails
    closed, and a successfully restored checkpoint must exactly match the requested runtime config.
97. Android runtime persistence occurs only from the M7A SUSPENDED safe boundary; cold restore
    returns SUSPENDED and requires an explicit resume.
98. Android build/JNI integration links against the canonical vn97_runtime library rather than
    copying or reimplementing recurrent-state semantics in Kotlin.
99. M7B1 remains compute/lifecycle infrastructure. Android permissions, Keystore approval secrets
    and AppDeviceAdapter side effects must remain behind the completed M6 authority contract.

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
- cognition model output must cross a strict typed/structured schema boundary;
- retrieval embeddings must be produced locally by trusted VN97 inference rather than accepted
  as arbitrary model-supplied float arrays;
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

### M5 - Reasoning and planning controller - complete at cognition-adapter contract
M5A adds a deterministic bounded plan state machine, dependency ordering, confidence-driven
verification, bounded VN97MEM1 retrieval, interruption/resume semantics, an explicit
WAITING_EXTERNAL boundary and atomic VN97PLN1 safe-point checkpoints.

M5B adds the typed CognitionBackend loop for bounded plan proposal, pre-execution refinement,
memory-query generation, dependency/memory context assembly, proposal execution and
reflection/verification while preserving M5A budgets and the M6 external-authority boundary.

M5C adds VN97CognitionAdapter plus the VN97InferenceEngine boundary. The reference engine binds
VN97TK1 directly to VN97LanguageCore recurrent generation and derives retrieval vectors from the
final normalized hidden state. Strict VN97COG1 JSON schemas fail closed before typed cognition
objects enter M5B. Model-quality reasoning still depends on trained/imported VN97-native
weights; full Android C++ execution remains M7.

### M6 - Tool and authority fabric - complete at platform-neutral contract
M6A establishes the execution authority contract: sealed typed capability registry,
deny-by-default exact-scope policy, request-bound HMAC approval, bounded capability leases,
immutable hash-chained action receipts, crash-safe successful-result replay and the only
authorized transition from M5 `WAITING_EXTERNAL` to side-effect execution.

M6B adds the platform-neutral production capability pack: root-confined strict-UTF-8 file
read/write, SHA-256 compare-and-swap replacement, HTTPS-only bounded text/JSON retrieval with
DNS/IP SSRF protection and pinned TLS connection, plus exact-package launch and clipboard
adapter contracts. All handlers remain behind M6A policy/approval/lease/audit. The concrete
Android platform adapter belongs to M7.

M6C adds a sealed-catalog external-intent binder and VN97COG1 external-intent operation. The
final action request is rebuilt from the active WAITING_EXTERNAL planner step, then presented
through bounded one-shot exact-digest approval sessions. Cognition cannot mint policy grants,
approval tokens or leases. Android presentation/Keystore integration remains M7 work.

### M7 - Android native runtime - in progress
M7A adds the platform-neutral native runtime session ABI: opaque JNI-safe handles, explicit
CREATED/ACTIVE/SUSPENDED lifecycle, bounded recurrent state, backend-profile resolution and
VN97RUN1 safe checkpoint/restore. Restored sessions remain SUSPENDED.

M7B1 adds the permission-free Android/Kotlin ↔ JNI wrapper around M7A, typed runtime ownership and
bounded atomic VN97RUN1 checkpoint persistence. Host regression exercises create/state/lifecycle/
checkpoint/restore without requiring an emulator.

M7B2 will add the concrete Android AppDeviceAdapter, platform permission broker and
Keystore-backed M6 approval-secret provider without changing the M6 trust contract.

M7C will add OS-lifecycle-aware continuation, background scheduling within Android limits,
hardware scheduling and battery/thermal governance.

### M8 - Interactive 3D assistant
Low-latency avatar shell, speech/vision hooks and continuity with the sovereign cognition state.

### M9 - Capability acquisition
Controlled import/adaptation pipeline that converts compatible learned capability into
VN97-native packages without silently mutating trusted runtime code.
