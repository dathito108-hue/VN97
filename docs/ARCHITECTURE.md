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

100. Android approval HMAC material is generated and retained by AndroidKeyStore; the
     platform API never exports the secret key as bytes.
101. Android approval tokens preserve the exact M6 canonical HMAC-SHA256 payload fields and
     lowercase-hex signature representation, so platform realization does not create a second
     token format.
102. Android approval validity uses epoch wall-clock nanoseconds compatible with the M6 reference
     contract; prompt sessions remain bounded, expiring and process-local.
103. Raw approval-MAC access, unrestricted token minting and the one-shot coordinator are
     module-internal. Public Android code resolves only an active prompt or verifies a token.
104. Android OS permission state is a prerequisite only. A granted permission never substitutes
     for M6 policy, user approval, capability lease or immutable audit.
105. Android app/device actions preserve M6B bounds and validation: exact ASCII package IDs and
     bounded UTF-8 clipboard text.
106. On API 33+, package launch uses the package-visibility-independent launch IntentSender.
     Older Android versions use the launch-intent fallback and fail closed when no visible target
     can be resolved.
107. VN97 clipboard writes are marked sensitive to suppress unnecessary content previews where
     supported; this does not change the M6 approval requirement.
108. The Android platform adapter exposes no shell command, arbitrary Intent URI, accessibility
     action or unrestricted device-control primitive.
109. The M7B2 platform library declares no permissions itself. Host-declared permissions are
     checked by capability ID before platform execution and still remain subordinate to M6.
110. Android platform side-effect implementation objects are module-internal; trusted runtime
     assembly must place them behind the completed M6 capability/authority path.

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

M7B2 adds the AndroidKeyStore-backed M6-compatible approval controller, OS permission prerequisite
broker and internal exact-package/clipboard platform adapter. It preserves M6 request-digest
binding, one-shot approval and capability bounds without exposing a side-effect bypass.

M7C adds persisted JobScheduler continuation, cold-process VN97RUN checkpoint restore/persist handoff and
deterministic battery/thermal compute governance. Host continuation callbacks remain bounded and
cancellable and do not gain side-effect authority.

M7D adds the native VN97 language execution step: full/factorized tied embedding, per-layer
RMSNorm, VN97T2 selective projections, fused exact-ZOH recurrence, residual, final RMSNorm and
tied logits. RuntimeSession now binds executed recurrent state to a 32-byte model identity and
persists bound sessions as VN97RUN2 while retaining VN97RUN1 restore compatibility.

The remaining M7 work is Android-side activated-model loading/JNI inference wiring plus deeper
hardware-aware scheduling/telemetry tuning on real devices. The recurrent/language execution
math and checkpoint model-binding contract are now fixed.

### M8 - Interactive 3D assistant - in progress
M8A adds the permission-free OpenGL ES 3.0 avatar shell, monotonic typed visual-state bridge,
bounded animation smoothing, adaptive frame pacing and typed tap/long-press/drag interactions.
The renderer is presentation-only and cannot bypass M6 authority.

M8B adds bounded listening/speaking envelopes, PCM16 RMS metering, deterministic timed viseme
lookup, presentation-state mapping and continuous jaw/width/round mouth controls. The avatar
remains permission-free and independent from STT/TTS/cognition execution.

M8C adds the bounded VN97AV1 native avatar asset validator, Android JNI validation wrapper and
semantic rig-pose layer for richer head/jaw/arm/brow/smile animation while keeping the M8A/M8B
typed-state boundary stable.

M8 is complete at the interactive-avatar architecture contract. Higher-quality meshes, voice,
animation assets and model behavior are capability/training/package concerns rather than a new
renderer authority architecture.

### M9 - Capability acquisition - in progress
M9A adds the VN97CAP1 data-only package format, strict canonical provenance/section manifest,
multi-layer integrity validation and content-addressed atomic staging. Staging is explicitly not
activation and cannot load code, mutate M6 authority or modify live model/runtime state.

M9B adds canonical VN97SIG1 detached publisher signatures, scoped/revocable Ed25519 trust,
staged-byte revalidation, runtime compatibility profiles and deterministic typed adaptation
planning. A compatibility plan is bound to package, signature and profile fingerprints but still
cannot activate or mutate live VN97 state.

M9C adds current-trust revalidation, write-ahead transactional activation, crash recovery,
bounded per-capability rollback stacks and a provenance-visible VN97INV1 inventory. M9 is complete
at the controlled acquisition/trust/compatibility/activation architecture contract.

101. Android continuation uses JobScheduler persisted jobs and does not emulate an
    always-running process when the OS has suspended or killed the app.
102. Cold-process continuation reconstructs runtime state only from validated VN97RUN1/VN97RUN2 through
    NativeRuntimeOwner; restored sessions remain SUSPENDED until the JobService explicitly resumes.
103. Continuation work is supplied by the host Application through ContinuationWorkProvider;
    the platform library does not embed a second cognition backend or planner.
104. Battery/thermal policy is deterministic and independently testable from Android signal
    collection. Severe thermal state or critical unplugged battery blocks compute.
105. Each runnable governor profile exposes bounded token and wall-time budgets plus a retry-delay
    hint. Continuation callbacks are required to honor those ceilings and cancellation.
106. JobService stop signals are propagated through ContinuationContext; onStopJob requests OS
    rescheduling rather than pretending the interrupted wake completed.
107. A successful continuation wake suspends the runtime and atomically persists the current VN97RUN checkpoint before
    reporting completion to JobScheduler.
108. RECEIVE_BOOT_COMPLETED exists solely to support Android persisted JobScheduler continuity;
    it does not grant cognition an external-action capability.
109. The continuation JobService is non-exported and protected by BIND_JOB_SERVICE.
110. M7C background execution never bypasses M6; any external effect initiated by continued
    cognition must still pass the existing M6C/M6A authority path.

111. The avatar renderer is presentation only. It receives typed visual state and has
    no capability registry, approval authority, lease access or device-action adapter.
112. AvatarCommand sourceSequence must increase monotonically; stale commands fail closed rather
    than visually overwriting a newer cognition/runtime state.
113. Avatar animation parameters are finite and bounded before reaching the renderer; render
    interpolation never expands the trusted input domain.
114. AvatarStateBridge publishes immutable atomic snapshots so the GL render thread never reads a
    partially updated semantic state.
115. Tap, long-press and drag are emitted as typed AvatarInteraction observations only; they are
    not approvals, external action requests or implicit M6 authority.
116. M8A uses a procedural OpenGL ES 3.0 shell with no third-party 3D engine dependency.
117. Frame pacing is mode-aware and bounded from 5 to 60 FPS to reduce idle/sleeping GPU work
    while preserving low-latency active interaction.
118. Android avatar lifecycle explicitly starts/stops Choreographer scheduling with view/host
    lifecycle and uses RENDERMODE_WHEN_DIRTY rather than unconditional continuous rendering.
119. The avatar module manifest declares no permission and cannot independently access network,
    files, apps, clipboard, microphone or camera.
120. Future speech/vision/avatar features may feed typed state or interactions, but any external
    side effect must still traverse the existing M6C/M6A authority path.

121. M8 speech synchronization consumes presentation inputs only; it does not own a speech
    recognizer, speech synthesizer, microphone permission or cognition execution loop.
122. SpeechTimeline is bounded to at most 4096 sorted non-overlapping cues and at most one hour;
    invalid ordering, overlap, cue duration or time overflow fails closed.
123. Viseme playback lookup is deterministic and binary-search based; absent/out-of-range cues
    produce a REST mouth pose rather than extrapolating untrusted state.
124. Continuous jaw-open, mouth-width and lip-round parameters are finite and unit-bounded before
    entering AvatarCommand, immutable frame state or renderer smoothing.
125. Listening and speaking attack/release envelopes are mode-gated. Audio energy observed outside
    LISTENING/SPEAKING cannot pre-charge a later avatar state.
126. PCM16 level metering is a bounded pure computation over caller-owned samples and never opens
    an Android audio device or implicitly requests RECORD_AUDIO.
127. CognitionPresentationState is a presentation mapping only; it does not replace or mutate the
    M5 planner/cognition state machines.
128. SpeechAvatarSynchronizer serializes envelope/state publication and still relies on the M8A
    monotonic sourceSequence check to reject stale visual updates.
129. VN97AvatarView publishSpeech() only updates visual state and requests rendering; it does not
    turn speech/touch presentation into M6 approval or external-action authority.
130. Any future STT/TTS/audio I/O implementation must remain outside the renderer and preserve the
    existing M6 authority boundary for external side effects.

131. VN97AV1 is the trusted native avatar asset format; the core validator does not execute
    or embed third-party model/asset parsers.
132. VN97AV1 requires fixed magic/version/header size, zero reserved fields, exact contiguous
    offsets/strides/total length and both header/payload CRC32 integrity.
133. Asset geometry is bounded to 200000 vertices, 600000 triangle indices and 128 joints;
    allocation arithmetic and offsets are overflow checked before data traversal.
134. Vertex positions/normals must be finite and bounded, normals have bounded magnitude, and every
    mesh index must reference an existing vertex.
135. Rigged vertices require skin-weight sum exactly 255 and every nonzero-weight joint index must
    exist; unrigged assets require zero total skin weight.
136. Joint hierarchy is topologically ordered, has a root, validates reserved fields, finite
    transforms, near-unit quaternions and positive bounded scales.
137. Native AvatarAssetView references caller-owned bytes only during parsing/read operations;
    the Android ValidatedAvatarAsset wrapper keeps a defensive immutable byte copy after validation.
138. The avatar JNI bridge exposes validation metadata only and has no file/network/app/device
    side effect or M6 approval/policy/lease capability.
139. AvatarRigAnimator consumes immutable M8 frame state and emits only finite bounded semantic
    pose controls; it does not mutate cognition, planner or asset bytes.
140. Future avatar asset acquisition must validate VN97AV1 and use the controlled M9 import/
    provenance path; CRC32 never substitutes for package authentication or M6 authority.

141. VN97CAP1 packages are data-only containers; executable/script/plugin/native-library
    section roles are rejected before a package can be staged.
142. The VN97CAP1 fixed header/table requires canonical offsets, exact total length, zero reserved
    fields, bounded section count and one manifest section at index zero.
143. Every section is SHA-256 bound in the binary table; the package also binds the complete
    table+payload digest and a header CRC32 before manifest parsing.
144. The manifest is strict canonical UTF-8 JSON with exact keys, no duplicates/non-finite values,
    bounded identifiers and exact section role/format/size/SHA-256 binding.
145. Capability provenance records origin, source SHA-256 and license metadata, but M9A does not
    treat those fields or package CRC/hash as a publisher signature or trust decision.
146. CapabilityStager validates the entire package before persistence and uses a trusted
    non-symlink root, no-follow operations, directory locking, fsync and create-only publication.
147. Staged package identity is the full-package SHA-256 and staging the exact same package is
    idempotent; a conflicting digest path fails closed.
148. Staging never activates a capability, loads native code, registers an M6 handler, changes
    policy/approval/lease state or mutates live VN97 model/runtime state.
149. M9 activation must consume only already-validated staged packages and apply a separate typed
    compatibility/trust decision before any learned data becomes active.
150. External acquisition/download of VN97CAP1 remains an M6-authorized side effect; M9 package
    validation/staging does not create a parallel network/file authority path.


151. VN97SIG1 is a strict canonical detached-signature envelope with bounded exact fields; it
    is not interpreted until the referenced staged VN97CAP1 bytes have been revalidated.
152. The signature message is domain separated and binds algorithm, publisher key ID, full package
    SHA-256, capability ID and capability version.
153. Trusted publisher keys are Ed25519 public keys scoped by segment-aware capability namespace,
    allowed capability kind, version interval and explicit revocation state.
154. Trust verification reopens the staged digest path with no-follow semantics, reparses VN97CAP1
    and requires staged digest and manifest identity to match before accepting a signature.
155. The built-in Ed25519 verifier fails closed when its optional crypto backend is unavailable;
    an unavailable verifier is never treated as successful trust.
156. VerifiedCapability records publisher key identity and signature-envelope digest but grants no
    activation, M6 authority, handler registration or runtime mutation power.
157. CompatibilityProfile is immutable typed policy and exposes a deterministic SHA-256 fingerprint
    over profile ID, runtime API, supported kinds, version range and ordered format rules.
158. A non-direct section requires exactly one approved AdapterSpec whose input role/format and
    output accepted format match; zero matches or multiple matches fail closed.
159. Lossy adaptation is denied by default and requires an explicit caller policy decision; model
    output or package metadata cannot turn it on implicitly.
160. CompatibilityPlan binds package digest, publisher key, signature digest, profile ID/fingerprint
    and runtime API version. M9B plans are proposals only; M9C must revalidate them transactionally
    before activation.


161. Activation re-runs M9B staged-byte, detached-signature and current trust-store verification
    before reserving a transaction; stale VerifiedCapability objects do not bypass revocation or
    staged-file changes.
162. Activation recomputes the compatibility plan from fresh verified bytes/profile/adapters and
    requires exact equality with the submitted plan before any backend prepare call.
163. CapabilityActivationBackend is host-selected trusted runtime code. prepare must not mutate live
    runtime state; commit is the mutation boundary; inspect and rollback must support crash recovery.
164. VN97INV1 persists a reserved write-ahead transaction before backend prepare and persists the
    prepared backend token/artifact digest before backend commit.
165. Recovery resolves one durable pending transaction by backend inspection: committed activations
    finalize, prepared activations roll back, and already rolled-back transactions finalize/clear.
166. Inventory stores a bounded activation stack per capability so rollback removes the current
    transaction and restores the previous backend revision/provenance rather than losing history.
167. Exact already-active package/signature/profile/plan/backend activation is idempotent; same-version
    replacement and version downgrade are denied by default unless trusted host policy opts in.
168. VN97INV1 is strict canonical JSON with exact schemas, monotonic generation history, bounded
    stack/history/size, no-follow access, advisory locking, fsync and atomic same-directory replace.
169. Public InventorySnapshot exposes active provenance/trust/profile/plan/runtime revision but never
    backend transaction tokens; tokens remain internal solely for rollback/recovery.
170. M9 activation does not register M6 external capabilities, mint approval/policy/leases or create
    network/file authority. Package acquisition and other external effects remain governed by M6.


171. Native language execution follows the canonical VN97LanguageCore math and reuses VN97T2
    packed projections plus the existing fused selective exact-ZOH recurrence; no second model
    backend or alternate language architecture is introduced.
172. LanguageModelView is a non-owning trusted in-memory view and is not a package/file parser.
    External or persisted model bytes must first pass the M9 trust/activation path before a trusted
    loader constructs the view.
173. A language model carries a nonzero 32-byte model identity. RuntimeSession binds to that
    identity only after the first successful inference step.
174. Runtime language geometry must exactly match the recurrent session n_layers/d_model/d_state;
    mismatch fails before state mutation.
175. Full tied F32 embedding/head and factorized tied embedding/head are both supported; factorized
    input/output use the same token factors and projection, preserving exact tying.
176. After model binding, metadata-only RuntimeSession::Advance is rejected so sequence_position
    cannot diverge from the number of recurrent language steps actually executed.
177. VN97RUN1 remains the unbound checkpoint format and continues to restore. Model-bound sessions
    serialize as VN97RUN2 with the same state payload plus model-bound flags/identity and header CRC.
178. A restored model-bound VN97RUN2 session accepts inference only from the identical model ID.
    A legacy unbound checkpoint with nonzero sequence position or nonzero recurrent state cannot
    be rebound implicitly to an arbitrary model.
179. Language workspace is caller/runtime-owned bounded scratch; learned model memory is never
    copied into RuntimeSession and RuntimeSession never takes ownership of model-view pointers.
180. M7D adds no external side-effect authority. Model acquisition/activation remains M9-governed,
    and Android JNI/model-loader wiring must preserve the same M6/M9 boundaries.

