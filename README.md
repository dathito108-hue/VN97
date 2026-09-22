# VN97

VN97 is a mobile-first sovereign intelligence project built around a recurrent selective
state-space core, physically packed ternary execution, a lossless native tokenizer, compact
multimodal frontends and explicit sovereign memory.

## Completed foundations

- M0: stable selective SSM reference with constant recurrent state;
- M1: physical VN97T2 2-bit ternary weights, scalar kernel, ARM64 NEON backend and dispatch;
- M2A: associative parallel affine scan for full-sequence training/reference execution;
- M2B: native fused recurrent state update/readout for prefill and token-step;
- M2C: native fused selective ZOH dynamics without expanded decay/drive tensors;
- M3A: VN97TK1 lossless tokenizer + native codec + optional factorized tied embeddings;
- M3B: audio/vision modality adapters + native preprocessing + dense embedding ingress;
- M4A: VN97MEM1 sovereign memory journal + bounded working memory + deterministic retrieval;
- M4B: mmap native memory store + incremental zero-copy index + native append/retrieve/compact;
- M5A: bounded reasoning/planning state machine + VN97PLN1 safe checkpoints;
- M5B: typed cognition backend loop + bounded retrieval/verification/refinement orchestration;
- M5C: VN97-owned tokenizer/model cognition adapter + strict structured inference contract;
- M6A: typed capability registry + deny-by-default authority/approval/lease/audit fabric;
- M6B: production capability pack for confined files, sovereign HTTPS and platform actions;
- M6C: strict external-intent binding + one-shot exact-request approval sessions;
- M7A: native runtime session ABI + VN97RUN1 safe checkpoint/restore;
- M7B1: Android/Kotlin ↔ JNI bridge + atomic runtime checkpoint owner;
- M7B2: Android Keystore approval authority + permission-gated app/clipboard platform adapter.

## M3 multimodal contract

Text remains VN97TK1 token IDs. Audio and vision do not enter the recurrent core as raw
waveform/pixels. They are converted to compact frame/patch feature rows, projected with the same
ternary projection machinery used elsewhere, normalized to d_model, prefixed by the exact
reserved modality embedding, then passed through VN97LanguageCore.forward_embeddings().

Default reference profiles:

- audio: mono PCM, 320-sample frame / 320-sample hop;
- vision: RGB NCHW, non-overlap 16x16 patches;
- vision patch features include explicit normalized y/x coordinates.

The native runtime provides equivalent audio framing and vision patch extraction in
libvn97_modality.a. Projection weights can reuse VN97T2 packed matvec rather than introducing a
second modality-specific weight format.

## M4 sovereign memory contract

Recurrent state is not used as a substitute for persistent memory. M4 separates bounded working
memory from append-only episodic/semantic VN97MEM1 storage.

VN97MEM1 records carry stable IDs, timestamp, importance, source provenance, optional retrieval
vectors, parent ancestry and SHA-256 content identity. Frames are CRC32 protected. The native
MemoryStore adds exclusive single-writer locking, mmap-backed records, incremental suffix
indexing, exact deterministic retrieval, torn-tail recovery and atomic provenance-preserving
compaction without duplicating journal vectors into a second resident vector table.

## M5A reasoning/planning contract

PlanController provides a deterministic bounded state machine above model inference:

- ordered dependency graph with stable step IDs;
- stable plan ID derived from canonical goal + step graph + budget;
- bounded transitions, retries, memory queries and retrieved hits;
- confidence thresholds and explicit verification state;
- interruption/pause that requeues unfinished internal reasoning;
- VN97MEM1 retrieval context with stable evidence record IDs;
- EXTERNAL steps stop at WAITING_EXTERNAL instead of executing a side effect.

VN97PLN1 checkpoints store the immutable plan definition plus runtime lifecycle at safe points.
A checkpoint cannot contain a RUNNING internal step. Payload bytes are canonical JSON protected
by SHA-256 and saved with atomic replace/fsync. The digest is an integrity mechanism, not an
authority grant or author signature.

M5A does not execute tools or device actions. M6 remains the authority/capability boundary.

## M5B cognition loop contract

CognitionLoop connects M5A to a typed CognitionBackend without giving the backend direct access
to planner state mutation or external side effects. The backend may propose an immutable plan,
a retrieval vector, an internal step result and a verification decision.

The loop validates plan size, external-step count, UTF-8 context/result budgets, retrieval vector
shape and response schema. VN97MEM1 evidence IDs are attached by the controller from actual
retrieval results and propagated through dependencies; the cognition backend cannot invent them.

Backend exceptions consume the existing bounded retry policy. Schema/shape/size violations fail
closed. EXTERNAL steps still stop at WAITING_EXTERNAL before any backend/tool execution.

Plan refinement is allowed only before execution begins. A refinement creates a new immutable
plan/plan_id rather than mutating the existing graph.

M5B is an orchestration contract. Production model-quality planning/verification still depends
on a VN97 cognition adapter and trained/native model capability.

## M5C VN97 cognition adapter

VN97CognitionAdapter implements the M5B CognitionBackend contract through a VN97InferenceEngine.
The reference TorchVN97InferenceEngine uses VN97TK1 plus VN97LanguageCore directly; no external
LLM/backend is introduced.

Cognition operations use a bounded VN97COG1 prompt envelope and must return exactly one strict
JSON object. Markdown/prose wrappers, duplicate keys, missing/extra fields, invalid numeric
values and unknown enums fail closed.

For memory retrieval, the model emits retrieval text/weights rather than an arbitrary embedding
array. The reference engine derives the vector locally from the final normalized VN97 hidden
state and requires VN97MEM1.vector_dim == d_model.

M5 is now complete at the cognition-adapter contract. This does not claim trained AGI-level
reasoning quality: capability depends on trained/imported VN97-native weights. Full C++ Android
model execution remains an M7 runtime concern; external side effects still remain behind M6.

## M6A tool + authority fabric

M6A connects the existing M5 `WAITING_EXTERNAL` boundary to a sealed typed capability registry
without giving cognition direct tool authority.

An `ExternalActionRequest` binds the immutable plan/step objective to an exact capability,
canonical scope and canonical JSON payload. Unknown capabilities fail closed. Explicit
`PolicyGrant` values match the exact principal + capability + scope digest; there is no wildcard
fallback.

Capabilities that require approval use HMAC-SHA256 `ApprovalToken` values bound to the exact
request digest and validity window. The authority gate issues process-local capability leases
bounded by scope, lifetime and use count. A lease not issued by the active gate, or one that is
expired/exhausted/mismatched, is rejected.

`ExternalExecutionFabric` is the only M6A path from `WAITING_EXTERNAL` to a registered handler.
It verifies the planner step, registry schema, policy, approval and lease before invoking the
side effect. Denial leaves the planner waiting; handler failure is auditable and fails closed.

Every authorization decision produces an immutable `ActionReceipt`. Optional local persistence
uses append-only, fsynced canonical JSONL with a SHA-256 receipt hash chain. A successful receipt
is written before the planner receives the result; replay of the exact request digest therefore
returns the receipted result without executing the side effect twice after a crash/restart.

The receipt hash chain is an integrity/idempotency mechanism, not an author signature. Approval
authenticity comes from the trusted HMAC secret and policy boundary. M6A adds no third-party AI
or service dependency. Production internet/file/app/device capability implementations remain
later M6 work.

## M6B production capability pack

M6B supplies concrete platform-neutral capabilities behind the sealed M6A registry:

- `file.read`: strict UTF-8 reads from configured trusted roots;
- `file.write`: create-only or SHA-256 compare-and-swap replace with fsync/atomic commit;
- `web.fetch`: HTTPS-only bounded text/JSON GET with DNS/IP SSRF protection and pinned TLS target;
- `app.launch`: exact Android-style package launch through an injected platform adapter;
- `device.clipboard.write`: bounded system clipboard write through the same platform adapter.

File paths are canonical relative POSIX paths. Each directory component and target is opened with
no-follow semantics on Linux/Android, so traversal and symlink escape fail closed. Writes use an
exclusive VN97 directory lock, fsynced temporary files and atomic publication; replacing an
existing file requires its current SHA-256 digest.

The HTTPS fetcher accepts only port 443, forbids credentials/fragments/redirects, requires every
resolved address to be globally routable, then connects to one validated IP while retaining the
original hostname for TLS SNI/certificate verification. Responses are bounded and limited to
strict text/JSON.

Every M6B descriptor requires explicit M6A approval by default. Registration provides handlers
only; it does not expose registry mutation, policy, approval secrets or lease controls to
cognition. Android app/clipboard behavior is injected through `AppDeviceAdapter`; the concrete
Android implementation remains an M7 responsibility. No third-party AI or inference service is
introduced.

## M6C external intent + approval UX contract

M6C connects the M5 `WAITING_EXTERNAL` boundary to M6A requests without turning model output
into authority. `VN97CognitionAdapter.propose_external_intent()` may propose only a capability
ID, string-valued scope and JSON payload from a trusted sealed capability catalog.

`ExternalIntentBinder` rebuilds the final `ExternalActionRequest` from the active
`PlanController`: plan ID, step ID and objective are never copied from model output. The binder
revalidates the selected capability against the sealed registry and fails closed if the waiting
step changes while cognition is producing the intent.

`ExternalApprovalCoordinator` creates deterministic presentation data for the exact request
digest and manages bounded, expiring, thread-safe, one-shot approval sessions. A UI denial,
expiry, stale/tampered prompt or second concurrent resolution cannot mint an approval token.
Only an explicit trusted UI resolution calls `ApprovalAuthority.approve()`; cognition never
receives policy grants, HMAC secrets, approval-token minting or lease mutation APIs.

M6 is complete at the platform-neutral tool/authority contract. Android UI presentation,
Keystore-backed approval secrets and concrete platform lifecycle/permission integration belong
to M7 and must preserve these M6 invariants.

## M7A native runtime session ABI

M7A introduces the native session boundary that Android/JNI will wrap. JNI-facing code uses an
opaque nonzero `uint64_t` handle registry rather than exporting raw C++ pointers. Registry
lookups retain a `shared_ptr` for the duration of each call, so destroying a handle invalidates
future calls without freeing a session that is still in use by an in-flight operation.

A runtime session owns the recurrent state shape
`[layers, batch, d_model, d_state]`, resolved recurrent/packed backends and a monotonic sequence
position. State size multiplication is overflow checked and the reference runtime caps recurrent
state storage at 512 MiB. Imported state must be finite and exact-sized.

Lifecycle is explicit: `CREATED → ACTIVE ↔ SUSPENDED`. State mutation from the host and
checkpoint creation are prohibited while ACTIVE. A restored session always returns SUSPENDED;
restore never silently resumes execution.

`VN97RUN1` is a versioned little-endian checkpoint containing exact shape, requested backend
profile, sequence position and the full recurrent state. Header and payload CRC32 values detect
corruption and exact blob length/state count are validated. The checkpoint CRC is an integrity
mechanism, not a cryptographic author signature or M6 authority grant.

AUTO backend selection resolves through the existing VN97 recurrent and packed-ternary backend
dispatch. An explicitly requested backend that is unavailable on the current device fails closed
both on create and restore.

M7A intentionally does not implement Android UI, permissions, Keystore, scheduling or background
services. M7B will wrap this stable ABI with JNI/platform adapters while preserving the completed
M6 authority path.

## M7B1 Android JNI runtime bridge

M7B1 adds a permission-free Android library module around the fixed M7A runtime ABI. Kotlin owns
only an opaque nonzero `Long` handle; JNI translates typed arrays/status values to the M7A C ABI
and never exposes a native pointer to managed code. Kotlin serializes handle use against
`close()`, while the M7A registry independently retains native lifetime for an in-flight call.

`NativeRuntimeSession` exposes typed create/restore/info/state/lifecycle/checkpoint operations.
Native status codes are mapped to explicit `NativeRuntimeStatus` values and failures throw a
typed `NativeRuntimeException`. A native handle that cannot fit in signed JVM `Long` fails
closed instead of wrapping negative.

`AtomicCheckpointStore` is for trusted app-internal VN97 runtime state, not arbitrary user file
access. It rejects symlink roots/targets, bounds checkpoint bytes, writes a same-directory
temporary file, fsyncs file data, requires atomic replace and fsyncs the parent directory through
JNI. `NativeRuntimeOwner` restores a checkpoint only if its exact runtime configuration matches
the requested session, and persistence happens only at the M7A SUSPENDED safe boundary.

The Android runtime module declares no permissions and contains no internet/app/device-action
handler. M6 remains the sole authority path for external side effects. M7B2 will add the Android
Keystore-backed approval-secret provider, permission broker and concrete M6B `AppDeviceAdapter`
behind that existing authority boundary.

A reproducible host regression under `android/runtime/host-test/run.sh` compiles the JNI bridge
against the real M7A native sources and runs the Kotlin lifecycle/checkpoint flow on a JVM without
requiring an Android emulator.

## M7B2 Android authority + platform adapter

M7B2 realizes the completed M6 approval and app/device contracts on Android without creating a
second authority model.

`AndroidApprovalController` owns a bounded one-shot approval coordinator. The HMAC key is created
inside `AndroidKeyStore` as HmacSHA256 and is never exported as raw bytes. Approval tokens keep
the exact M6 field semantics and canonical signing payload:
`approval_id, expires_ns, issued_ns, issuer, principal, request_digest`.
Android uses epoch nanoseconds derived from wall-clock time so validity windows are compatible
with the M6 reference contract.

The public Android API can create/resolve a pending approval prompt and verify a token, but the
raw MAC provider, approval authority and coordinator are module-internal. Denial, expiry,
tampered prompts and concurrent duplicate resolution cannot mint more than one token.

`AndroidPermissionBroker` treats Android runtime permissions only as an OS prerequisite.
Permission state does not grant M6 authority, create an approval or issue a lease.

The concrete app/device implementation remains internal to `AndroidPlatformRuntime`:
- API 33+ app launch uses `getLaunchIntentSenderForPackage()`;
- API 26-32 uses the package manager launch-intent fallback and fails closed if package visibility
  prevents resolving the target;
- clipboard writes are bounded to the M6B 16 KiB UTF-8 limit and are marked sensitive;
- there is no shell, arbitrary Intent, accessibility or unrestricted device-control primitive.

The `:platform` manifest requests no permission by itself. A host may configure capability-to-
permission requirements, but M6 policy + explicit approval + lease remain mandatory before a
side-effect handler is invoked.

## M7D native language inference execution

M7D closes the gap between the native recurrent/checkpoint runtime and the canonical Python
language core. `libvn97_language.a` executes one recurrent token step using the same model math:

`tied embedding → per-layer RMSNorm → VN97T2 in/dt/B/C projections → exact-ZOH fused selective
SSM → VN97T2 out projection → residual → final RMSNorm → tied LM head`.

Both full F32 tied embeddings and the M3A factorized tied embedding/head are supported. The
factorized path computes token factors × projection on input and hidden × projectionᵀ × token
factorsᵀ on output, preserving exact weight tying.

All learned linear projections inside the SSM remain VN97T2 and use the already-resolved packed
backend. Recurrent dynamics use the existing fused selective backend. M7D therefore does not
introduce a second language implementation or external model backend.

`LanguageModelView` is a non-owning trusted in-memory view. It deliberately is **not** a new
model-file/package format. Its 32-byte `model_id` is the trusted identity of the activated model
artifact (normally its SHA-256 or an equivalent trusted digest). A later Android loader/JNI block
will construct this view from activated package storage.

`RuntimeSession::InferStep()` is allowed only while ACTIVE and only when language-model
dimensions match the runtime state geometry. The first successful inference binds the session to
the 32-byte model identity. Later inference with another identity fails closed before recurrent
state mutation.

Once a session is model-bound, the old metadata-only `Advance()` path is rejected so sequence
position cannot advance without the recurrent state actually executing a token.

Checkpoint compatibility is versioned:

- unbound sessions continue to write/restore the original 64-byte `VN97RUN1`;
- model-bound sessions write `VN97RUN2`, which adds flags + the 32-byte model identity and a
  header CRC while preserving shape/backend/sequence/state payload semantics;
- VN97RUN1 restore remains supported;
- a restored unbound legacy checkpoint with nonzero sequence/state cannot be attached to an
  arbitrary model by inference.

The Android Kotlin runtime already recognizes the new native status codes and continues to treat
checkpoint bytes opaquely, so VN97RUN2 persistence does not change AtomicCheckpointStore.

## M7C Android continuity + compute governance

M7C adds OS-lifecycle-aware continuation using Android JobScheduler. Jobs are persisted across
process death and reboot through Android's scheduler contract. The library JobService recreates
the runtime from a validated VN97RUN1/VN97RUN2 checkpoint, resumes only at a safe suspended boundary, invokes a host-provided
`ContinuationWork`, then suspends and atomically persists state again.

The host Application implements `ContinuationWorkProvider`; this keeps task-specific cognition
outside the platform library while still allowing cold-process reconstruction.

`AndroidComputeGovernor` maps battery/charging and thermal signals into one of four bounded
profiles: BLOCKED, LOW_POWER, BALANCED or PERFORMANCE. Each profile carries token/run-time and
retry-delay ceilings. Severe thermal state or critical unplugged battery blocks computation and
asks JobScheduler to reschedule.

The continuation service itself owns no M6 capability authority. App/device/network/file side
effects still require M6C exact intent/approval and M6A policy/lease/audit.

## M8A interactive 3D avatar foundation

M8A adds a permission-free Android `:avatar` module built directly on OpenGL ES 3.0. It has no
Unity/Filament/third-party renderer dependency and no capability/tool access.

The visual contract is typed and one-way:

`VN97 cognition/runtime → AvatarCommand → AvatarStateBridge → AvatarRenderer`

Commands carry a strictly increasing source sequence plus bounded mode/gesture/speaking/blink/
gaze/energy values. Stale state and non-finite animation inputs fail closed. Rendering reads an
immutable atomic snapshot and applies bounded temporal smoothing.

The reference renderer is a procedural low-poly 3D shell with head/body/eyes/mouth/arms,
mode-specific visual accents, speaking-mouth motion, blink/gaze motion and bounded gestures.
It uses one compact cube mesh/VBO and model transforms instead of shipping external 3D assets.

`VN97AvatarView` converts tap/long-press/drag into typed `AvatarInteraction` events only.
Interactions are observations for the host; they are not M6 approvals, capability requests or
device actions.

Frame pacing is adaptive: sleeping 5 FPS, idle 15 FPS, thinking/error 30 FPS and active
listening/speaking/approval/execution 60 FPS. This keeps active interaction responsive without
forcing permanent 60 FPS rendering while idle.

M8A also repairs the pre-existing Android settings file that contained a literal `\n` between
module includes, then registers `:runtime`, `:platform` and `:avatar` as separate modules.

## M8B speech + viseme synchronization

M8B connects presentation-level speech timing to the M8A avatar without embedding an STT/TTS
engine or requesting microphone permission.

`SpeechAvatarSynchronizer` accepts bounded `SpeechAvatarInput` values and maps a small
presentation-only cognition state into the existing `AssistantMode`. Listening and speaking
levels use independent attack/release envelopes. Audio levels are gated by mode, so microphone
or playback energy observed while VN97 is not LISTENING/SPEAKING cannot pre-charge a later
animation.

`SpeechTimeline` stores at most 4096 sorted, non-overlapping `VisemeCue` entries within a
bounded timeline. Playback lookup is binary-search based. Eleven generic VN97 visemes map to
continuous jaw-open, mouth-width and lip-round parameters, which are then scaled by the current
speaking envelope and smoothed by the existing avatar motion filter.

`AudioLevelMeter.rmsPcm16()` provides a sovereign PCM16 RMS meter for bounded windows without
a DSP dependency. It does not capture audio; callers supply PCM that they already own.

M8B extends `AvatarCommand` and immutable frame/render state with listening level and continuous
mouth parameters. `AvatarRenderer` uses those values for attentive head lift and viseme-aware
mouth geometry. `VN97AvatarView.publishSpeech()` is the typed Android entry point.

The avatar module remains permission-free. Actual microphone capture, speech recognition,
speech synthesis and voice-model quality are separate runtime/capability concerns and may not
bypass M6 authority or the M8 typed presentation boundary.

## M8C native avatar asset + rig contract

M8C adds the sovereign `VN97AV1` avatar asset format and native validator. The trusted runtime
does not embed glTF/Unity/Filament or another third-party asset parser.

`VN97AV1` uses a fixed little-endian header, exact contiguous vertex/index/joint sections,
header and payload CRC32, bounded counts/strides and exact total length. The native parser validates
finite geometry, normal magnitude, index ranges, skin weights, topologically ordered joint
parents, normalized joint quaternions and positive bounded scale before exposing a view.

The Android `NativeAvatarAsset` wrapper validates a bounded byte array through JNI and returns
metadata plus a defensive immutable copy. JNI performs validation only; it owns no file/network
or authority operation.

`AvatarRigAnimator` maps the existing M8A/M8B immutable frame state into bounded semantic rig
controls for head pitch/yaw, jaw, arm roll, brow raise, smile, mouth width and lip rounding.
This keeps asset/rig detail below the stable typed avatar-state boundary.

CRC32 is corruption detection only. External asset acquisition/authentication belongs to the
controlled M9 import pipeline and must still pass the existing M6 authority boundary for any
external side effects.

## M9A controlled capability package + staging

M9A introduces `VN97CAP1`, a data-only capability container for bringing compatible learned
assets into VN97 without silently mutating trusted runtime code.

A package contains one canonical strict-JSON manifest followed by one or more data sections.
The fixed binary header/table binds exact section offsets, sizes and SHA-256 digests; the
package header also carries CRC32 corruption detection plus a SHA-256 digest over the section
table and payload. The full package SHA-256 becomes its content-addressed identity.

The manifest binds capability ID/version/kind, source provenance metadata and each section's
role/format/size/SHA-256. Roles that imply executable code, plugins, scripts or native/shared
libraries are rejected by the M9A contract.

`CapabilityStager` validates the complete package before writing anything. Staging uses a
trusted non-symlink directory, directory locking, no-follow opens, same-directory temporary
files, fsync and create-only hard-link publication to
`<package-sha256>.vn97cap1`. Re-staging the same package is idempotent.

Staging is deliberately **not activation**: M9A does not register handlers, load native code,
change model weights in-place, mutate M6 policy or make the package executable. Source hashes
and provenance fields are metadata/integrity claims, not cryptographic publisher trust.
Signature/trust policy and typed activation belong to later M9 milestones.

## M9B publisher trust + compatibility planning

M9B adds a second gate after M9A staging. A staged package is re-opened with no-follow semantics,
re-parsed as VN97CAP1 and checked against its staged digest/manifest before publisher trust is
evaluated.

`VN97SIG1` is a bounded canonical JSON detached signature envelope. Its domain-separated
signing message binds algorithm, publisher key ID, full VN97CAP1 package SHA-256, capability ID
and capability version. `CapabilityTrustStore` scopes each Ed25519 publisher key by canonical
capability namespace, capability kind, version range and revocation state. Namespace matching is
segment-aware: a trust scope for `vision` permits `vision.edge` but not `visionevil`.

`Ed25519Verifier` is the production adapter and loads the optional crypto backend lazily. If the
backend is absent, verification fails closed rather than silently accepting an unsigned package.

Only a successfully revalidated and signature-verified package becomes a `VerifiedCapability`.
That object still does not activate anything.

Compatibility is a separate pure planning step. `CompatibilityProfile` declares supported kinds,
version range and accepted formats per section role. `AdapterSpec` may provide one explicit
format conversion path. Direct sections remain unchanged; incompatible sections require exactly
one matching adapter. Ambiguous paths fail closed and lossy adaptation is denied by default.

`CompatibilityPlan` binds the package digest, publisher key ID, signature-envelope digest,
runtime profile ID/fingerprint and runtime API version. This prevents M9C from activating a plan
against different trust evidence or a silently changed compatibility profile.

M9B still performs no activation, rollback, runtime mutation, M6 handler registration or policy
change. Those transactional state changes remain M9C work.

## M9C transactional activation + provenance inventory

M9C is the first milestone that can make an imported capability active. Activation is deliberately
host-controlled and transactional; cognition/model output cannot call a backend, mint an
activation transaction or bypass M6.

Before a transaction is reserved, M9C re-runs M9B staged-byte and publisher-signature verification
against the **current** stage root, signature envelope and trust store, then recomputes the exact
compatibility plan. A revoked publisher key, changed staged file, stale plan or silently changed
profile therefore fails before backend `prepare()`.

`CapabilityActivationBackend` is a trusted host/runtime interface with four operations:
`prepare`, `commit`, `inspect` and idempotent `rollback`. Prepare must not mutate the live
runtime. M9C persists a write-ahead `reserved` record before prepare, then persists backend token
and artifact SHA-256 as `prepared` before commit. Commit returns a bounded runtime revision.

`VN97INV1` is a canonical crash-recoverable inventory. It stores a bounded per-capability
activation stack, provenance/trust/profile/plan fingerprints, generation history and at most one
pending transaction. Backend transaction tokens remain internal and are not exposed through the
public `InventorySnapshot`.

Crash recovery queries the trusted backend with the persisted token. A committed activation is
finalized into inventory; an uncommitted prepared activation is rolled back; an already-rolled-back
transaction is cleared/finalized. Rollback pops the current activation stack entry, so rolling back
v2 restores the prior v1 activation instead of merely marking v2 inactive.

Exact re-activation of the already-active package/plan/backend is idempotent. Same-version
replacement and downgrades are denied by default and require explicit trusted caller policy.

Inventory persistence uses an existing non-symlink root, no-follow file access, advisory locking,
canonical JSON, same-directory temporary files, fsync and atomic replace. Reference bounds are
64 activation records per capability, 4096 history events and 4 MiB inventory size.

M9C does not register M6 external handlers or grant external authority. Acquiring packages remains
an M6-governed side effect; activation only changes a trusted capability backend through this
transaction protocol.

## Native execution libraries

- libvn97_packed_ternary.a
- libvn97_recurrent.a
- libvn97_selective.a
- libvn97_tokenizer.a
- libvn97_language.a
- libvn97_modality.a
- libvn97_memory.a
- libvn97_runtime.a
- libvn97_avatar_asset.a

See:

- docs/ARCHITECTURE.md
- docs/TOKENIZER_V1.md
- docs/EMBEDDING_COMPRESSION_M3A.md
- docs/MODALITY_ADAPTERS_M3B.md
- docs/MEMORY_V1.md
- docs/PLANNER_M5A.md
- docs/COGNITION_M5B.md
- docs/COGNITION_ADAPTER_M5C.md
- docs/AUTHORITY_M6A.md
- docs/CAPABILITIES_M6B.md
- docs/EXTERNAL_INTENT_M6C.md
- docs/RUNTIME_M7A.md
- docs/RUNTIME_M7B1.md
- docs/RUNTIME_M7B2.md
- docs/NATIVE_LANGUAGE_M7D.md
- docs/NATIVE_SELECTIVE_M2C.md

## Local verification

    python -m pip install -e '.[dev]'
    pytest

    cmake -S native -B native/build
    cmake --build native/build
    ctest --test-dir native/build --output-on-failure

Production-only native build:

    cmake -S native -B native/build-prod -DVN97_BUILD_TESTS=OFF
    cmake --build native/build-prod

Android M6 approval compatibility regression:

    android/platform/host-test/run.sh

Avatar integration:

- android/avatar — OpenGL ES 3.0 avatar shell, typed state bridge and interactions
- android/avatar/host-test — deterministic state/filter/frame-policy regression

- docs/AVATAR_M8A.md
- docs/SPEECH_AVATAR_M8B.md
- docs/AVATAR_ASSET_M8C.md
- docs/CAPABILITY_PACKAGE_M9A.md
- docs/CAPABILITY_TRUST_M9B.md
- docs/CAPABILITY_ACTIVATION_M9C.md
