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
- M7B1: Android/Kotlin ↔ JNI bridge + atomic runtime checkpoint owner.

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

## Native execution libraries

- libvn97_packed_ternary.a
- libvn97_recurrent.a
- libvn97_selective.a
- libvn97_tokenizer.a
- libvn97_modality.a
- libvn97_memory.a
- libvn97_runtime.a

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
