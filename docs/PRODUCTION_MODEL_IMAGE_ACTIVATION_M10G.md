# M10G — Production VN97MI1 Model-Image Activation Backend

M10G supplies the first concrete trusted-host backend for the generic M10F transaction
coordinator.

The production Android path is now structurally:

`VN97CAP1 -> VN97SIG1 -> stage -> publisher trust -> DIRECT model-image plan -> native-validated VN97MI1 backend -> VN97INV1 -> M10B NativeActivatedModel`

No model or backend is selected by cognition.

## Direct-only contract

`VN97ModelImageActivationBackend` accepts only:

- capability `model.language`;
- kind `weights`;
- exactly one package section;
- role `model_image`;
- format `VN97MI1`;
- `DIRECT` compatibility;
- no adapter;
- no lossy conversion.

The production profile is `vn97.android.model-image.v1`.

## Prepare

Prepare reopens the freshly trusted staged package, copies exactly the manifest-proven
`packageOffset/size` range into a transaction candidate and recomputes SHA-256.

The digest must equal the section digest.

The candidate is then passed to `VN97ModelImageCandidateValidator`. On Android the validator
opens the candidate read-only through `ParcelFileDescriptor` and calls
`NativeModelImageCandidateValidator`, which uses the existing native VN97MI1 loader and expected
SHA-256. The native handle is destroyed immediately; candidate validation never returns an active
model and grants no trust.

Only a native-valid candidate is returned as M10F PREPARED.

## Commit / inspect / rollback

Backend transaction state is app-private and restart-readable:

- `<token>.prepared.vn97mi1`;
- `<token>.committed`;
- `<token>.rolledback`.

Commit revalidates the candidate natively, hard-links it to:

`vn97-capabilities/artifacts/<artifact_sha256>.vn97mi1`

then fsyncs the artifact directory and writes an atomic committed marker. The runtime revision is
`mi1-<artifact_sha256>`.

A process restart can inspect PREPARED or COMMITTED from filesystem state alone.

Rollback is idempotent and writes a durable rolled-back marker. A committed artifact is retained
as inert content-addressed cache; VN97INV1 remains the sole active-capability selector, so an
unreferenced artifact cannot become active.

## Android provisioning root

`AndroidVN97CapabilityProvisioner` binds all production paths under:

`noBackupFilesDir/vn97-capabilities`

with:

- `stage/` for M10D content-addressed VN97CAP1/VN97SIG1;
- `artifacts/` for committed VN97MI1;
- `inventory.vn97inv1.json` for M10F active provenance;
- `.model-image-transactions/` for backend crash recovery.

This is exactly the root already consumed by the M10B activated-model loader.

## Verification

Actual isolated gates run before branch upload:

- `M10G_MODEL_IMAGE_BACKEND_PASS`
- `M10G_NATIVE_CANDIDATE_VALIDATOR_SYNTAX_PASS`
- `M10G_ANDROID_VALIDATOR_SYNTAX_PASS`

The backend behavior gate covers section extraction/hash, validator invocation, PREPARED restart,
commit, COMMITTED restart, idempotent commit, durable rollback, DIRECT-only rejection, capability
mismatch, staged-package tampering and confirms the backend does not write VN97INV1 itself.

Full JNI validation of a real VN97MI1 image and full Android Gradle/device instrumentation remain
device/build gates; the syntax stubs are not claimed as those gates.
