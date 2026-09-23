# M10F — Android Compatibility Planning + Transactional VN97INV1 Activation

M10F ports the generic M9B compatibility planner and M9C transactional inventory coordinator into
the Android runtime. It deliberately keeps concrete model-image extraction/JNI commit logic in a
separate backend milestone.

Canonical acquisition is now structurally available on Android:

`VN97CAP1 -> VN97SIG1 -> staging -> publisher trust -> compatibility plan -> transactional backend -> VN97INV1`

No model/backend selected by cognition is introduced.

## Compatibility planning

`VN97CompatibilityProfile` declares:

- canonical profile ID;
- runtime API version;
- supported capability kinds;
- unsigned-32 capability version range;
- one unique format rule per role.

The profile fingerprint matches the canonical M9B shape: ordered format rules and sorted supported
kinds under SHA-256.

Each package section becomes a `VN97SectionPlan`.

- accepted source format -> `DIRECT`;
- otherwise exactly one adapter must match role/input and produce an accepted target;
- zero adapters fail closed;
- multiple adapters fail as ambiguous;
- lossy adapters require explicit `allowLossy=true`.

`VN97CompatibilityPlan.sha256()` hashes the canonical M9C plan identity object.

## VN97INV1 store

`VN97CapabilityInventoryStore` persists the same canonical `VN97INV1` shape consumed by M10B:

- monotonic generation;
- sorted capability stacks;
- maximum 64 committed entries per capability;
- maximum 4096 history events;
- one optional write-ahead transaction;
- backend token retained only in internal records for rollback/recovery.

The store uses an app-private lock file, strict canonical UTF-8 JSON, bounded reads, symlink
rejection, fsync, atomic replace and directory fsync.

Public snapshots omit backend tokens.

## Transaction coordinator

`VN97CapabilityActivationCoordinator.activate(...)` revalidates trust at activation time and
recomputes the compatibility plan before backend preparation.

Write-ahead activation:

1. persist `reserved`;
2. backend `prepare`;
3. persist `prepared` token + artifact SHA-256;
4. backend `commit`;
5. atomically append provenance to VN97INV1 and clear pending.

Exact already-active package/signature/profile/plan/backend is idempotent. Same-version replacement
and downgrade are denied unless trusted host code opts in.

Rollback persists a prepared rollback transaction before invoking the backend, then pops the exact
active stack record after backend rollback.

## Crash recovery

`recover(...)` implements M9C semantics:

- reserved activation -> clear safely because no backend token was returned;
- prepared + COMMITTED -> finalize provenance;
- prepared + PREPARED -> rollback then clear;
- prepared + ROLLED_BACK -> clear;
- pending rollback -> ensure backend rollback then pop the exact activation.

Missing recovery backend fails closed and leaves the transaction durable.

## Backend boundary

M10F defines `VN97CapabilityActivationBackend` with:

- `prepare(verified, plan)`;
- `commit(token)`;
- `inspect(token)`;
- `rollback(token)`.

The backend is supplied by trusted host/runtime code, never cognition.

M10F does **not** yet implement the production `vn97.model_image` backend. Therefore this
milestone by itself still cannot provision a live Android model; M10G will bind the direct
`model_image/VN97MI1` section to app-private artifacts and native validation.

## Verification

Isolated `kotlinc -Werror` fault-injection gate:

`M10F_TRANSACTIONAL_ACTIVATION_PASS`

Coverage:

- direct compatibility;
- explicit adapter path;
- missing/lossy adapter denial;
- first activation;
- exact idempotent re-activation;
- v1 -> v2 upgrade;
- v2 -> v1 rollback;
- same-version replacement denial;
- process-death simulation after backend commit and recovery/finalization;
- process-death simulation during reserved prepare and safe recovery.

The production M10F sources are tested independently from M10D/M10E with narrow contract stubs;
the next concrete-backend milestone must add an end-to-end real staging/trust/activation gate.
