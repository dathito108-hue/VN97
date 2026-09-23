# M10B — Trusted Model Inventory → Real App Assistant Session

M10B connects the installable M10A app shell to the existing canonical VN97 production assistant
without exposing a raw "open any model file" path.

## Trust boundary

Android accepts a model only through a committed M9 inventory:

`VN97INV1 active record -> app-private artifact SHA-256 -> TrustedActivatedModelArtifact -> NativeActivatedModel`

The app never constructs `TrustedActivatedModelArtifact` directly.

The runtime reads `inventory.vn97inv1.json` with strict UTF-8/canonical JSON validation and
requires:

- schema `VN97INV1`;
- no unresolved `pending` activation transaction;
- bounded/sorted activation stacks;
- valid hashes, IDs, tokens and history generations;
- active capability `model.language`;
- backend `vn97.model_image`.

The active record's `artifact_sha256` selects exactly:

`<inventory-root>/artifacts/<sha256>.vn97mi1`

The loader hashes a duplicate descriptor of the exact opened file, compares that digest to the
inventory, and only then passes the descriptor through the existing internal M9 activated-model
bridge. The native loader independently verifies model identity again.

## App lifecycle

`VN97Application` owns one process-scoped `VN97AppAssistant`.

On launch:

1. M10A remains `MODEL_REQUIRED`;
2. the app checks the app-private M9 inventory root;
3. missing active model keeps input disabled;
4. corrupt/mismatched evidence moves the app to ERROR;
5. valid activation opens one native VN97 model and one M7 memory-backed production assistant;
6. only then does the UI enter READY.

Send is executed off the UI thread. YIELDED cognition is advanced in a bounded loop. Completed
turns return through the existing M7X/M7Y memory write-back path. Approval-required turns stop at
the existing M6 boundary; M10B does not auto-approve them.

M10B intentionally supplies an empty policy-grant list. No app/device external effect is enabled
by default. A later UI authority milestone must surface explicit user policy/approval controls.

## No fallback

There is still no Transformer/LLaMA/cloud/parallel model path. If M9 evidence or VN97MI1 is absent,
chat stays disabled.

## Verification

Host parser regression:

`M10B_INVENTORY_EVIDENCE_PASS`

The Android descriptor/native loading path still requires Android build/device instrumentation;
host parser success is not reported as APK/device PASS.
