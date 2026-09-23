# M10H — Trusted Capability Import + Explicit Model Activation

M10H closes the user-facing gap between the Android M10D–M10G provisioning runtime and the
installable M10 app.

The canonical path is now:

`user-selected VN97CAP1 + VN97SIG1 + publisher Ed25519 public key`
`-> M10D content-addressed stage`
`-> M10E publisher verification`
`-> M10F compatibility plan`
`-> explicit Review UI`
`-> explicit Trust & Activate`
`-> M10F crash-safe activation transaction`
`-> M10G native-validated VN97MI1 artifact`
`-> VN97INV1`
`-> M10B NativeActivatedModel + production assistant`

## No self-trust

A package never supplies its own trusted public key.

The user selects the publisher Ed25519 public key independently. M10H accepts either exactly 32 raw
bytes or exactly 64 lowercase hexadecimal characters.

For each review, M10H constructs a narrowly scoped trust policy:

- signature key ID must match VN97SIG1;
- capability namespace is only `model` / `model.*`;
- kind is only `weights`;
- capability version is 1 through unsigned-32 max;
- no revoked key.

The public key itself is not written into VN97INV1 and no new M6 grant is created.

The explicit Trust & Activate action also enrolls the exact key_id/public-key binding in the
app-private canonical `VN97PTR1` registry. Review does not mutate that registry. Reusing the same
key_id with different key bytes fails closed; key rotation therefore requires a distinct key ID.
The registry has atomic replace + fsync, symlink rejection, canonical JSON validation and a fixed
model/weights policy. This preserves publisher provenance across process restart instead of leaving
the trust anchor only in RAM.

## Two-phase user decision

`VN97ModelImageProvisioningSession.review(...)` stages and verifies bytes, checks publisher
signature and creates the exact DIRECT production compatibility plan. It does not mutate active
inventory.

The UI displays:

- publisher key ID;
- SHA-256 fingerprint of the selected public key;
- package SHA-256;
- capability/version;
- source origin/license;
- compatibility plan SHA-256.

Only a separate **Trust & Activate** action persists the publisher key in VN97PTR1 and calls
`activateReviewed()`. The session retains the
exact verified capability, compatibility plan and trust store from the review so the UI cannot
silently swap identity between review and activation. The M10F coordinator revalidates staged
bytes, trust and compatibility again immediately before activation.

## Crash recovery

Before normal app model attachment and before reviewed activation, M10H calls the existing M10F
recovery coordinator with the M10G backend.

A committed-but-not-finalized activation is finalized into VN97INV1; a merely prepared activation
is rolled back according to M10F semantics.

## App integration

The app now has three Android document selectors:

- VN97CAP1 package;
- VN97SIG1 signature envelope;
- publisher Ed25519 public key.

Provisioning controls are disabled while a chat turn or M6 approval is active. A model cannot be
reloaded while the production assistant has an active turn or pending approval.

After activation succeeds, the process-scoped M10B assistant closes its previous canonical model
resources and reopens from the newly authoritative VN97INV1 entry.

## Verification

M10H adds a checkout-level regression wired into the existing runtime host-test runner:

`M10H_TRUSTED_IMPORT_ACTIVATION_PASS`

That regression is designed to exercise the real M10D/M10E/M10F/M10G production sources with a
generated Ed25519 keypair and assert that Review leaves inventory empty, Trust & Activate commits
VN97INV1, the artifact is content-addressed, the M10B evidence parser can read the resulting
inventory, and the model-image candidate validator is invoked.

In the implementation session, the container could not resolve GitHub for a checkout, so that
checkout-level regression was **not** reported as executed. Two isolated `kotlinc -Werror` gates
were actually run against the new orchestration/app contracts:

- `M10H_PROVISIONING_CONTRACT_PASS`
- `M10H_APP_PROVISIONING_SYNTAX_PASS`

Full Android document-picker UI, JNI validation against a real VN97MI1 image and APK/device
instrumentation remain Android build/device gates.
