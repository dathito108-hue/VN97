# M19E — Production Readiness / Artifact Intake Gate

M19E converts the remaining production blockers into one canonical,
machine-verifiable preflight.

The final release path is now:

```text
real M10P/M10R/M11C winner + physical evidence + signing inputs
  -> M19B VN97RC1
  -> vn97-production-release --preflight-only
  -> VN97READY1 == READY
  -> same vn97-production-release command without --preflight-only
  -> M19C candidate-bound signing
  -> M19D Android release orchestration
  -> signed APK + release evidence
```

There is no second readiness implementation. Normal production release calls the
same M19E engine first and refuses to enter M10N/signing/build unless it returns
`READY`.

## VN97READY1

M19E emits canonical strict JSON with schema `VN97READY1`.

It records:

- READY/BLOCKED status;
- clean Git HEAD commit identity;
- Android application/version identity;
- VN97RC1 candidate summary:
  - candidate manifest SHA-256;
  - selected candidate ID;
  - checkpoint SHA-256;
  - tokenizer SHA-256;
  - reconstructed model-image SHA-256;
  - device-evidence count;
  - distinct physical-device profile count;
  - speech/vision modality flags;
- one boolean result for every readiness domain;
- sorted blocker records with stable code/domain/message fields.

The report contains no private-key bytes, signing passwords, keystore password,
or other signing secrets.

## Canonical checks

The readiness gate currently covers:

- repository:
  - real VN97 repository root;
  - clean tracked Git worktree;
  - resolvable 40-hex HEAD commit;
- release identity:
  - `ai.vn97.app`;
  - versionCode at or above the M19 floor;
  - `1.0.0` or `1.0.0-rcN` version name;
- source bootstrap slot:
  - must remain README-only;
- candidate:
  - complete M19B/M19C VN97RC1 directory;
  - all checkpoint/tokenizer/report/device-evidence hashes reverified;
- release metadata:
  - publisher key ID;
  - capability version;
  - source origin;
  - source license;
- publisher private key:
  - safe regular file;
  - exactly 32 raw Ed25519 seed bytes or 64 lowercase hex;
  - must stay outside repository;
- fresh held-out validation:
  - at least one language validation file;
  - speech validation required only for speech-enabled VN97RC1;
  - vision validation required only for vision-enabled VN97RC1;
  - modality validation is rejected when the candidate does not contain that
    modality;
- quality thresholds:
  - finite positive language max validation loss;
  - finite positive speech/vision max validation loss when corresponding
    modality is enabled;
- Python production dependencies:
  - torch;
  - cryptography;
- Android signing environment:
  - VN97_RELEASE_KEYSTORE;
  - VN97_RELEASE_STORE_PASSWORD;
  - VN97_RELEASE_KEY_ALIAS;
  - VN97_RELEASE_KEY_PASSWORD;
- Android keystore:
  - safe non-empty regular file;
  - outside repository;
- build/verification tools:
  - Gradle;
  - apksigner;
  - aapt;
- output:
  - requested final output path does not already exist;
  - an existing safe ancestor directory is available for atomic publication.

## Preflight mode

Use the exact production command with:

```text
--preflight-only
```

Example:

```text
vn97-production-release \
  --preflight-only \
  --repository-root /work/VN97 \
  --release-candidate-dir /release/vn97rc1 \
  --private-key /secure/publisher.private \
  --key-id publisher.main \
  --capability-version 1 \
  --source-origin vn97-release-candidate \
  --source-license proprietary \
  --validation-input /release/heldout-chat.jsonl \
  --max-validation-loss 3.0 \
  --output-dir /release/final
```

Exit behavior:

- `0`: VN97READY1 status is READY;
- `2`: VN97READY1 status is BLOCKED.

The canonical report is printed to stdout.

Optionally:

```text
--readiness-report /release/vn97ready1.json
```

writes the same canonical bytes atomically. That option is preflight-only.

## Normal release mode

Without `--preflight-only`, `vn97-production-release` runs the same M19E
readiness engine first.

If status is BLOCKED:

- M10N is not entered;
- no Ed25519 signature is generated;
- Gradle release build is not entered;
- no final output directory is published.

The failure lists the stable VN97READY1 blocker codes.

If status is READY, release continues through the existing M19C/M19D path.

M10N/Torch signing/release code is imported only after the readiness gate passes
inside the release CLI path.

## Final publication

A successful normal M19D/M19E release now publishes:

```text
VN97-production.apk
bootstrap-release.vn97bootrel6.json
release-attestation.vn97apk1
production-readiness.vn97ready1
```

The readiness report is the machine-readable preflight snapshot for the exact
release invocation. APK correctness and signing identity remain independently
verified by M19D/VN97APK1.

## Current repository status

The release plumbing is now complete through readiness, signing, build,
verification and atomic publication.

The repository still does not contain the real production inputs required for a
READY report:

- real production VN97RC1 generated from M10P/M10R/M11C outputs;
- physical-phone release VN97MOBEVID1 evidence inside that candidate;
- publisher Ed25519 private key;
- Android release keystore/passwords;
- fresh held-out release validation inputs.

Therefore the honest current state is expected to be BLOCKED until those
external production artifacts exist.

No CI fixture is allowed to be represented as production intelligence or a
final production APK.
