# M19D — Production Release Orchestrator + Signed APK Attestation

M19D closes the final mechanical release chain between a qualified VN97RC1 and
a publishable Android release artifact.

The canonical production path is:

```text
M10P / M10R / M11C
  -> M19B VN97RC1
  -> M19C candidate-bound M10N signing
  -> M19D external staged bootstrap assets
  -> M19A release Gradle gate
  -> Android release signing
  -> apksigner verification
  -> APK payload/version verification
  -> VN97APK1 attestation
  -> atomic release publication
```

No second model, planner, memory engine, signing implementation or AI backend is
introduced.

## External release asset root

M19D never copies production bootstrap assets into the repository.

Android release builds now optionally consume:

`VN97_RELEASE_ASSET_ROOT`

The directory is an Android assets root and must contain:

```text
vn97-bootstrap/
  model.vn97cap1
  model.vn97sig1
  publisher.ed25519
```

When configured, M19A verifies that this root:

- is a real directory;
- is not a symlink;
- is outside the repository tree.

The normal source slot remains:

`android/app/src/main/assets/vn97-bootstrap/README.txt`

M19D requires that source slot to contain only that README, preventing a stale
or manually copied production payload from competing with the externally staged
candidate.

## Production orchestrator

The new command is:

`vn97-production-release`

It requires:

- a complete M19B `VN97RC1` directory;
- the M19C Ed25519 private-key path and publisher metadata;
- fresh held-out language validation inputs/criteria;
- fresh speech/vision validation inputs/criteria when those adapters exist;
- external Android signing environment;
- Android/Gradle build tools;
- a new output directory.

Android signing still uses the M19A environment contract:

- `VN97_RELEASE_KEYSTORE`
- `VN97_RELEASE_STORE_PASSWORD`
- `VN97_RELEASE_KEY_ALIAS`
- `VN97_RELEASE_KEY_PASSWORD`

The keystore must remain outside the repository.

## Pre-signing fail-closed order

Before M10N opens the Ed25519 private key, M19D verifies:

1. repository root and source bootstrap slot;
2. the complete VN97RC1 directory through the M19C verifier;
3. release output path is not already published;
4. all Android signing environment fields exist;
5. Android keystore is a non-empty regular non-symlink file outside repo;
6. Gradle is available;
7. `apksigner` is available;
8. `aapt` is available.

Only then does M19D call the existing M10N signer.

M10N still owns all model quality, modality, mobile-budget, production report,
device evidence and private-key boundaries.

## Bootstrap verification

M19D captures canonical `VN97BOOTREL6` from M10N and requires:

- report schema is exactly `VN97BOOTREL6`;
- report candidate manifest SHA equals the verified VN97RC1 SHA;
- `signed_source_sha256` equals that same candidate SHA;
- report package/public-key identities equal emitted bytes.

M19D then parses the emitted VN97CAP1 and VN97SIG1 and re-verifies:

- VN97CAP1 source SHA equals VN97RC1 manifest SHA;
- VN97SIG1 package/capability/version claims equal VN97CAP1;
- publisher key is exactly 32 bytes;
- Ed25519 signature verifies with `publisher.ed25519`.

This is an independent post-signing check before Android build.

## Release build

M19D creates a temporary asset root outside the repository and sets:

`VN97_RELEASE_ASSET_ROOT=<temporary-root>`

It then executes:

```text
gradle -p android --no-daemon :app:clean :app:assembleRelease
```

M19A remains authoritative for:

- exact bootstrap asset bounds;
- VN97CAP1 magic;
- external Android signing configuration;
- version contract;
- generated VN97REL1;
- release-only `VN97_TURNKEY_REQUIRED=true`.

The Android private keystore is never copied into the repository or APK assets.

## Signed APK verification

A release APK is not published merely because Gradle returned success.

M19D requires:

```text
apksigner verify --verbose --print-certs
```

to succeed and extracts the SHA-256 digest of every signer certificate.

It also runs:

```text
aapt dump badging
```

and records the actual APK:

- application ID;
- versionCode;
- versionName.

The application ID must be `ai.vn97.app`.

## APK payload verification

M19D opens the APK as a ZIP and rejects duplicate ZIP entry names.

It requires the exact entries:

- `assets/vn97-bootstrap/model.vn97cap1`
- `assets/vn97-bootstrap/model.vn97sig1`
- `assets/vn97-bootstrap/publisher.ed25519`
- `assets/vn97-release/release.vn97rel1`

The three bootstrap entries must be byte-for-byte identical to the M19C staged
signed bytes.

VN97REL1 is parsed independently in Python with:

- strict UTF-8;
- exact line count/fields;
- strict base64;
- exact byte bounds;
- lowercase SHA-256 validation.

Its application/version must equal the actual `aapt` identity and its
bootstrap sizes/SHA-256 values must equal the actual bytes inside the APK.

## VN97APK1

After all checks pass, M19D emits canonical strict JSON:

`release-attestation.vn97apk1`

Schema: `VN97APK1`.

It binds:

- application ID;
- versionCode/versionName;
- APK byte count;
- APK SHA-256;
- sorted unique APK signer-certificate SHA-256 identities;
- VN97RC1 manifest SHA-256;
- VN97BOOTREL6 report SHA-256;
- VN97CAP1 SHA-256;
- VN97SIG1 SHA-256;
- publisher public-key SHA-256;
- VN97REL1 SHA-256.

A strict parser is included for later audit. It rejects non-canonical JSON,
duplicate keys, malformed UTF-8, missing/unexpected fields, invalid digests,
unsorted/duplicate signer identities, or invalid APK metadata.

## Atomic publication

The user-facing output directory must not already exist.

Only after all signing/build/attestation checks pass does M19D atomically publish:

```text
VN97-production.apk
bootstrap-release.vn97bootrel6.json
release-attestation.vn97apk1
```

A failed model gate, signing failure, Gradle failure, APK signature failure,
payload mismatch or attestation mismatch cannot publish a partially valid final
release directory.

## Current production boundary

M19D completes the release machinery, but it still does not create artificial
production intelligence or artificial phone evidence.

The repository currently lacks the real M10P/M10R/M11C winner and physical
VN97MOBEVID1 evidence required to create an honest VN97RC1.

Therefore a true final APK remains blocked on real production campaign outputs,
not on missing release plumbing.

That distinction is intentional.
