# M19A — Release / Turnkey Production Gate

M19A turns the existing M10S zero-provision contract into an enforceable
production release boundary.

It does not claim that the final production APK exists yet. The repository
intentionally does not contain a release keystore or a production bootstrap
payload. A release build must fail until both are supplied through the release
pipeline.

The canonical architecture remains:

```text
Code 1
  -> Code 2 hardware/mobile-aware
  -> signed VN97 production model
  -> canonical planner / VN97MEM1 / M6
  -> Android turnkey release
```

No Transformer/LLaMA/cloud backend, alternate planner, alternate memory engine,
or second model runtime is introduced.

## Production application identity

M19A advances the Android package to the M19 release line:

- application ID: `ai.vn97.app`
- versionCode: `190100`
- versionName: `1.0.0-rc1`

The version code is strictly above all earlier development APKs in this repo,
so a signed production package can use Android's normal upgrade semantics.

## Mandatory public bootstrap

A release must contain exactly these public bootstrap files:

- `assets/vn97-bootstrap/model.vn97cap1`
- `assets/vn97-bootstrap/model.vn97sig1`
- `assets/vn97-bootstrap/publisher.ed25519`

The source directory may also contain only the explanatory `README.txt`.

The Gradle release gate requires:

- the bootstrap root to be a real directory, not a symlink;
- every required asset to be a regular non-symlink file;
- `model.vn97cap1` size: 96 bytes through 512 MiB;
- the first eight package bytes to be exactly `VN97CAP1`;
- `model.vn97sig1` size: 1 through 16384 bytes;
- `publisher.ed25519` to be exactly the raw 32-byte Ed25519 public key;
- no unexpected file in the bootstrap source directory.

Full package structure, section digests, signature binding, publisher scope,
compatibility and VN97MI1 native validation remain enforced by the existing
canonical provisioning pipeline on-device.

## Release signing boundary

Production APK signing material must be supplied externally through:

- `VN97_RELEASE_KEYSTORE`
- `VN97_RELEASE_STORE_PASSWORD`
- `VN97_RELEASE_KEY_ALIAS`
- `VN97_RELEASE_KEY_PASSWORD`

The release build fails if any field is missing.

The keystore must:

- be a non-empty regular file;
- not be a symlink;
- live outside the repository tree.

No private signing key, keystore, password, or signing secret is added to VN97
source control or APK assets.

## VN97REL1 release manifest

For a valid release payload Gradle generates:

```text
assets/vn97-release/release.vn97rel1
```

VN97REL1 binds:

- application ID;
- version code;
- version name;
- model package byte count + SHA-256;
- signature envelope byte count + SHA-256;
- publisher public-key byte count + SHA-256.

The generated manifest contains no secret.

The runtime parser is bounded to 4096 bytes, requires strict UTF-8, exact
fields, lowercase SHA-256 values, expected asset bounds, and a trailing newline.

## Runtime release binding

A turnkey release verifies VN97REL1 before accepting the bundled bootstrap.

The runtime:

1. verifies the manifest application/version identity against the actual APK;
2. verifies the small VN97SIG1 bytes against the release manifest;
3. sends the bundled VN97CAP1, VN97SIG1 and publisher key through the canonical
   trust/provisioning pipeline;
4. compares the trusted provisioning `packageSha256` with the VN97REL1 model
   SHA-256;
5. compares the trusted publisher-key SHA-256 with VN97REL1;
6. activates only through the existing transactional model activation path.

The large model asset is not redundantly hashed a second time before staging.
The canonical VN97CAP1 parser/trust path already hashes and verifies it while
producing the trusted provisioning identity used for the VN97REL1 comparison.

## APK upgrade behavior

M10S handled first-launch bootstrap. M19A extends this to package upgrades.

The M18A `ACTIVATION` recovery domain now enforces the turnkey bootstrap when
`BuildConfig.VN97_TURNKEY_REQUIRED=true`.

Therefore all execution-entry surfaces share the same release rule:

- chat;
- floating assistant;
- autonomous continuation;
- paper trading;
- game agent;
- reboot/package-replaced recovery.

If the currently activated model package SHA-256 already equals the VN97REL1
model SHA-256, activation is a no-op.

If a new APK carries a different canonical signed model package, the new bundle
must pass the existing provisioning/activation pipeline before production
execution continues.

A stale model from the previous APK cannot silently remain authoritative.

## Fail-closed behavior

A production release fails closed when:

- public bootstrap files are missing or malformed at build time;
- signing inputs are incomplete;
- a keystore is placed inside the repository;
- the release version violates the M19 contract;
- VN97REL1 is missing, malformed, oversized or bound to another APK identity;
- the bundled package/publisher identity differs from VN97REL1;
- normal signature/trust/compatibility/native-open validation fails.

There is no fallback to manual model import in release mode and no fallback to
another AI backend.

## Current repository state

At M19A merge time the repository still contains only the bootstrap slot
`README.txt`; it does not contain a real production `model.vn97cap1`,
`model.vn97sig1`, or `publisher.ed25519`.

That is intentional.

The normal development/debug APK remains buildable. The production release gate
must reject a release until a release-qualified canonical VN97 checkpoint is
converted and signed through the existing M10N path and signing credentials are
provided outside source control.

## Validation contract

`M19ATurnkeyReleaseManifestTest.kt` verifies:

- valid manifest parsing;
- APK application/version binding;
- package identity binding;
- publisher identity binding;
- signature identity binding;
- publisher-size bounds;
- trailing-data rejection;
- invalid SHA rejection;
- malformed UTF-8 rejection;
- manifest-size bounds.

M19A CI also asserts that the current repository correctly fails the real
release bootstrap/signing gates while the full debug APK still compiles and
verifies.

That negative release check is evidence that the final production path cannot
silently emit an unsigned or intelligence-empty APK.
