# M19L — Final Turnkey Release Materialization

M19L is the final production materialization gate for VN97.

It does not replace M19D production release. It wraps the existing canonical
`vn97-production-release` with one prerequisite gate and one independent post-build
verification layer.

## Canonical flow

```text
VN97RUN1
  -> M19I preflight
  -> training
  -> VN97PRODCAMP1
  -> M19J physical VN97MOBEVID1
  -> M19F VN97INTAKE1 + VN97RC1
  -> M19K VN97CLOSE1 / VN97READY1
  -> M19L production materialization
  -> signed VN97-production.apk
  -> VN97APK1
  -> VN97FINAL1
```

## Command

```text
vn97-production-materialize \
  --manifest production-run.vn97run1 \
  --workspace-root <workspace> \
  --repository-root <VN97> \
  --private-key <publisher-ed25519-private-key> \
  --key-id <publisher-key-id> \
  --capability-version <version> \
  --source-origin <origin> \
  --source-license <license> \
  --validation-input <fresh-held-out-language-data> \
  --max-validation-loss <threshold> \
  --speech-validation-input <fresh-held-out-speech-manifest> \
  --max-speech-validation-loss <threshold> \
  --release-output-dir <new-release-directory> \
  --receipt <new-vn97final1-path>
```

Android signing remains external through the existing environment:

```text
VN97_RELEASE_KEYSTORE
VN97_RELEASE_STORE_PASSWORD
VN97_RELEASE_KEY_ALIAS
VN97_RELEASE_KEY_PASSWORD
```

## Mandatory M19K gate

M19L first runs M19K in inspect-only mode with the same release inputs.

If M19K does not return `READY_TO_RELEASE`, M19L does not build or sign anything.
The CLI prints the canonical VN97CLOSE1 and exits 2.

Therefore missing physical evidence, fresh release-validation inputs, publisher key,
Android signing credentials or toolchain dependencies remain visible as their real
production blockers.

## Canonical release builder only

After the READY gate M19L delegates to the existing:

```text
vn97-production-release
```

with the exact current VN97RC1 and operator-supplied fresh release-validation/signing
inputs.

M19L also propagates the bound VN97RUN1 physical-device thresholds into the existing
release builder:

- minimum device runs;
- text prefill p95;
- decode/token p95;
- PSS;
- thermal status;
- speech prefill p95 when configured;
- energy-counter requirement/delta when configured.

Final speech model-image/state budgets are propagated from VN97RUN1 speech options when
present.

M19L does not create another APK builder, another signer or another release path.

## Canonical release directory

The M19D builder must publish exactly:

```text
VN97-production.apk
bootstrap-release.vn97bootrel6.json
production-readiness.vn97ready1
release-attestation.vn97apk1
```

M19L rejects missing, extra, symlinked or malformed release artifacts.

`VN97FINAL1` is deliberately a sidecar receipt and is not inserted into this four-file
release directory.

## Independent post-build verification

After the canonical builder returns, M19L does not trust stdout alone.

It reopens the published release and verifies:

### VN97READY1

- canonical JSON;
- status READY;
- exact VN97RUN1 repository commit;
- exact VN97RC1 manifest SHA;
- exact model-image SHA;
- exact SHA matching the M19K READY closure.

### VN97APK1

- canonical JSON;
- exact APK byte count;
- exact APK SHA-256;
- exact VN97RC1 manifest SHA.

### VN97BOOTREL6

- canonical JSON;
- schema VN97BOOTREL6;
- exact VN97RC1 manifest binding;
- signed source SHA equal to VN97RC1;
- hash equal to VN97APK1 bootstrap report claim.

### Embedded APK release payload

M19L opens the APK ZIP directly and requires the canonical embedded assets:

```text
assets/vn97-bootstrap/model.vn97cap1
assets/vn97-bootstrap/model.vn97sig1
assets/vn97-bootstrap/publisher.ed25519
assets/vn97-release/release.vn97rel1
```

It rejects duplicate ZIP entries.

It then verifies:

- VN97CAP1 source SHA points to the exact VN97RC1 manifest;
- VN97SIG1 package/capability/version claims match VN97CAP1;
- Ed25519 signature verifies with the embedded publisher key;
- VN97REL1 package/signature/publisher sizes and SHA identities match embedded bytes;
- VN97APK1 package/signature/publisher/release-manifest hashes match embedded bytes;
- VN97BOOTREL6 package/public-key hashes match embedded bytes.

### External Android verification

M19L independently reruns:

```text
apksigner verify --verbose --print-certs
aapt dump badging
```

and requires:

- signer certificate SHA list exactly equals VN97APK1;
- application id is `ai.vn97.app`;
- version code/name match VN97APK1;
- application/version also match VN97READY1.

## VN97FINAL1

Only after all checks pass does M19L emit canonical `VN97FINAL1`.

It binds:

- VN97RUN1 manifest SHA;
- exact repository commit;
- M19K closure-report SHA;
- VN97READY1 SHA;
- VN97RC1 manifest SHA;
- VN97APK1 SHA;
- VN97BOOTREL6 SHA;
- VN97REL1 SHA;
- final APK SHA and byte count;
- application/version identity;
- APK signer certificate SHA identities.

Status is always:

```text
MATERIALIZED
```

Any non-materialized condition is represented by VN97CLOSE1 or by a hard verification
failure. M19L never emits a partial VN97FINAL1.

## Receipt immutability

`--receipt` creates a new sidecar file atomically and refuses overwrite.

This differs intentionally from M19K status reports, which may be replaced as campaign
state advances. VN97FINAL1 represents one final immutable release materialization.

## Honest boundary

Repository CI cannot produce a real VN97FINAL1 production release because it does not
hold the user's production publisher key, Android keystore, fresh held-out release data
or physical-device evidence.

CI can verify the M19L orchestration/report contracts. A real VN97FINAL1 only exists
after those external production prerequisites are supplied.

## Architecture boundary

M19L adds no model, trainer, planner, memory system, benchmark engine, signing backend
or alternate APK path.

The architecture remains:

```text
Code 1
  -> Code 2 hardware/mobile-aware
  -> one VN97 production model
  -> canonical evidence/intake/readiness
  -> canonical signed turnkey APK
```

## M19M clean-device acceptance

VN97FINAL1 proves the production APK was materialized and cryptographically verified.
The canonical next physical step is:

```text
vn97-turnkey-accept \
  --final-receipt <VN97FINAL1> \
  --release-dir <M19L release dir> \
  --repository-root <VN97 checkout> \
  --serial <physical clean-device adb serial> \
  --output <VN97ACCEPT1>
```

M19M installs only the exact VN97-production.apk, proves bundled bootstrap/READY state,
floating service, permission readiness, one canonical autonomous goal, a real reboot and
post-reboot persistence before emitting VN97ACCEPT1.
