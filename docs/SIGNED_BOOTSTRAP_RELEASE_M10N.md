# M10N — Signed VN97 Bootstrap Release CLI

M10N is the canonical local signer that converts already-qualified VN97
intelligence into the exact public bootstrap assets consumed by Android.

It does not train a model and does not introduce another inference backend.

## Inputs

Raw/development mode accepts:

- `model.vn97ck1`;
- `tokenizer.vn97tk1`;
- one or more held-out validation JSONL files;
- optional speech/vision training + held-out validation inputs when those
  adapters exist;
- optional production-campaign and device-evidence gates;
- a local Ed25519 private key;
- publisher key ID, capability version and source/license metadata;
- destination asset directory.

M19C also adds candidate-bound mode:

```text
--release-candidate-dir <VN97RC1 directory>
```

In candidate-bound mode the checkpoint, tokenizer, production report,
speech/vision training reports, device evidence and tile geometry are resolved
from the verified candidate and cannot be overridden by parallel file inputs.

## Quality and mobile gates

Before signing, M10N performs fresh canonical checks:

- held-out language loss/top-1/target-token criteria;
- held-out speech criteria for speech-enabled checkpoints;
- held-out vision criteria for vision-enabled checkpoints;
- exact mobile model-image and recurrent-state budgets;
- VN97PRODCAMP1 identity when supplied/resolved;
- VN97MOBEVID1 thresholds when supplied/resolved.

In M19C mode every VN97RC1 device-evidence file is reloaded and checked using
the current M10N thresholds.

The Ed25519 private-key file is not opened until these gates pass.

## Private-key boundary

The private key is opened with a no-follow descriptor and must be either:

- exactly 32 raw Ed25519 seed bytes; or
- exactly 64 lowercase hexadecimal characters.

It is used only to instantiate the canonical
`Ed25519PrivateKeySigner`.

It is never copied into VN97CAP1, VN97SIG1, the release report or Android
assets.

The optional signing dependency remains:

```text
pip install -e '.[signing]'
```

## Signed package provenance

Raw mode retains checkpoint provenance:

```text
CapabilitySource.source_sha256 = checkpoint_sha256
```

M19C candidate mode signs the release-candidate identity instead:

```text
CapabilitySource.source_sha256 =
  sha256(release-candidate.vn97rc1)
```

CapabilitySource is part of VN97CAP1. VN97SIG1 signs the VN97CAP1 package
identity, so the VN97RC1 manifest SHA is cryptographically bound to the public
bootstrap package.

## Outputs

The canonical atomic writer produces exactly:

- `model.vn97cap1`;
- `model.vn97sig1`;
- `publisher.ed25519`.

These are the three public assets consumed by the Android bootstrap path.

## Release report

The current report schema is `VN97BOOTREL6`.

It records:

- checkpoint/tokenizer/model-image identities;
- fresh validation results and criteria;
- mobile footprint/budget;
- optional speech/vision validation;
- optional raw device evidence;
- production-campaign-report SHA;
- package SHA;
- publisher public-key SHA;
- `signed_source_sha256`;
- optional M19C release-candidate identity.

For candidate mode the report records the VN97RC1 manifest SHA, deterministic
candidate ID, tile geometry and all device-evidence SHA identities.

## M19C production example

```text
vn97-bootstrap-release \
  --release-candidate-dir out/vn97-release-candidate \
  --validation-input heldout-chat.jsonl \
  --validation-format chat \
  --max-validation-loss 3.0 \
  --min-validation-target-tokens 1000 \
  --private-key /secure/publisher.private \
  --key-id publisher.main \
  --capability-version 1 \
  --source-origin vn97-release-candidate \
  --source-license proprietary \
  --assets-dir android/app/src/main/assets/vn97-bootstrap
```

The Android M19A release gate then binds those signed public assets to
VN97REL1 and external APK signing.

## Trust/deployment chain

```text
VN97CK1
 -> fresh canonical VN97 model
 -> VN97MI1
 -> VN97CAP1
 -> VN97SIG1
 -> M10J/M19A bootstrap assets
 -> Android signature/trust/compatibility/native-open validation
 -> canonical activation
```

M19C prepends VN97RC1 qualification to that chain for production release
candidates. It does not bypass or replace any M10N validation.

## Intelligence boundary

M10N packages existing trained VN97 intelligence only after explicit measured
quality gates pass.

Passing the release gate is evidence for the supplied validation/evidence sets;
it is not by itself a claim of AGI or universal capability.
