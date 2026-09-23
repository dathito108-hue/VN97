# M10K — Production VN97 Bootstrap Bundle Builder

M10K closes the export/package/signing gap between a trained canonical VN97 model and the M10J
turnkey APK bootstrap slot.

It does **not** train a model and does not create substitute/random production weights.

## Canonical build path

`VN97LanguageCore + VN97TokenizerPackage`
`-> build_model_image(...)`
`-> VN97MI1`
`-> VN97CAP1(model.language / weights / model_image:VN97MI1)`
`-> VN97SIG1 Ed25519 signature`
`-> assets/vn97-bootstrap/{model.vn97cap1, model.vn97sig1, publisher.ed25519}`

Those are exactly the three files consumed by M10J.

## API

`build_bootstrap_bundle(...)` requires:

- an already-trained `VN97LanguageCore`;
- a tokenizer whose vocabulary exactly matches the model;
- bounded source provenance through `CapabilitySource`;
- an unsigned-32 capability version;
- a signer exposing one 32-byte Ed25519 public key and a 64-byte signature.

The builder serializes the supplied model only. It never initializes, trains, downloads, imports or
mutates intelligence.

The bundle self-checks:

- canonical VN97MI1 identity;
- model.language / weights capability identity;
- exactly one model_image / VN97MI1 section;
- package SHA-256;
- VN97SIG1 package/capability/version/key binding.

## Production signer

`Ed25519PrivateKeySigner` accepts a raw 32-byte Ed25519 private key and is backed by the optional
Python `cryptography` package:

`pip install -e '.[signing]'`

Private keys are never written into the bundle or APK. Only the 32-byte public key is emitted as
`publisher.ed25519`.

## Asset writer

`write_bootstrap_assets(...)` writes exactly the three M10J filenames using same-directory
temporary files, fsync and atomic per-file replace, then fsyncs the directory and verifies the
written bytes.

If an interrupted update leaves only part of a new set, M10J intentionally classifies the slot as
INCOMPLETE and fails closed rather than activating mixed identities.

## Remaining intelligence boundary

A genuinely turnkey chat APK still needs a trained, tokenizer-bearing VN97 checkpoint. M10K makes
the packaging path deterministic once those weights exist; it does not pretend the tiny native test
fixture is assistant intelligence.
