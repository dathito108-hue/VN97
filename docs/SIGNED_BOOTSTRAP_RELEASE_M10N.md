# M10N — Signed VN97 Bootstrap Release CLI

M10N reduces the production release path to one explicit local command after training.

`vn97-bootstrap-release` consumes:

- M10L `model.vn97ck1`;
- canonical `tokenizer.vn97tk1`;
- a local Ed25519 private key file;
- publisher key ID, capability version and source/license metadata;
- an explicit destination asset directory.

It produces exactly the three public files consumed by M10J:

- `model.vn97cap1`;
- `model.vn97sig1`;
- `publisher.ed25519`.

## Private-key boundary

The private key is opened with a no-follow descriptor and must be either:

- exactly 32 raw Ed25519 seed bytes; or
- exactly 64 lowercase hexadecimal characters.

The key is used only to instantiate M10K `Ed25519PrivateKeySigner`. It is never copied to the
bundle, asset directory, training report, stdout or APK.

The CLI requires the optional signing dependency:

`pip install -e '.[signing]'`

## Trust/deployment chain

The CLI does not bypass any existing layer:

`VN97CK1 -> fresh canonical VN97LanguageCore -> M10K VN97MI1 -> VN97CAP1 -> VN97SIG1 -> M10J assets`

M10J still performs Android-side package, signature, compatibility and native model validation
before VN97INV1 activation.

## Safety

Checkpoint loading uses M10L safe descriptor loading. Tokenizer and private-key files use bounded
regular-file descriptor reads with symlink rejection. Asset writes use M10K fsync/atomic writer.

The tokenizer vocabulary must exactly match the checkpoint model vocabulary.

## Example

`vn97-bootstrap-release --checkpoint out/model.vn97ck1 --tokenizer out/tokenizer.vn97tk1 --private-key publisher.key --key-id publisher.main --capability-version 1 --source-origin vn97-training --source-license proprietary --assets-dir android/app/src/main/assets/vn97-bootstrap`

After that, the existing Android APK workflow packages a signed bootstrap that M10J can activate
automatically on first launch.

## Intelligence boundary

M10N packages existing trained VN97 intelligence. It does not generate training data, train a
teacher model, or convert an unrelated model/backend into VN97.
