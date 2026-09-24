# M19C — Signed Release Candidate Bridge

M19C closes the production provenance gap between M19B `VN97RC1` and the
existing M10N Ed25519 bootstrap signer.

The canonical release path is now:

```text
M10P/M10R/M11C
  -> VN97CK1 + VN97TK1 + VN97PRODCAMP1 + VN97MOBEVID1
  -> M19B VN97RC1 qualification
  -> M19C M10N candidate-bound signing
  -> signed VN97CAP1 / VN97SIG1 / publisher.ed25519
  -> M19A VN97REL1 Android release gate
  -> production APK
```

M19C does not add a second signer, model, planner, memory engine, inference
backend or cloud dependency.

## Candidate-bound M10N mode

`vn97-bootstrap-release` now accepts:

```text
--release-candidate-dir <VN97RC1 directory>
```

When this option is used, M10N resolves all candidate-controlled inputs from the
directory itself:

- `model.vn97ck1`;
- `tokenizer.vn97tk1`;
- `production-campaign-report.json`;
- optional speech training report;
- optional vision training report;
- all `device-evidence/*.json`;
- tile rows/columns from VN97RC1.

The caller may not simultaneously override checkpoint, tokenizer, production
report, modality reports or device evidence.

Explicit `--tile-rows` / `--tile-cols` are allowed only when they exactly
match VN97RC1.

Raw M10N mode remains available for development/backward compatibility.

## Directory re-verification

Before any model evaluation or private-key read, M19C reopens the candidate
directory and requires the exact M19B layout.

The loader rejects:

- candidate-root symlinks;
- missing or extra top-level entries;
- missing or extra device-evidence entries;
- unexpected files such as private keys;
- symlinked candidate files;
- checkpoint/tokenizer/report/evidence byte-size violations;
- any SHA-256 mismatch against VN97RC1;
- renamed or reordered evidence files.

Large checkpoint/tokenizer identities are verified with bounded streaming reads
rather than copied into a second in-memory buffer solely for M19C.

## Fresh M10N reconstruction

Directory identity alone is not sufficient.

After M19C resolves the candidate, the unchanged M10N release logic still:

1. loads VN97CK1 through the canonical deployment loader;
2. parses VN97TK1 and verifies vocabulary compatibility;
3. runs fresh held-out language release validation;
4. runs fresh speech validation for speech-enabled checkpoints;
5. runs fresh vision validation for vision-enabled checkpoints;
6. reconstructs the exact VN97MI1 using the VN97RC1 tile geometry;
7. reapplies mobile model/state budget limits;
8. verifies VN97PRODCAMP1 against checkpoint/tokenizer/VN97MI1 identity;
9. verifies every candidate VN97MOBEVID1 record using current M10N evidence
   thresholds.

The Ed25519 private key is opened only after all of the above passes.

## Winner and evidence binding

M19C additionally requires:

- `VN97RC1.selected_candidate_id` to equal
  `VN97PRODCAMP1.selected_candidate_id`;
- speech/vision presence flags to equal the actual checkpoint adapters;
- speech/vision training-report SHA values to equal VN97RC1;
- every evidence SHA in VN97RC1 to exist in the candidate;
- every actual evidence summary to equal the VN97RC1 summary for that digest.

The compared evidence summary includes:

- manufacturer/model/API/ABI;
- run count;
- text prefill p95;
- decode/token p95;
- optional speech prefill p95;
- peak PSS;
- maximum thermal status;
- optional battery energy-counter delta.

A hand-edited VN97RC1 cannot therefore substitute different measured values
while retaining the same evidence identity.

## Cryptographic candidate provenance

Raw M10N releases retain the historical signed source provenance:

```text
CapabilitySource.source_sha256 = checkpoint_sha256
```

Candidate-bound M19C releases use:

```text
CapabilitySource.source_sha256 = sha256(release-candidate.vn97rc1)
```

`CapabilitySource` is inside the canonical VN97CAP1 manifest. VN97CAP1's
package SHA is what VN97SIG1 signs with Ed25519.

Therefore the exact canonical VN97RC1 manifest identity is cryptographically
bound into the signed production package itself.

VN97RC1 in turn binds the checkpoint, tokenizer, production report, modality
reports, model-image identity and all accepted device-evidence identities.

## VN97BOOTREL6

M10N release output advances to `VN97BOOTREL6`.

The report adds:

- `signed_source_sha256`;
- `release_candidate` (null in raw mode);
- candidate manifest SHA-256;
- selected candidate ID;
- candidate tile geometry;
- all candidate device-evidence SHA-256 identities.

In candidate mode:

```text
report.signed_source_sha256
  == report.release_candidate.manifest_sha256
  == VN97CAP1.manifest.source.source_sha256
```

The report itself is release evidence. The cryptographic binding is provided by
the VN97CAP1/VN97SIG1 pair.

## M19A handoff

M19C intentionally reuses `write_bootstrap_assets()`.

After all quality/evidence checks and signing, the exact public files are
written atomically:

- `model.vn97cap1`;
- `model.vn97sig1`;
- `publisher.ed25519`.

M19D now supplies the three public inputs through an external temporary
`VN97_RELEASE_ASSET_ROOT` rather than copying production bootstrap files into
the repository.

M19A then generates VN97REL1, checks APK/version/bootstrap identity and requires
external Android release signing material. M19D verifies the resulting signed
APK and publishes VN97APK1 only after all checks pass.

No private Ed25519 key or Android keystore is copied into APK assets.

## Production example

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

Speech/vision enabled checkpoints additionally supply fresh held-out
speech/vision validation inputs and thresholds. Their training reports are
resolved from VN97RC1 automatically.

## Current production boundary

M19C completes the signed bridge machinery, but it still cannot manufacture a
real production candidate.

The repository does not currently contain the real M10P/M10R/M11C winner and
physical-phone release evidence required by M19B.

Until those real artifacts exist:

- no honest production VN97RC1 can be assembled;
- M19C cannot honestly create production bootstrap assets from repository
  fixtures;
- M19A must continue to fail closed for a production APK.

That boundary is deliberate.
