# M19B — Production Bootstrap Qualification / VN97RC1

M19B inserts one explicit production qualification boundary between the
training/evidence pipeline and M10N private-key signing.

The canonical architecture remains:

```text
Code 1
  -> Code 2 hardware/mobile-aware
  -> VN97 production campaign winner
  -> VN97RC1 release candidate
  -> M10N signed bootstrap
  -> M19A turnkey Android release gate
```

M19B does not create a second model, planner, memory engine, inference backend,
teacher model, cloud dependency, or alternate architecture.

## Why this block exists

Before M19B, the repository already had:

- M10P deterministic best-candidate selection;
- M10Q exact mobile footprint admission;
- M10R sealed final language holdout;
- M11C unified production campaign + real-device evidence machinery;
- M10N quality/evidence-gated bootstrap signing;
- M19A signed turnkey Android release enforcement.

The missing production handoff was an explicit artifact set proving that the
exact checkpoint/tokenizer/report/mobile-evidence identities belong together
*before* any private signing key is opened.

M19B defines that handoff as `VN97RC1`.

## Required inputs

`vn97-release-candidate` requires:

- `model.vn97ck1`;
- `tokenizer.vn97tk1`;
- `production-campaign-report.json` with schema `VN97PRODCAMP1`;
- one or more canonical `VN97MOBEVID1` device-evidence files;
- `speech-training-report.json` when the checkpoint contains the canonical
  AudioFrameAdapter;
- `vision-training-report.json` when the checkpoint contains the canonical
  VisionPatchAdapter.

No private signing key or Android release keystore is accepted by this command.

## Identity reconstruction

Qualification does not trust report hashes alone.

The CLI:

1. loads and integrity-verifies VN97CK1 through M10L;
2. parses the exact VN97TK1 tokenizer and requires vocabulary compatibility;
3. reads canonical VN97PRODCAMP1;
4. takes the deployment tile geometry from that report;
5. rebuilds the unsigned canonical VN97MI1 preview from the exact checkpoint,
   tokenizer and embedded adapters;
6. requires the reconstructed VN97MI1 SHA-256 to equal
   `VN97PRODCAMP1.model_image_sha256`;
7. reconstructs the exact mobile footprint and requires byte-for-byte equality
   with the report footprint;
8. reapplies the report's mobile budget;
9. requires the speech validation and sealed speech release observations in
   VN97PRODCAMP1 to still satisfy their recorded criteria;
10. validates speech/vision training-report identity when those adapter weights
    exist;
11. verifies every device-evidence file against the reconstructed VN97MI1 and
    the configured performance/resource limits.

A copied report cannot authorize a different checkpoint, tokenizer, packed
model image, adapter set or device evidence.

## Device evidence

Each accepted VN97MOBEVID1 record is checked with the existing
`VN97DeviceEvidenceCriteria` implementation.

The candidate command supports:

- minimum benchmark runs;
- max text-prefill p95;
- max decode/token p95;
- max peak PSS;
- max thermal status;
- optional max speech-prefill p95;
- optional required battery energy counter;
- optional max absolute energy-counter delta;
- a minimum number of distinct device profiles.

Every accepted evidence record must:

- bind the exact reconstructed VN97MI1 SHA-256;
- be from Android API 26 or newer;
- pass the configured criteria;
- have a unique evidence SHA-256.

Distinct profile identity is:

```text
manufacturer / model / sdk_int / abi
```

The default minimum is one profile. A production release policy can raise it
without changing the model architecture.

## VN97RC1 manifest

A successful qualification emits canonical strict JSON:

`release-candidate.vn97rc1`

Schema: `VN97RC1`.

It binds:

- selected deterministic candidate ID;
- VN97CK1 SHA-256 + byte count;
- VN97TK1 SHA-256 + byte count;
- VN97PRODCAMP1 SHA-256;
- reconstructed VN97MI1 SHA-256;
- deployment tile rows/columns;
- speech/vision enabled flags;
- required speech/vision training-report SHA-256 values;
- all accepted device-evidence identities and measured summaries.

The parser rejects:

- non-canonical JSON;
- malformed UTF-8;
- unexpected keys;
- invalid SHA-256 values;
- duplicate device evidence;
- missing evidence;
- speech/vision report mismatch with enabled modalities;
- device evidence below Android minSdk 26;
- oversized manifests.

## Atomic candidate assembly

The output directory must not already exist.

M19B assembles into a same-parent temporary directory and copies every source
through no-follow regular-file descriptors. Each copy is hashed while streaming
and must equal the identity already verified during qualification.

The candidate directory contains:

- `model.vn97ck1`;
- `tokenizer.vn97tk1`;
- `production-campaign-report.json`;
- optional canonical speech/vision training reports required by the checkpoint;
- `device-evidence/*.json`;
- `release-candidate.vn97rc1`.

Files and directories are fsynced before the temporary directory is atomically
renamed into the requested output path.

A partially assembled candidate is therefore never published as the requested
candidate directory.

## M10N handoff

VN97RC1 is a pre-signing qualification artifact. It does not replace M10N.

The assembled files feed the existing M10N command:

```text
vn97-release-candidate
  -> qualified VN97RC1 directory
  -> vn97-bootstrap-release
  -> fresh held-out release validation
  -> optional production/mobile evidence gates
  -> private Ed25519 key opened only after all gates pass
  -> model.vn97cap1 / model.vn97sig1 / publisher.ed25519
```

M10N still reconstructs VN97MI1 and verifies release quality before signing.
M19B therefore adds a pre-signing evidence handoff rather than weakening any
existing release gate.

M19C consumes this directory directly through `--release-candidate-dir`,
re-verifies every bound identity/evidence file, and signs the VN97RC1 manifest
SHA into VN97CAP1 provenance before emitting the Android bootstrap assets.

## Current repository status

At the time M19B is implemented, neither `main` nor the historical M10R/M11C
branches contain a real generated production winner:

- no production `model.vn97ck1`;
- no production `tokenizer.vn97tk1`;
- no `VN97PRODCAMP1` output from a real campaign;
- no physical-phone `VN97MOBEVID1` release evidence.

Those files are intentionally not manufactured by CI.

Therefore M19B can complete the qualification/assembly machinery, but it cannot
truthfully create a production VN97RC1 from the current repository contents.

The next production operation after M19B is to run a real M10P/M10R/M11C
campaign and collect representative physical-device evidence. Only those real
outputs are eligible to enter `vn97-release-candidate`.

## Validation

M19B adds:

- a lightweight stdlib host contract for VN97RC1 serialization/parsing and
  rejection behavior;
- full integration regressions for checkpoint/tokenizer/report/device-evidence
  qualification and assembly;
- Python compile checks for the production CLI.

The lightweight contract is suitable for the Android CI workflow without
pulling an additional Torch installation solely for the M19B gate.
