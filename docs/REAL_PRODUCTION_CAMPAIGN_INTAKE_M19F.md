# M19F — Real Production Campaign Execution Handoff / Evidence Intake

M19F closes the gap between the existing real training/evidence executors and
the M19 release-candidate/readiness chain.

It does **not** add another trainer or model path.

The canonical production sequence is now:

```text
vn97-campaign
  -> VN97CAMP2 language winner
  -> vn97-production-campaign
  -> VN97PRODCAMP1 unified speech-enabled winner
  -> physical Android VN97MOBEVID1 capture(s)
  -> vn97-production-intake
  -> VN97INTAKE1 + release-candidate/
  -> vn97-production-release --preflight-only
  -> VN97READY1
  -> signed production release
```

## Why M19F exists

Before M19F, each executor already existed:

- M10P/M10R: real multi-candidate language training, validation selection and
  sealed release evaluation;
- M11C: one canonical speech-adapter training/finalization pass on the fixed
  language winner;
- Android M11C: physical-device evidence capture;
- M19B: VN97RC1 qualification;
- M19E: final release readiness.

What was missing was one artifact-intake transaction proving that the files
handed from those stages belong to the same winner and the same model image.

M19F adds that chain-of-custody boundary.

## Canonical language-campaign input

`--language-campaign-dir` must be the direct output of `vn97-campaign` and
contain exactly:

```text
campaign-report.json
model.vn97ck1
tokenizer.vn97tk1
```

M19F requires:

- strict canonical `VN97CAMP2`;
- exact root keys;
- one unique selected candidate row;
- selected row status `ELIGIBLE`;
- selected row checkpoint SHA equals the report selected checkpoint SHA;
- checkpoint file SHA equals that same identity;
- tokenizer file SHA equals the report tokenizer identity;
- train/validation/release dataset identities are valid lowercase SHA-256.

This preserves M10P/M10R deterministic winner provenance.

## Canonical production-campaign input

`--production-campaign-dir` must be the direct current M11C output:

```text
model.vn97ck1
tokenizer.vn97tk1
speech-training-report.json
production-campaign-report.json
```

M19F requires strict canonical `VN97PRODCAMP1` and proves that it chains to
the supplied VN97CAMP2 winner:

- selected candidate ID matches;
- language campaign report SHA matches the actual campaign report bytes;
- language checkpoint SHA matches the VN97CAMP2 winner;
- tokenizer SHA matches;
- unified checkpoint file SHA matches `unified_checkpoint_sha256`;
- production tokenizer file SHA matches;
- model-image identity is valid;
- speech training dataset identity is valid.

The `VN97SPEECHTRAIN1` report is also checked against:

- base language checkpoint;
- unified production checkpoint;
- tokenizer;
- production speech-training dataset identity.

## Physical-device evidence intake

One or more `--device-evidence` arguments are required.

Every file is parsed with the existing canonical `VN97MOBEVID1` parser.

Before the expensive M19B reconstruction step, M19F already requires:

- unique evidence SHA-256;
- Android API >= 26;
- evidence model-image SHA equals `VN97PRODCAMP1.model_image_sha256`.

M19F also supports the same release policy knobs forwarded to M19B:

- minimum distinct device profiles;
- minimum benchmark runs;
- max text-prefill p95;
- max decode/token p95;
- max peak PSS;
- max thermal status;
- optional max speech-prefill p95;
- optional required energy counter;
- optional maximum absolute battery energy-counter delta.

Distinct profile identity is:

```text
manufacturer / model / sdk_int / abi
```

These are real physical-device observations. Repository CI fixtures do not
qualify as production evidence.

## M19B handoff

After the cheap provenance checks pass, M19F calls the existing
`vn97-release-candidate` implementation rather than duplicating its logic.

M19B then:

- loads the canonical unified checkpoint;
- parses the tokenizer;
- reconstructs VN97MI1 from the exact deployment geometry;
- requires reconstructed VN97MI1 SHA to equal VN97PRODCAMP1;
- reconstructs mobile footprint and reapplies the campaign budget;
- validates speech-training report identity;
- rechecks all device evidence against the reconstructed model image and
  configured limits;
- atomically builds VN97RC1.

M19F reloads that VN97RC1 and again requires its:

- selected candidate;
- checkpoint;
- tokenizer;
- production report;
- speech report;
- model-image identity;
- device evidence set

to equal the intake chain.

## VN97INTAKE1

A successful intake emits canonical strict JSON:

`production-intake.vn97intake1`

Schema: `VN97INTAKE1`.

It binds:

- selected candidate ID;
- VN97CAMP2 report SHA;
- language winner checkpoint SHA;
- VN97PRODCAMP1 report SHA;
- VN97SPEECHTRAIN1 report SHA;
- unified production checkpoint SHA;
- tokenizer SHA;
- production VN97MI1 SHA;
- sorted physical-device evidence SHA identities;
- distinct device-profile count;
- final VN97RC1 manifest SHA.

A strict parser is included for independent audit.

## Atomic intake bundle

The requested output directory must not already exist.

M19F stages in a same-parent temporary directory and publishes only after M19B
and all cross-checks pass.

The final bundle is:

```text
production-intake.vn97intake1
language-campaign-report.json
release-candidate/
  model.vn97ck1
  tokenizer.vn97tk1
  production-campaign-report.json
  speech-training-report.json
  device-evidence/
  release-candidate.vn97rc1
```

The copied language campaign report is rehashed while copying and must still
equal the verified VN97CAMP2 report identity.

The large pre-speech language checkpoint is deliberately not duplicated into
the intake bundle. Its identity remains bound by VN97CAMP2, VN97PRODCAMP1 and
VN97INTAKE1.

## CLI

Example after real campaign execution and phone evidence collection:

```text
vn97-production-intake \
  --language-campaign-dir /production/language \
  --production-campaign-dir /production/unified \
  --device-evidence /phones/device-a/vn97-mobile-evidence.json \
  --device-evidence /phones/device-b/vn97-mobile-evidence.json \
  --min-distinct-device-profiles 2 \
  --min-device-runs 5 \
  --max-text-prefill-p95-ms 2000 \
  --max-text-decode-p95-ms-per-token 200 \
  --max-speech-prefill-p95-ms 3000 \
  --max-device-thermal-status 5 \
  --output-dir /production/intake
```

The resulting release candidate is:

```text
/production/intake/release-candidate
```

and can be passed directly to M19E/M19D:

```text
vn97-production-release \
  --preflight-only \
  --release-candidate-dir /production/intake/release-candidate \
  ...
```

## Running the real campaign

M19F intentionally reuses the existing canonical executors.

First run the language campaign on real governed training data:

```text
vn97-campaign \
  --input <training> \
  --validation-input <validation> \
  --release-input <sealed-release> \
  --campaign <VN97CAMPDEF1> \
  --max-parameters <budget> \
  --max-validation-loss <criterion> \
  --output-dir /production/language
```

Then finalize the fixed winner with real speech data:

```text
vn97-production-campaign \
  --language-campaign-dir /production/language \
  --speech-input <speech-training-manifest> \
  --speech-validation-input <speech-validation-manifest> \
  --speech-release-input <sealed-speech-release-manifest> \
  --max-speech-validation-loss <criterion> \
  --output-dir /production/unified
```

Then activate the exact resulting model image in the developer APK and collect
fresh M11C device evidence on representative physical phones.

M19J now automates the physical-phone step without changing the M19F gate:

```text
vn97-device-evidence-campaign \
  --manifest production-run.vn97run1 \
  --workspace-root <workspace> \
  --repository-root <VN97 checkout> \
  --serial <physical-phone>
```

It rebuilds the exact VN97PRODCAMP1 model image, provisions it through the
existing signed developer path, collects canonical VN97MOBEVID1 on explicit
real phones, reapplies the M19F criteria, and publishes the evidence directory
only after the whole requested device set passes.

Only after those real operations should `vn97-production-intake` be run.

M19G can now freeze this entire production configuration in one `VN97RUN1`
manifest and drive the same canonical stages through:

```text
vn97-production-run --stage train
-> physical evidence capture
-> vn97-production-run --stage intake
```

The M19G runner does not replace M19F; it invokes this same intake path and then
re-opens VN97INTAKE1/VN97RC1 for identity verification.

## Honest boundary

M19F completes the software handoff for a real campaign.

It does not create:

- training corpora;
- sealed evaluation corpora;
- GPU/CPU training compute;
- real phone thermal/memory/latency measurements;
- publisher private keys;
- Android signing credentials.

Therefore this milestone can make the pipeline ready to ingest real artifacts,
but cannot truthfully claim a production winner has already been trained or a
final production APK exists until those external inputs are actually supplied.
