# M19I — Zero-Compute Production Preflight

M19I moves production feasibility and semantic input validation in front of
real VN97 training compute.

The canonical production sequence is now:

```text
VN97RUN1
  -> verify
  -> preflight
  -> language training
  -> speech production training
  -> physical VN97MOBEVID1
  -> intake
  -> VN97READY1
  -> signed APK
```

For the normal first run:

```text
vn97-production-run --stage train
```

M19I runs automatically before the first training subprocess.

There is no production-run flag to bypass it.

## What "zero-compute" means

M19I means zero **model training compute**.

It does not mean zero CPU work.

Preflight may perform deterministic preprocessing required to prove that
training is feasible:

- read/parse governed datasets;
- build dataset fingerprints;
- learn the deterministic byte-BPE tokenizer from the training split;
- encode language examples;
- construct supervised windows;
- parse/normalize PCM speech;
- tokenize speech transcripts;
- instantiate VN97 geometry only on Torch `meta` device to count parameters;
- calculate exact mobile-footprint geometry.

It does **not**:

- materialize real candidate model weights;
- create an optimizer;
- execute language forward/backward training;
- execute speech-adapter forward/backward training;
- evaluate trained quality;
- select a winner using holdout quality;
- write a campaign checkpoint;
- write VN97CAMP2/VN97PRODCAMP1.

## Canonical report

M19I emits canonical strict JSON:

```text
VN97PREFLIGHT1
```

The report binds to the exact VN97RUN1 manifest SHA-256.

It records:

### Language

- training/validation/sealed-release semantic dataset SHA-256;
- record counts;
- tokenizer SHA-256/byte count/vocab size;
- train/validation/release supervised-window counts;
- validation/release supervised target-token totals.

### Speech

- train/validation/sealed-release semantic speech dataset SHA-256;
- example counts;
- validation/release target-token totals;
- maximum observed audio frame count.

The speech dataset identities use the existing M11C loader contract, including
decoded PCM fingerprints.

### Candidates

For every VN97CAMPDEF1 candidate:

- deterministic candidate ID;
- exact parameter count from a meta-device model probe;
- language-only model-image bytes;
- language recurrent-state bytes;
- speech-enabled model-image bytes;
- speech-enabled recurrent-state bytes;
- admission/rejection status.

### Blockers

Stable blocker codes are emitted for feasibility conditions that can be known
before training, including:

- requested training device unavailable;
- impossible validation target-token minimum;
- impossible sealed release target-token minimum;
- impossible speech validation/release example minimums;
- impossible speech validation/release token minimums;
- no candidate fits language parameter/mobile budgets;
- a candidate could legally win language selection but cannot fit the final
  speech-enabled mobile budget.

## Language semantic checks

M19I deliberately reuses the canonical campaign/data code.

It uses the same:

- `training_cli._load_records`;
- text/chat modes and max byte/example limits;
- `require_disjoint_dataset_splits`;
- deterministic `learn_byte_bpe`;
- language encoding rules;
- `VN97TrainingConfig`;
- `build_training_windows`;
- `VN97ReleaseCriteria`;
- VN97CAMPDEF1 candidate parser.

Therefore production input errors such as these are discovered before training:

- malformed UTF-8;
- malformed JSONL;
- invalid text/chat records;
- empty supervised chat target;
- train/validation/release semantic overlap;
- impossible sequence/stride/window configuration;
- max-window overflow;
- invalid candidate geometry;
- invalid release thresholds.

The release split is parsed/encoded only for semantic/window feasibility. It is
not evaluated and is not used to select a candidate.

The real M10R rule remains unchanged: quality evaluation of the sealed release
set happens only after validation has selected the winner.

## Tokenizer feasibility

The tokenizer is learned exactly as M10P would learn it:

```text
training corpus
  -> learn_byte_bpe
  -> VN97TK1 bytes
  -> tokenizer SHA-256
```

Preflight uses that tokenizer to prove that:

- language train/validation/release examples can be encoded;
- supervised windows exist;
- speech transcripts fit the configured target-token limit;
- all candidate geometry uses the correct final vocabulary size;
- model-image footprint includes the exact tokenizer byte count.

The tokenizer is not published by M19I. The actual campaign reconstructs it
again from the same VN97RUN1-bound inputs.

Any deterministic mismatch would later surface through VN97CAMP2 hashes.

## Candidate parameter and mobile feasibility

For every candidate M19I builds the exact VN97Config geometry.

Parameter count uses the existing M10P meta-device probe:

```text
with torch.device("meta")
```

No real candidate parameter storage is allocated.

The existing exact VN97MI1 footprint estimator then checks:

### Language admission

- `max_parameters`;
- language model-image budget;
- recurrent-state budget;
- language deployment tile geometry.

### Final speech-enabled admission

M19I also calculates the same candidate after the canonical
`AudioFrameAdapter` is added.

It applies the M11C:

- final model-image budget;
- final recurrent-state budget;
- final tile geometry.

This closes an important pre-training failure mode.

If a candidate is legal under M10P language budgets and therefore could become
the validation winner, but would certainly exceed the final M11C
speech-enabled mobile budget, the entire production preflight is BLOCKED.

Operators must then tighten the search space/budgets before spending training
compute.

M19I does not silently remove that candidate, because silently changing the
candidate search space would alter the campaign policy.

## Speech PCM and split checks

M19I uses the existing M11C `load_speech_manifest` for all three splits.

That loader verifies before speech training:

- strict UTF-8 JSONL;
- exact audio/text record shape;
- safe relative audio paths;
- no symlink escape;
- valid WAV;
- mono PCM16;
- 16 kHz sample rate;
- bounded 20 ms–30 s duration;
- non-truncated PCM payload.

It then uses the existing:

```text
require_disjoint_speech_splits
```

which rejects overlap by decoded-audio fingerprint and record fingerprint.

Therefore renamed copies of the same audio cannot cross train/validation/release
boundaries unnoticed.

## Speech frame/token feasibility

Using the canonical default AudioAdapterConfig geometry, M19I calculates frame
count exactly as the adapter's unfold operation would:

```text
1 + floor((samples - frame_size) / hop_size)
```

Every example must fit:

- `speech_max_frames`;
- `speech_max_target_tokens`.

Transcript target size matches M11C teacher forcing:

```text
<text> + tokenizer(transcript) + <eos>
```

M19I also proves that the validation and sealed-release splits contain enough
examples/target tokens to ever satisfy their configured release criteria.

Loss/accuracy thresholds cannot be proven before training and remain real
quality gates in M10P/M10R/M11C.

## Device prerequisite

VN97RUN1 already forbids `device=auto`.

M19I additionally checks the explicit device before training:

- CPU is accepted;
- CUDA requires CUDA availability and a valid requested device index;
- MPS requires an available MPS backend;
- XPU requires an available XPU backend;
- device types without a deterministic preflight availability contract are
  rejected for production.

This prevents a production job from parsing all data and only then discovering
that the requested accelerator does not exist.

## Runner stages

M19I adds:

```text
vn97-production-run --stage preflight
```

On semantic success it prints canonical VN97PREFLIGHT1.

Exit code:

- `0`: report status READY;
- `2`: report status BLOCKED.

Optional:

```text
--preflight-report /production/evidence/run.vn97preflight1
```

writes the same canonical report bytes to a new path.

The following execution stages automatically run the same preflight first:

- `language`;
- `production`;
- `train`;
- `all`.

If preflight is BLOCKED, no training subprocess is launched.

`intake` does not rerun M19I because it performs no model training; M19G still
rehashes all VN97RUN1-bound source inputs before intake.

## Output collision ordering

Before automatic preflight, M19G checks the output paths relevant to the
requested training stage.

For example `train` requires all of these to be absent:

- language output;
- production output;
- intake output.

Thus an obviously invalid/overwrite-prone run fails even before tokenizer/PCM
preprocessing.

For `all`, missing physical evidence also fails before preflight/training.

## VN97RUNEXEC1 integration

M19I extends VN97RUNEXEC1 with:

```text
preflight_report_sha256
```

Stage consistency is now:

- `verify`: no preflight/output hashes;
- `preflight`: preflight hash only;
- `language`: preflight + VN97CAMP2;
- `production` / `train`: preflight + VN97CAMP2 + VN97PRODCAMP1;
- `intake`: existing campaign/intake/release-candidate hashes, no rerun
  preflight hash;
- `all`: preflight + full campaign/intake/release-candidate chain.

This makes it auditable that a training execution actually passed the exact
preflight report associated with its VN97RUN1.

## Failure classes

Some invalid inputs fail before a VN97PREFLIGHT1 object can be constructed.

Examples:

- malformed JSONL;
- semantic split overlap;
- corrupt WAV;
- transcript/window construction failure;
- invalid campaign option geometry.

These are hard preflight failures and also prevent training.

Conditions that can be summarized after successful parsing/geometry inspection
are represented as a BLOCKED VN97PREFLIGHT1 report.

## Production boundary

M19I cannot prove that a candidate will meet accuracy/loss quality thresholds.

Those require actual learning and remain guarded by:

- M10P validation;
- M10R sealed language release evaluation;
- M11C speech validation;
- M11C sealed speech release evaluation.

M19I proves that the run is structurally and computationally feasible enough to
justify spending real training compute.

It does not manufacture quality evidence, production intelligence, physical
phone evidence or signing credentials.

## Architecture boundary

M19I introduces no new model or trainer.

Canonical architecture remains:

```text
Code 1
  -> Code 2 hardware/mobile-aware upgrade
  -> one VN97 model
  -> one production campaign
```

The preflight consumes the same VN97RUN1 and uses the same canonical dataset,
tokenizer, candidate, speech and mobile-budget contracts as the actual
production executors.
