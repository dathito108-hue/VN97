# M19G — Reproducible Real Production Run Manifest

M19G makes a real VN97 production campaign reproducible as one bound execution
contract instead of a collection of manually copied shell commands.

It adds:

- canonical `VN97RUN1` production-run manifests;
- canonical `VN97RUNEXEC1` execution receipts;
- `vn97-production-run` staged orchestration;
- exact input-byte binding for language datasets, candidate search space,
  speech manifests and every referenced WAV;
- Git/source/runtime-environment binding;
- one manifest-controlled device-evidence policy and output layout.

M19G does **not** add another trainer, selector, model backend, evidence gate or
release path. It only invokes the existing canonical stages.

## Canonical execution chain

```text
VN97RUN1
  -> verify bound repository / inputs / environment
  -> M19I zero-compute production preflight
  -> vn97-campaign
  -> VN97CAMP2
  -> vn97-production-campaign
  -> VN97PRODCAMP1
  -> physical phone evidence collection
  -> vn97-production-intake
  -> VN97INTAKE1 + VN97RC1
  -> M19E/M19D release path
```

The invoked production stages remain:

- M10P/M10R `vn97-campaign`;
- M11C `vn97-production-campaign`;
- M19F `vn97-production-intake`.

## VN97RUN1

`VN97RUN1` is strict canonical UTF-8 JSON.

Top-level fields are exactly:

```text
schema
repository_commit
environment
inputs
language
speech
intake
outputs
```

Unknown fields, duplicate JSON object keys, non-canonical JSON, NaN/Infinity,
unsafe paths and invalid option names are rejected.

The SHA-256 of the exact canonical manifest bytes is the run identity used by
`VN97RUNEXEC1`.

## Repository binding

The manifest records the exact lowercase 40-hex Git commit.

Before any execution stage, the runner requires:

- the supplied repository root is a real directory;
- it is a VN97 source checkout;
- `git rev-parse HEAD` equals `repository_commit`;
- the tracked Git worktree is clean;
- the running `production_run_cli.py` bytes equal the file in that checkout;
- the running `production_run_manifest.py` bytes equal the file in that
  checkout.

Each training/intake subprocess receives:

```text
PYTHONPATH=<bound-repository>/src
```

first in its import path.

A different installed `vn97` package therefore cannot silently replace the
source code bound by the manifest.

## Runtime environment binding

The manifest records:

- exact Python version;
- exact installed Torch package version;
- platform system;
- platform machine architecture.

It also requires explicit language and speech `device` options. `auto` is
rejected for production-run manifests.

The `verify` stage reports whether the current environment matches without
starting training.

Any real execution stage refuses to run when it does not match.

This is a configuration/runtime reproducibility guarantee. It is **not** a
claim that floating-point training is bitwise deterministic across different
GPU models, drivers, BLAS/CUDA kernels or hardware revisions.

If such a rerun produces different checkpoint/model-image bytes, later
VN97PRODCAMP1, VN97INTAKE1 and VN97RC1 hashes expose that difference and old
physical-device evidence cannot pass against a new model-image identity.

## Bound language inputs

The manifest binds by relative path, byte count and SHA-256:

- one or more language training files;
- one or more language validation files;
- one or more sealed language release files;
- one `VN97CAMPDEF1` candidate definition.

All relative paths:

- use canonical POSIX spelling;
- may not be absolute;
- may not contain `..`;
- stay inside the production workspace.

The runner rehashes every file through a no-follow descriptor before execution.

Training/validation/release files must also be physically distinct by
filesystem device/inode.

M10R still owns semantic record-overlap detection during campaign execution.

## Candidate search-space binding

The candidate definition is re-parsed before execution.

M19G requires:

- schema `VN97CAMPDEF1`;
- 1–64 candidates;
- exact candidate keys;
- positive d_model/n_layers/d_state;
- non-negative seed;
- embedding rank null or in `[1, d_model)`;
- finite positive learning rate;
- no duplicate candidate identity after learning-rate normalization.

The file's exact bytes are also SHA-bound by VN97RUN1.

## Bound speech inputs

Each of the three speech splits contains:

- one bound speech JSONL manifest;
- a sorted exact list of every WAV file referenced by that manifest;
- raw byte count and SHA-256 for every WAV.

The runner parses each speech manifest before execution and requires the bound
audio set to exactly equal its references.

Train, validation and sealed release speech manifests must be distinct, and
their bound audio paths may not overlap.

M11C still applies its own semantic speech split/disjointness checks and PCM
format validation during the actual production campaign.

## Manifest-controlled stage options

`language.options` is a whitelist mapping directly to the canonical
`vn97-campaign` flags.

At minimum it requires:

- explicit `device`;
- `max_parameters`;
- `max_validation_loss`.

All existing M10P/M10R resource, tokenizer, training, mobile-budget and sealed
release criteria can be stored in this object.

`speech.options` maps only to the existing
`vn97-production-campaign` flags and requires:

- explicit `device`;
- `max_speech_validation_loss`.

`intake.options` maps only to existing M19F evidence-policy flags, including:

- minimum distinct device profiles;
- minimum benchmark runs;
- text/speech p95 limits;
- PSS limit;
- thermal limit;
- optional battery energy-counter policy.

No arbitrary command-line flag can be injected through VN97RUN1.

## Physical evidence directory

Physical phone evidence is different from a training input: it only exists
after the exact production model image has been created and activated.

Therefore VN97RUN1 binds:

- one canonical relative evidence directory;
- the M19F evidence policy;

but does not pre-bind evidence SHA values that do not exist yet.

At `intake` time the runner requires the evidence directory to contain only
regular `.json` files and parses each through the existing VN97MOBEVID1
parser.

The actual evidence SHA-256 identities are then permanently bound by
VN97INTAKE1 and VN97RC1.

The evidence directory may not overlap any production output directory.

## Output layout

VN97RUN1 fixes three non-nested output directories:

- language campaign output;
- unified production campaign output;
- final M19F intake output.

Output paths are relative to the production workspace and may not escape it.

A stage refuses to overwrite the output it is responsible for.

This prevents a "rerun" from silently replacing an existing result and calling
it the same execution.

## Stages

The command is:

```text
vn97-production-run \
  --manifest production-run.vn97run1 \
  --workspace-root /production/workspace \
  --repository-root /work/VN97 \
  --stage <stage>
```

Supported stages:

### verify

Rehashes and validates all manifest-bound inputs, checks Git/source identity and
reports:

- manifest SHA-256;
- repository commit;
- environment readiness;
- physical-evidence readiness.

It does not start training.

### preflight

Runs M19I semantic/feasibility validation without optimizer/backprop or real
candidate model-weight materialization.

It emits VN97PREFLIGHT1 and returns 0 for READY or 2 for BLOCKED.

The training stages `language`, `production`, `train` and `all`
automatically run the same preflight before launching any training subprocess.

### language

Requires downstream outputs to be absent and runs only canonical
`vn97-campaign`.

The result is re-opened through the M19F VN97CAMP2 inspector.

### production

Requires an existing valid language output, requires production/intake outputs
to be absent, and runs only canonical `vn97-production-campaign`.

The resulting VN97PRODCAMP1 output is re-opened through the M19F production
inspector.

### train

Runs `language` and then `production` under the same manifest.

This is the normal first physical production execution before phone evidence
exists.

### intake

Requires valid existing language + production outputs, requires physical
evidence, and runs only canonical `vn97-production-intake`.

It then re-opens VN97INTAKE1 and VN97RC1 and requires their candidate-manifest
identities to agree.

### all

Runs language + production + intake.

Because physical evidence must already exist before this stage starts, `all`
is mainly useful for an exact replay where evidence from the expected reproduced
model image is already available.

If evidence is absent, `all` fails **before** training compute is spent.

For a first campaign, use:

```text
train
-> capture physical evidence on the resulting model image
-> intake
```

## VN97RUNEXEC1

Every stage emits deterministic canonical JSON with schema `VN97RUNEXEC1`.

It binds:

- VN97RUN1 manifest SHA-256;
- repository commit;
- stage;
- environment readiness;
- device-evidence readiness;
- VN97CAMP2 report SHA when present;
- VN97PRODCAMP1 report SHA when present;
- VN97INTAKE1 SHA when present;
- VN97RC1 manifest SHA when present.

The receipt parser enforces stage consistency:

- `verify`: no preflight/output report hashes;
- `preflight`: VN97PREFLIGHT1 hash only;
- `language`: preflight + language report;
- `production` / `train`: preflight + language + production reports;
- `intake`: existing language + production + intake + VN97RC1 chain;
- `all`: preflight + full language + production + intake + VN97RC1 chain.

Use:

```text
--receipt /production/evidence/run.vn97runexec1
```

to atomically save the exact emitted receipt.

## Example VN97RUN1 shape

Hash/byte values below are placeholders only:

```json
{
  "environment": {
    "platform_machine": "x86_64",
    "platform_system": "Linux",
    "python_version": "3.12.7",
    "torch_version": "2.5.1"
  },
  "inputs": {
    "campaign_definition": {
      "bytes": 512,
      "path": "config/campaign.json",
      "sha256": "<64 lowercase hex>"
    },
    "language_release": [
      {"bytes": 1000, "path": "data/release.jsonl", "sha256": "<64 lowercase hex>"}
    ],
    "language_training": [
      {"bytes": 1000000, "path": "data/train.jsonl", "sha256": "<64 lowercase hex>"}
    ],
    "language_validation": [
      {"bytes": 100000, "path": "data/validation.jsonl", "sha256": "<64 lowercase hex>"}
    ],
    "speech_release": {
      "audio": [
        {"bytes": 32044, "path": "speech/release/001.wav", "sha256": "<64 lowercase hex>"}
      ],
      "manifest": {
        "bytes": 64,
        "path": "speech/release.jsonl",
        "sha256": "<64 lowercase hex>"
      }
    },
    "speech_training": {
      "audio": [
        {"bytes": 32044, "path": "speech/train/001.wav", "sha256": "<64 lowercase hex>"}
      ],
      "manifest": {
        "bytes": 64,
        "path": "speech/train.jsonl",
        "sha256": "<64 lowercase hex>"
      }
    },
    "speech_validation": {
      "audio": [
        {"bytes": 32044, "path": "speech/validation/001.wav", "sha256": "<64 lowercase hex>"}
      ],
      "manifest": {
        "bytes": 64,
        "path": "speech/validation.jsonl",
        "sha256": "<64 lowercase hex>"
      }
    }
  },
  "intake": {
    "device_evidence_dir": "device-evidence",
    "options": {
      "min_device_runs": 5,
      "min_distinct_device_profiles": 2
    }
  },
  "language": {
    "options": {
      "device": "cuda:0",
      "max_parameters": 100000000,
      "max_validation_loss": 3.0
    }
  },
  "outputs": {
    "intake": "out/intake",
    "language": "out/language",
    "production": "out/production"
  },
  "repository_commit": "<40 lowercase hex>",
  "schema": "VN97RUN1",
  "speech": {
    "options": {
      "device": "cuda:0",
      "max_speech_validation_loss": 3.0
    }
  }
}
```

The real file must use canonical compact JSON, not the pretty-printed example
above.

## Relation to M19F and final release

After:

```text
vn97-production-run --stage intake
```

the manifest-controlled intake directory contains the M19F release candidate.

That candidate goes unchanged into:

```text
vn97-production-release \
  --release-candidate-dir <workspace>/<outputs.intake>/release-candidate \
  ...
```

M19E then determines whether signing/build prerequisites are READY.

M19G therefore freezes **how the real model/evidence candidate was produced**;
M19E/M19D freeze **whether and how that candidate may be signed and shipped**.

M19H now provides the canonical authoring path for this manifest:

```text
vn97-production-seal bootstrap
-> populate real production workspace
-> vn97-production-seal seal
-> production-run.vn97run1
-> vn97-production-run --stage verify
```

The sealer auto-discovers and hashes the conventional language/speech workspace
instead of requiring operators to hand-author file byte counts and SHA-256
identities. The resulting file is still this exact VN97RUN1 schema and is
verified by the same M19G parser/verifier before publication.

## Honest reproducibility boundary

VN97RUN1 does not claim that:

- one GPU and another GPU will produce bit-identical floating-point training;
- a different driver/kernel/BLAS implementation is numerically identical;
- missing production data can be recreated from hashes;
- physical phone evidence can be simulated.

It guarantees that the declared code commit, core runtime package versions,
input bytes, candidate search space, training/evaluation policies, speech audio
corpus and stage outputs are explicitly bound and auditable.

If output identities differ, downstream cryptographic hashes expose the
difference instead of hiding it.
