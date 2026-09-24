# P2 — Real Medium Pilot GitHub Workflow

The first real VN97 medium pilot is available as a manual GitHub Actions workflow:

`.github/workflows/p2-medium-pilot.yml`

It consumes one previously sealed `VN97-P2-Corpus-<commit>` artifact and runs the
fixed P2 candidate on CPU.

The workflow does not create or approve the corpus. Corpus preparation remains a
separate workflow with its own explicit source-license acknowledgement.

## Dispatch inputs

A real run requires three manual inputs:

- `corpus_run_id`: the exact GitHub Actions run that produced the corpus artifact;
- `corpus_commit`: the exact 40-hex repository commit bound by VN97P2BUNDLE1;
- `execution_acknowledgement`: exactly `RUN_P2_MEDIUM_CPU_PILOT`.

The explicit execution acknowledgement exists because this step can consume
meaningful GitHub Actions minutes.

The workflow is manual-only. Pushes and pull requests never start model training.

## Artifact handoff verification

Before importing PyTorch training state, the workflow downloads exactly:

`VN97-P2-Corpus-<corpus_commit>`

from the supplied corpus run ID.

`vn97-p2-pilot-handoff verify-corpus` then verifies:

- exact artifact root, corpus and evidence file sets;
- complete SHA256SUMS coverage;
- every file SHA-256;
- VN97P2BUNDLE1 canonical identity;
- the exact corpus repository commit;
- VN97CORPUS1 schema/profile/manifest identity;
- bundle-to-manifest identity and SHA-256;
- every training/validation/release split byte count, record count and SHA-256.

Only after this handoff passes does training start.

## Training

The workflow installs a CPU PyTorch runtime and the current VN97 package, then runs:

```bash
vn97-medium-pilot \
  --corpus-dir p2-corpus-artifact/corpus \
  --output-dir p2-medium-output \
  --device cpu
```

The candidate remains the frozen P2 profile:

- d_model 192;
- 6 layers;
- d_state 16;
- factorized rank 96;
- sequence length 256;
- batch size 2;
- 2 epochs;
- maximum 2,048 training windows;
- maximum 256 validation windows;
- maximum 256 sealed-release windows.

The workflow has a 120-minute hard timeout. A timeout is not a failed intelligence
judgement; it means the hosted CPU is not an economical compute target for this
fixed pilot. The same sealed corpus/profile can then move to rented GPU compute.

## VN97P2RUN1

After a successful VN97PILOT1, the workflow seals a second receipt:

`pilot-run.vn97p2run1.json`

VN97P2RUN1 binds:

- training repository commit;
- corpus repository commit;
- corpus GitHub run ID;
- training GitHub run ID;
- VN97P2BUNDLE1 ID and SHA-256;
- VN97CORPUS1 manifest ID;
- fixed P2 candidate ID;
- VN97PILOT1 SHA-256;
- campaign report SHA-256;
- checkpoint SHA-256;
- tokenizer SHA-256;
- VN97MI1 SHA-256;
- pilot profile SHA-256;
- explicit CPU device;
- Python version;
- PyTorch version;
- deterministic VN97P2RUN1 identity.

This makes the real pilot traceable from public source acquisition through the exact
trained checkpoint.

## Pilot artifact

A successful run uploads:

`VN97-P2-Pilot-<training-commit>-<training-run-id>`

containing:

```text
output/
  tokenizer.vn97tk1
  model.vn97ck1
  campaign-report.json
  model.vn97mi1
  pilot.vn97pilot1.json
  pilot-run.vn97p2run1.json

input-evidence/
  p2-corpus-bundle.vn97p2bundle1.json
  corpus.vn97corpus1.json

SHA256SUMS
```

Retention is seven days.

## Decision after P2

P2 is a pipeline/learning proof. It is not the production winner.

A valid VN97PILOT1 + VN97P2RUN1 proves that the real public corpus can pass the
canonical tokenizer, medium training, validation, sealed release evaluation,
checkpoint reload and VN97MI1 export path.

Only then should P3 spend rented GPU compute on multiple production candidates.
