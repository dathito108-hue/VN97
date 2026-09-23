# M10R — Sealed Final Holdout / Selection-Leakage Guard

M10R makes the production campaign a true three-way evaluation pipeline.

The canonical VN97 model architecture is unchanged. M10R changes only how
training evidence is partitioned and when each partition is allowed to affect a
decision.

## Three-way contract

A production campaign now requires three physically and semantically separate
datasets:

1. **training** — learns VN97TK1 and updates canonical VN97LanguageCore weights;
2. **validation** — admits/rejects candidates and performs deterministic M10P
   best-checkpoint ranking;
3. **release** — evaluates only the already-selected winner.

The release split never participates in candidate ranking.

## Leakage protection

M10R applies two independent checks before training:

- filesystem device/inode identity rejects reuse through hard links or the same
  physical file;
- canonical record fingerprints reject copied records across training,
  validation and release files.

Text records are fingerprinted from canonical `{"text": ...}` JSON. Chat
records are fingerprinted from canonical role/content message arrays. The
fingerprints are domain separated as `VN97DATAREC1`.

The tokenizer is still learned only from training records. Validation and
release text never enters tokenizer learning.

## Sealed release timing

Release JSONL is parsed early only so leakage can be rejected before expensive
training. It is not tokenized into model inputs and no model forward pass is
performed on it until the validation-selected winner is fixed.

After M10P deterministic selection:

`selected VN97CK1 -> canonical reload -> release windows -> release quality gate`

The selected checkpoint is reloaded from its exact deployment bytes before the
release evaluation.

If the winner fails the release gate, the campaign fails before creating the
output directory. It does **not** try the second-best candidate; doing so would
turn the release set into another model-selection set.

## CLI additions

`vn97-campaign` now requires `--release-input` and supports bounded release
controls:

- `--release-format`;
- `--release-max-input-bytes`;
- `--release-max-examples`;
- `--release-max-windows`;
- `--release-batch-size`;
- `--release-max-validation-loss`;
- `--release-min-validation-accuracy`;
- `--release-min-validation-target-tokens`.

If release quality thresholds are omitted, the corresponding validation
thresholds are reused.

## Report

A successful campaign writes `VN97CAMP2`.

The report preserves candidate validation observations and M10Q mobile
footprints, and adds:

- release dataset SHA-256;
- release quality criteria;
- final release loss / top-1 accuracy / target-token count / window count.

Only after the sealed release gate passes are `model.vn97ck1`,
`tokenizer.vn97tk1` and `campaign-report.json` published.

## Architecture boundary

M10R adds no teacher model, Transformer/LLaMA path, cloud inference service,
alternate backend or manual winner override.

The production path remains:

training
-> canonical VN97 candidates
-> M10Q mobile admission
-> validation/M10N quality eligibility
-> deterministic M10P selection
-> sealed final release gate
-> VN97CK1 + VN97TK1
-> signed M10N bootstrap release
-> Android activation.
