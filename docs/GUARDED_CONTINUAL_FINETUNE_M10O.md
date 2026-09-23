# M10O — Guarded Continual Fine-Tuning

M10O adds a safe iterative learning path for an existing VN97 deployment checkpoint.

It is not interrupted-optimizer resume. VN97CK1 remains a deployment checkpoint and never stores
pickle/object graphs or optimizer state.

## Canonical path

`parent VN97CK1 + same VN97TK1 + new local training JSONL + distinct held-out validation JSONL`
`-> baseline native VN97 validation`
`-> fresh AdamW fine-tune of the same VN97LanguageCore`
`-> final native VN97 validation`
`-> absolute M10N quality criteria + maximum allowed validation-loss regression`
`-> new VN97CK1 only if both gates pass`

The resulting checkpoint can then go through M10N for signing/release.

## Fixed identity

Fine-tuning loads the existing checkpoint through M10L and reuses the supplied VN97TK1 tokenizer.

The CLI does not expose model geometry or tokenizer-learning options, so d_model/layers/d_state,
embedding factorization and vocabulary cannot silently change during continual learning.

## Dataset separation

Training and validation input files are checked by device/inode identity. The same physical file,
including a hard-linked alias, cannot be used in both sets.

Both data sets retain M10M's bounded, local-only, no-follow JSONL loading.

## Quality gates

The final model must satisfy the M10N absolute release criteria:

- maximum validation loss;
- optional minimum top-1 accuracy;
- minimum held-out target-token count.

It must also satisfy:

`final_validation_loss <= baseline_validation_loss + max_validation_loss_increase`

The default allowed increase is zero.

No output directory/checkpoint/report is created until the gates pass.

## Output

`vn97-finetune` writes:

- `model.vn97ck1`;
- the unchanged `tokenizer.vn97tk1`;
- canonical `finetune-report.json` with schema `VN97FT1`.

The report binds parent/output checkpoint digests, tokenizer digest, training and validation dataset
digests, baseline/final validation metrics and optimization settings.

## Example

`vn97-finetune --checkpoint out/model.vn97ck1 --tokenizer out/tokenizer.vn97tk1 --input new-data.jsonl --validation-input holdout.jsonl --format chat --output-dir out-v2 --epochs 1 --learning-rate 1e-4 --max-validation-loss 3.0 --max-validation-loss-increase 0.0 --min-validation-target-tokens 1000`

## Intelligence boundary

M10O enables controlled iterative improvement of VN97 weights. It does not make a weak checkpoint
AGI automatically; quality remains determined by model capacity, training data, optimization and
the held-out evidence used by the gates.
