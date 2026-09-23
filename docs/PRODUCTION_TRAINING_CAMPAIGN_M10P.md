# M10P — Production VN97 Training Campaign / Best-Checkpoint Promotion

M10P turns one-off M10M training into a deterministic multi-candidate campaign.

## Fair comparison contract

Every candidate in one campaign uses the same:

- local training dataset;
- local held-out validation dataset;
- learned VN97TK1 tokenizer;
- sequence length/stride;
- batch size and epoch count;
- weight decay/gradient clipping;
- M10N validation criteria.

Training and validation inputs are compared by filesystem device/inode identity, so the same
physical file cannot be reused through a hard-link alias.

A candidate may vary only the explicit manifest fields:

- d_model;
- n_layers;
- d_state;
- embedding_rank;
- seed;
- learning_rate.

Candidate identity is the first 16 hex characters of SHA-256 over canonical candidate JSON.

## Mobile budget

`--max-parameters` is mandatory.

The exact model parameter count is first computed by instantiating the canonical VN97LanguageCore
on PyTorch's `meta` device, so an oversized candidate is rejected **before real weight storage is
allocated or training starts**. The count is asserted again after real model materialization.

## Eligibility and ranking

After native VN97 training, each candidate is evaluated on the exact shared holdout.

Only candidates passing the existing M10N `VN97ReleaseCriteria` are eligible.

Promotion ordering is fixed:

1. lower validation mean loss;
2. higher validation token top-1 accuracy;
3. lower parameter count;
4. lexicographically smaller deterministic candidate ID.

There is no manual "winner" override inside the campaign runner.

## Promotion artifact

Only the selected candidate is written as:

- `model.vn97ck1`;
- shared `tokenizer.vn97tk1`;
- `campaign-report.json` with schema `VN97CAMP1`.

The checkpoint is reloaded through M10L before it becomes an eligible observation, binding promotion
to a valid canonical deployment checkpoint.

The promoted checkpoint then flows into M10N for held-out release verification and signing.

## Campaign manifest

`VN97CAMPDEF1` example:

```json
{"schema":"VN97CAMPDEF1","candidates":[
  {"d_model":128,"n_layers":4,"d_state":8,"embedding_rank":32,"seed":97,"learning_rate":0.0003},
  {"d_model":192,"n_layers":6,"d_state":16,"embedding_rank":48,"seed":98,"learning_rate":0.0002}
]}
```

The manifest is bounded, strict UTF-8 JSON, rejects duplicate object keys and duplicate candidate IDs,
and is never allowed to introduce another architecture/backend.

## Boundary

M10P automates experiment selection; it does not create quality from insufficient data or compute.
The selected model still has to pass M10N before signing and APK bootstrap promotion.
