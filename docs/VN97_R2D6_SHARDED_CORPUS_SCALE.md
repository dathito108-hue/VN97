# VN97-R2D6 Sharded Production Corpus Scale

## Purpose

R2-D6 separates **memory feasibility** from **data-scale feasibility**.

R2-D5 can prove that the exact 1B model execution path fits a target CUDA
device. That does not mean the small preflight corpus is large enough to train
the model to useful production intelligence.

R2-D6 therefore builds an immutable sharded corpus index from one or more
already-sealed `VN97CORPUS1` corpora.

No model training is started by R2-D6.

## Why sharding is required

The preflight corpus can fit in one small training/validation/release set.
Production-scale data should not require one giant JSONL file or one giant
in-memory window tensor.

Each sealed corpus becomes three independent immutable shards:

- training;
- validation;
- release.

R2-D6 copies each original `VN97CORPUS1` manifest into the package so source
origins and approved licenses remain auditable.

The resulting package is designed to be consumed later by a streaming
production trainer.

## Definition

Create a definition next to the sealed corpus directories:

```json
{
  "schema": "VN97R2D6DEF1",
  "corpora": [
    {
      "path": "corpus-language",
      "task_families": ["language"]
    },
    {
      "path": "corpus-reasoning",
      "task_families": ["language", "reasoning"]
    }
  ]
}
```

Corpus paths are relative to the definition and may not escape its directory.

Every input corpus is independently re-verified using the R2-D5
`VN97CORPUS1` gates.

R2-D6 additionally rejects any exact record that appears in two different
sealed corpora, even if both source corpora are individually valid.

## Build

```bash
vn97-r2-corpus-scale build \
  --definition /data/r2d6-definition.json \
  --tokenizer /data/tokenizer.vn97tk1 \
  --output-dir /data/r2d6-corpus \
  --sequence-length 128
```

The output contains:

```
r2d6-corpus/
  tokenizer.vn97tk1
  r2d6-corpus-index.json
  manifests/
    corpus-00000.vn97corpus1.json
    ...
  shards/
    training-00000.jsonl
    validation-00000.jsonl
    release-00000.jsonl
    ...
```

The ordering is deterministic by sealed corpus manifest identity, not the
order in the definition file.

## Evidence measured per shard

Each shard records:

- source corpus manifest ID and SHA-256;
- split;
- shard SHA-256 and byte count;
- record count;
- input-token count;
- supervised target-token count;
- canonical training-window count;
- task families;
- approved source-license names.

Token/window counts use the same VN97TK1
`encode_chat_completion_messages` segmentation and non-overlapping
sequence windows used by the current R2 dense path.

The release shards remain held out.

## Scale policy

R2-D6 reports supervised **target tokens per model parameter**.

The current project policy is:

- **scale floor:** 8 target tokens / parameter;
- **scale target:** 20 target tokens / parameter.

For a roughly 1B-parameter model, these are intentionally multi-billion-token
data gates.

They are project planning gates, not a guarantee that a model meeting the
ratio will reach a particular intelligence level. Validation quality is still
decided later by held-out multi-axis evaluation.

The CLI may override the ratios for experiments, but the exact policy is
stored inside the index identity.

For a small P2-derived preflight corpus it is normal and desirable for both
scale gates to fail.

## Verify

```bash
vn97-r2-corpus-scale verify \
  --package-dir /data/r2d6-corpus
```

Verification checks:

- index identity;
- tokenizer identity;
- architecture fingerprint;
- source manifest files/hashes;
- source manifest ID set;
- shard provenance;
- recomputed shard ID;
- shard bytes/SHA-256;
- split totals;
- scale arithmetic and gate decisions;
- release-held-out flag.

## What R2-D6 does not do

R2-D6 does not load the 1B model and does not allocate GPU memory.

It also does not feed the sharded corpus into the current R2-D3 in-memory
trainer. The current trainer still expects materialized training windows.

The next production block must add a deterministic **streaming shard trainer**
with resume identity over:

- R2-D6 index ID;
- shard position/order;
- record/window position;
- optimizer state;
- production checkpoint.

That streaming block must retain the R2-D3 measured-memory and dense-first
rules.
