# VN97-R2D8 Production Curriculum Scheduling

## Purpose

R2-D8 controls **which task-family shards are seen, in what order, and at what
token share** during production training.

It does not add another model, adapter or backend. The same VN97-R2 weights,
R2-D7 streaming trainer, optimizer and resume semantics remain canonical.

The curriculum itself becomes immutable evidence identified by a
`VN97R2D8PLAN1` plan ID.

## Stage-aware family policy

R2-D8 prevents later task families from leaking into earlier production stages.

Allowed families are:

| Production stage | Allowed task families |
| --- | --- |
| `dense_pretrain` | `language`, `reasoning` |
| `instruction_reasoning` | `instruction`, `language`, `reasoning` |
| `tool_action` | `action`, `instruction`, `language`, `reasoning`, `tool` |
| `capability` | `action`, `capability`, `instruction`, `language`, `reasoning`, `tool` |

This is a stage-isolation rule, not a claim that any particular mixture is
universally optimal.

## Why one primary family per source manifest

R2-D6 permits a sealed source corpus to declare more than one task family.

For weighted curriculum accounting, counting one shard simultaneously as
`language` and `reasoning` would make the requested ratios ambiguous.
R2-D8 therefore requires an explicit **primary family** for every training
source manifest.

The selected primary family must already be declared by that source in the
R2-D6 index.

## Definition

Example dense-pretrain definition:

```json
{
  "schema": "VN97R2D8DEF1",
  "stage": "dense_pretrain",
  "epochs": 3,
  "seed": 9710,
  "family_weights": {
    "language": 0.7,
    "reasoning": 0.3
  },
  "primary_family_by_manifest": {
    "<manifest-sha-1>": "language",
    "<manifest-sha-2>": "reasoning"
  }
}
```

Requirements:

- weights are finite and strictly positive;
- weights sum to exactly 1 within a numerical tolerance;
- every training source manifest has exactly one assignment;
- no unknown/unused manifest assignment is allowed;
- every weighted family must have at least one training shard;
- the primary family must be declared by that D6 source.

R2-D8 does not hard-code one universal language/reasoning ratio. The exact
chosen policy is part of the plan identity and later validation evidence.

## Deterministic weighted scheduling

Each D6 epoch has a baseline token budget equal to one full pass of the D6
training target-token total.

R2-D8 schedules whole immutable shards using weighted fair progress:

1. group training shards by primary family;
2. deterministically shuffle each family pool from `seed + epoch` evidence;
3. choose the family furthest behind its normalized target-token progress;
4. append its next shard;
5. cycle deterministically through that family if additional visits are needed;
6. stop after the baseline token budget is reached and every weighted family
   has been visited.

The compiled epoch records:

- exact ordered shard IDs;
- order digest;
- family shard-visit counts;
- family target-token counts;
- requested weights;
- realized weights;
- absolute weight error;
- baseline token budget;
- actual scheduled target tokens.

## Granularity gate

Curriculum weights must be achievable with the available shard granularity.

For every epoch, R2-D8 requires:

```
abs(realized_weight - requested_weight) <= 0.10
```

for every weighted family.

If a family has only one very large shard and the requested ratio cannot be
approximated within this bound, plan compilation fails. The correct response is
to seal finer-grained R2-D6 corpora/shards, not to silently accept a materially
different curriculum.

## Build and verify

Compile against the exact R2-D6 package:

```bash
vn97-r2-curriculum build \
  --corpus-package /data/r2d6-corpus \
  --definition /data/r2d8-definition.json \
  --output /data/r2d8-plan.json
```

Verify later:

```bash
vn97-r2-curriculum verify \
  --corpus-package /data/r2d6-corpus \
  --plan /data/r2d8-plan.json
```

Verification does not merely verify the plan SHA. It recompiles the entire
schedule from the D6 index and definition evidence and requires byte-equivalent
semantic content.

## D7 integration

The production streaming CLI now requires:

```
--curriculum-plan /data/r2d8-plan.json
```

For dense pretraining the plan must have:

- stage `dense_pretrain`;
- the same D6 index ID;
- the same epoch count as the trainer;
- the same deterministic seed as the trainer.

D7 uses the exact R2-D8 shard order instead of its legacy per-epoch shuffle.

The D7 run identity now includes the curriculum plan ID. Therefore changing
weights, assignments, shard order, seed or epoch count invalidates an existing
resume checkpoint.

The best dense checkpoint also records the curriculum plan ID.

## Resume semantics

The R2-D7 cursor remains:

```
epoch -> shard_position -> record_index -> window_index
```

Because `shard_position` is now a position inside the immutable R2-D8 epoch
schedule, repeated shard visits are unambiguous.

Resume checkpoints still occur only at optimizer boundaries. A resumed run
recomputes the R2-D8 plan and epoch-order digest before accepting the cursor.

## What R2-D8 does not claim

A curriculum matching its requested weights is not proof that those weights
are optimal.

R2-D8 provides reproducibility and stage isolation so later held-out
experiments can compare policies scientifically without hidden changes in data
order.

Production progression remains:

```
dense_pretrain
-> instruction_reasoning
-> tool_action
-> capability
-> fast_path_alignment
-> fresh_validation
-> QAT
-> mobile_lowering
```

The later validation gate, not the curriculum definition itself, decides
whether trained intelligence is good enough to advance.
