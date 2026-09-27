# VN97-R2D12 Registry Freeze and Virtual Multi-Batch Corpus

## Purpose

R2-D12 freezes one verified R2-D11 registry generation and exposes all attached
R2-D6 batches as one immutable **virtual corpus view**.

It does not copy or rebuild training/validation/release shard data.

The production path becomes:

```
D10 packs
-> D11 append-only registry
-> D9/D6 per newly admitted batch
-> D12 frozen virtual view
-> D8 stage projection + curriculum
-> D7 streaming trainer
```

## Why a virtual view

R2-D11 deliberately keeps every batch independent. That makes incremental data
growth auditable, but D8/D7 need one deterministic corpus identity for a
training campaign.

R2-D12 bridges those requirements:

- old D6 packages remain unchanged;
- no giant merged JSONL is created;
- no D6 shard is copied;
- the frozen view combines only immutable metadata;
- runtime mounts map logical batch IDs to actual D6 package directories.

Mount paths are not part of `view_id`. Moving the same verified D6 package to
another path therefore does not change corpus identity.

## Freeze precondition

A generation may be frozen only when:

```
admitted_batches == attached_batches
```

There may be no admitted-but-unattached D10 pack at that generation.

This prevents a freeze from silently omitting a pending production batch.

R2-D12 may freeze a corpus below the production scale floor. Such a view is
useful for status, curriculum planning and integration tests, but R2-D7
production training still refuses to allocate the 1B model until the selected
stage projection passes the canonical scale floor.

## Mount definition

Mounts are local runtime locators, not corpus identity.

Schema:

```json
{
  "schema": "VN97R2D12MOUNTDEF1",
  "packages": {
    "<pack-id-1>": "batches/batch-001/d6",
    "<pack-id-2>": "batches/batch-002/d6"
  }
}
```

Every path is relative to `--workspace-root`.

The mount pack set must exactly equal the set of D11 attached packs at the
frozen generation.

Each mounted D6 package is fully verified and its `index_id` must equal the
D6 identity already recorded by the D11 attach event.

## Build

```bash
vn97-r2-virtual-corpus build \
  --registry-dir /data/r2d11-registry \
  --generation 42 \
  --workspace-root /data \
  --mount-definition /data/r2d12-mounts.json \
  --output-dir /data/r2d12-view
```

Output:

```
r2d12-view/
  r2d12-view.json
  r2d12-mounts.json
  tokenizer.vn97tk1
  r2d11-registry.json
  r2d11-ledger/
    000000.json
    ...
    000042.json
```

There is intentionally no `shards/` directory.

## Frozen D11 evidence

The view copies only small D11 evidence files:

- exact registry configuration;
- every ledger generation from 0 through the frozen generation;
- tokenizer.

The view identity binds:

- registry ID;
- frozen generation;
- frozen snapshot ID;
- final ledger SHA-256;
- registry config SHA-256;
- the complete ordered ledger-chain SHA/snapshot-ID list;
- attached `pack_id -> d6_index_id` pairs;
- the virtual corpus index.

Runtime verification checks the copied ledger chain again:

```
generation
registry_id
snapshot_id
previous_ledger_sha256
file SHA-256
```

So the frozen view preserves the exact D11 history that selected its batches.

## Virtual index

R2-D12 creates a metadata-only D6-compatible index.

Each source manifest and shard receives:

```
batch_pack_id
```

The original D6 shard filename remains relative to its batch package.

The virtual index recomputes:

- combined corpus manifest IDs;
- combined source manifests;
- combined shards;
- split byte/record/input-token/target-token/window totals;
- production parameter count;
- tokens/parameter;
- canonical 8-token/parameter floor status;
- canonical 20-token/parameter planning-target status.

Its identity is `VN97R2D12INDEX1`.

## Verify

```bash
vn97-r2-virtual-corpus verify \
  --view-dir /data/r2d12-view \
  --workspace-root /data
```

Verification:

1. verifies `view_id`;
2. verifies tokenizer identity;
3. verifies copied D11 registry evidence;
4. verifies the complete frozen ledger chain;
5. resolves every runtime mount;
6. fully verifies every mounted D6 package;
7. requires each D6 `index_id` to equal frozen D11 evidence;
8. reconstructs the combined virtual index from mounted packages;
9. requires exact equality with the frozen virtual index.

Mounting another dataset at the same path therefore cannot impersonate the
frozen corpus.

## Stage projections

The full D12 view may contain:

```
language
reasoning
instruction
tool
action
capability
```

R2-D8 stage isolation still applies.

D12 derives a new metadata-only projection from the family weights of the D8
definition/plan.

For example, dense pretraining can project only:

```
language
reasoning
```

while later tool/action training can project its own approved family set.

Projection recomputes its own:

- manifest set;
- shard set;
- split totals;
- tokens/parameter;
- 8/20 scale flags;
- `index_id`.

This is important: a full registry can have large total data while a specific
training stage is still below its required scale.

## Compile D8 against a virtual view

```bash
vn97-r2-curriculum build \
  --virtual-view /data/r2d12-view \
  --workspace-root /data \
  --definition /data/r2d8-dense-definition.json \
  --output /data/r2d8-dense-plan.json
```

D8 derives the projection from the definition's stage and weighted families.

Verify:

```bash
vn97-r2-curriculum verify \
  --virtual-view /data/r2d12-view \
  --workspace-root /data \
  --plan /data/r2d8-dense-plan.json
```

The plan is bound to the projected virtual `index_id`, not to filesystem
mount paths.

## Stream D7 from a virtual view

After the selected dense-pretrain projection passes the scale floor and the
matching measured CUDA preflight has passed:

```bash
vn97-r2-stream-train \
  --virtual-view /data/r2d12-view \
  --workspace-root /data \
  --curriculum-plan /data/r2d8-dense-plan.json \
  --preflight-receipt /data/r2d5-preflight-receipt.json \
  --work-dir /kaggle/working/r2d7-work \
  --output-dir /kaggle/working/r2d7-output \
  --sequence-length 128 \
  --micro-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --precision fp16 \
  --epochs 1 \
  --seed 9710 \
  --device cuda
```

D7:

- verifies the D12 view and all mounted D6 packages before training;
- loads the copied frozen tokenizer;
- derives the exact same stage projection used by D8;
- binds the projected index ID into streaming/resume identity;
- resolves each scheduled shard through `batch_pack_id`;
- preserves the existing epoch/shard/record/window resume cursor;
- does not make runtime mount paths part of model/checkpoint identity.

## Relocation

If D6 package directories move, edit or regenerate only:

```
r2d12-mounts.json
```

with new relative locations under the workspace root.

The mounted package must still verify to the same frozen D6 `index_id`.

Because paths are locator evidence rather than corpus identity, `view_id` and
D8/D7 resume identity remain unchanged.

## What D12 proves

A verified D12 view proves:

- which D11 generation was frozen;
- the exact D11 ledger chain through that generation;
- the exact set of attached packs;
- the exact D6 index identity of every batch;
- the exact combined metadata/token-scale view;
- that currently mounted D6 packages reproduce that frozen metadata.

It does **not** prove model quality, GPU fit or training completion.

Those remain separate gates:

```
D12 stage scale
-> D5 measured GPU preflight
-> D8 curriculum
-> D7 quota-bounded dense training
-> held-out quality validation
```
