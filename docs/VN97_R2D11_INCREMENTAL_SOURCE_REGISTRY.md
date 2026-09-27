# VN97-R2D11 Incremental Production Source Registry

## Purpose

R2-D11 lets VN97 grow its production corpus in many independently sealed
batches without rebuilding or rewriting earlier batches.

The production data chain is:

```
D10 source pack
-> D11 admission / cross-batch exact dedup
-> D9 batch campaign
-> D6 exact token index
-> D11 campaign attachment / cumulative progress
-> D8 curriculum
-> D7 streaming training
```

D11 is intentionally a registry and ledger, not another data converter.

## Why an append-only ledger

A production corpus large enough for a roughly 1B-parameter model will not be
created in one step.

Rebuilding all prior source packs, seals and token indexes every time a new
batch arrives would be slow and would make provenance harder to audit.

R2-D11 therefore keeps old batch evidence immutable and appends a new ledger
generation for each state transition.

The ledger sequence is:

```
000000.json  genesis
000001.json  admit_pack
000002.json  attach_campaign
000003.json  admit_pack
000004.json  attach_campaign
...
```

Every generation records the SHA-256 of the previous generation.

Changing an old ledger file therefore breaks the chain.

## Registry definition

The definition schema is `VN97R2D11DEF1`.

Example:

```json
{
  "schema": "VN97R2D11DEF1",
  "sequence_length": 128,
  "family_weights": {
    "language": 0.35,
    "reasoning": 0.25,
    "instruction": 0.15,
    "tool": 0.10,
    "action": 0.10,
    "capability": 0.05
  }
}
```

The family weights are a **data-scale allocation policy**, not a claim that the
same mixture must be used for every training stage.

R2-D8 still controls the stage-specific curriculum.

Weights must:

- use canonical R2-D9 families;
- be strictly positive;
- sum to 1.

The registry binds:

- exact VN97TK1 tokenizer SHA-256;
- tokenizer vocabulary size;
- R2 1B architecture fingerprint;
- exact parameter count;
- sequence length;
- canonical 8 target-token/parameter floor;
- canonical 20 target-token/parameter planning target;
- family allocation weights.

## Family scale requirements

For a family with allocation weight `w`:

```
family_floor = ceil(parameter_count * 8 * w)
family_target = ceil(parameter_count * 20 * w)
```

D11 reports, per family:

- exact attached training target tokens;
- floor required tokens;
- floor missing tokens;
- floor passed;
- planning-target required tokens;
- planning-target missing tokens;
- planning-target passed.

The global floor is still:

```
parameter_count * 8
```

`production_floor_ready=true` requires both:

1. the global 8-token/parameter floor has been reached;
2. every configured family has reached its allocated floor.

This prevents one very large family from hiding a completely missing family.

## Initialize

```bash
vn97-r2-source-registry init \
  --definition /data/r2d11-definition.json \
  --tokenizer /data/tokenizer.vn97tk1 \
  --registry-dir /data/r2d11-registry
```

The tokenizer is copied into the registry and becomes part of registry
identity.

## Admission phase

Before a D10 pack is allowed to become a new production batch:

```bash
vn97-r2-source-registry admit-pack \
  --registry-dir /data/r2d11-registry \
  --pack-dir /data/r2d10-pack-001
```

Admission performs:

- full D10 pack verification;
- family-allocation policy check;
- canonical chat record digest calculation;
- normalized target-token counting using the registry VN97TK1 and sequence
  length;
- dedup within the new pack;
- cross-family duplicate rejection;
- exact cross-batch duplicate check against all prior admitted digest shards.

A record present in any previous admitted batch causes the new admission to
fail, even if the family is the same.

This is stronger than D9's within-campaign same-family dedup because D11 treats
each already admitted batch as immutable historical evidence.

## Digest shards

Every admitted D10 pack creates one immutable digest shard:

```
digests/<pack_id>.jsonl
```

Each line contains:

```json
{
  "schema": "VN97R2D11DIGEST1",
  "digest": "<record digest>",
  "family": "reasoning",
  "source_id": "..."
}
```

Digest lines are strictly sorted.

Adding a later batch never rewrites an earlier digest shard.

## Batch-local D9 and D6

After admission, only that new pack needs to be processed:

```
D10 pack
-> D9 campaign
-> D6 index
```

Earlier D9 seals and D6 packages remain unchanged.

This is the main incremental property of D11.

## Attach exact campaign evidence

After the admitted batch has been sealed and indexed:

```bash
vn97-r2-source-registry attach-campaign \
  --registry-dir /data/r2d11-registry \
  --pack-dir /data/r2d10-pack-001 \
  --campaign-dir /data/r2d9-campaign-001 \
  --d6-package /data/r2d6-package-001
```

D11 verifies the complete chain:

```
D10 source receipts
== D9 source receipts
D9 seal manifest IDs/families
== D6 source manifest IDs/families
D6 tokenizer
== registry tokenizer
D6 architecture
== registry architecture
D6 sequence length
== registry sequence length
D6 scale policy
== canonical 8/20
```

The D9 global unique-record count must also equal the D11 admission count for
that pack.

Only then are the D6 **training** target tokens added to the cumulative
registry totals.

Normalized D10 token counts are kept as planning evidence, but production
floor progress is based only on attached D6 training-token evidence.

## Registry structure

```
r2d11-registry/
  registry.json
  tokenizer.vn97tk1
  ledger/
    000000.json
    000001.json
    ...
  digests/
    <pack-id>.jsonl
    ...
  evidence/
    <pack-id>/
      source-lock.vn97r2d10.json
      r2d10-source-pack.json
      r2d9-campaign.json
      r2d6-corpus-index.json
```

Only small identity/evidence files are copied into the registry.

Large normalized JSONL files and D9/D6 shard contents remain in their original
batch directories.

## Verify

```bash
vn97-r2-source-registry verify \
  --registry-dir /data/r2d11-registry
```

Verification checks:

- registry identity;
- tokenizer identity;
- architecture and parameter count;
- canonical 8/20 policy;
- contiguous ledger generations;
- previous-generation SHA chain;
- snapshot identities;
- immutable digest hashes and sorted digest contents;
- exact absence of cross-batch digest overlap;
- copied D10/D9/D6 evidence hashes;
- cumulative state recomputed entirely from ledger events.

## Status

```bash
vn97-r2-source-registry status \
  --registry-dir /data/r2d11-registry
```

Status reports:

- admitted batches;
- attached batches;
- admitted unique records by family;
- normalized target tokens by family;
- exact attached D6 training target tokens by family;
- missing floor/target tokens per family;
- global tokens/parameter;
- `production_floor_ready`.

## Important interpretation

The 8/20 policy is a project scale gate.

It does not prove model quality.

Reaching the floor means the production corpus has enough supervised
target-token volume under the configured family allocation to justify entering
the expensive dense-training campaign.

Model quality still requires later held-out validation.

## Next production boundary

After D11 is in place, data work becomes incremental:

```
new reviewed source
-> D10 pack
-> D11 admit
-> D9/D6 for that batch only
-> D11 attach
-> inspect missing-token status
-> repeat
```

Once D11 reports `production_floor_ready=true`, the project can freeze the
corpus registry generation, compile the R2-D8 dense-pretrain curriculum against
the admitted/attached corpus set, and proceed to measured GPU preflight and
quota-bounded R2-D7 training.
