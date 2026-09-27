# VN97-R2D9 Production Corpus Acquisition and Sealing Campaign

## Purpose

R2-D9 turns many **already normalized, explicitly license-approved chat JSONL
sources** into a large set of small immutable `VN97CORPUS1` seals suitable
for R2-D6 sharding and R2-D8 curriculum scheduling.

R2-D9 is deliberately not a general web crawler.

The production trust boundary is:

```
source-specific download / conversion receipt
-> normalized chat JSONL
-> pinned revision + SHA-256 + record count + license approval
-> R2-D9 global dedup / family assignment / split / seal
-> R2-D6 token-scale index
-> R2-D8 deterministic curriculum
-> R2-D7 streaming training
```

This keeps acquisition provenance auditable and prevents silent changes in
remote datasets from entering a production checkpoint.

## Canonical task families

R2-D9 accepts only families already recognized by R2-D8:

- `language`
- `reasoning`
- `instruction`
- `tool`
- `action`
- `capability`

Each normalized source has exactly one primary family.

A duplicate record found in two sources with the same family is deduplicated
deterministically by source ID order.

A duplicate record found under two **different** families is rejected. The
campaign must resolve that ambiguity explicitly instead of allowing one record
to affect two curriculum buckets.

## Definition

Example:

```json
{
  "schema": "VN97R2D9DEF1",
  "profile_id": "vn97-production-intelligence-v1",
  "shard_target_training_records": 10000,
  "validation_fraction": 0.01,
  "release_fraction": 0.01,
  "sources": [
    {
      "source_id": "licensed-language-source-v1",
      "origin": "https://example.org/dataset",
      "revision": "commit-or-release-id",
      "license": "Apache-2.0",
      "license_approved": true,
      "family": "language",
      "path": "sources/language.jsonl",
      "expected_sha256": "<64 lowercase hex>",
      "expected_records": 1000000,
      "max_bytes": 2147483648
    }
  ]
}
```

The source path is relative to the definition and may not escape the definition
directory.

The source must already be canonical chat JSONL:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

## Source identity gates

Before accepting a source, R2-D9 verifies:

- regular non-symlink local file;
- strict UTF-8 JSONL;
- canonical chat structure;
- explicit `license_approved=true`;
- canonical task family;
- exact expected SHA-256;
- exact expected record count;
- configured maximum byte bound;
- non-empty origin/revision/license/source ID.

The definition fingerprint is independent of source-list ordering because
sources are canonically sorted by source ID.

## Campaign-wide dedup

R2-D9 hashes every canonical chat record with the same VN97 R2 chat digest
used by pilot/training leakage checks.

Dedup is **global across every source in the campaign**, before train /
validation / release splitting.

For duplicates within one family:

```
lexicographically first source_id owns the record
```

The receipt records accepted and duplicate counts for every source.

For duplicates across families, the build fails.

## Deterministic split

For each family:

1. globally unique records are sorted by canonical record digest;
2. deterministic counts are calculated from
   `validation_fraction` and `release_fraction`;
3. the remaining records become training;
4. every split is non-empty.

This avoids random split drift between machines or resumed data campaigns.

## Seal granularity

`shard_target_training_records` controls the maximum intended training-record
granularity before R2-D6.

For a family:

```
desired_seals = ceil(training_records / shard_target_training_records)
```

Validation and release must each contain at least one record for every desired
seal.

If holdout volume is insufficient, R2-D9 fails instead of silently creating
coarser shards. The source campaign must add data or choose an explicitly
different shard target.

Training, validation and release records are balanced deterministically across
the seal count.

Each seal is created by the existing canonical `prepare_corpus()` path and is
immediately re-verified through the R2-D5 `VN97CORPUS1` verifier.

## Output

A successful campaign contains:

```
r2d9-campaign/
  r2d9-campaign.json
  r2d6-definition.json
  r2d8-primary-family.json
  seals/
    language-00000/
      corpus.vn97corpus1.json
      training.jsonl
      validation.jsonl
      release.jsonl
    language-00001/
      ...
    reasoning-00000/
      ...
```

`r2d6-definition.json` is directly consumable by:

```
vn97-r2-corpus-scale build
```

Every corpus entry contains exactly one task family.

`r2d8-primary-family.json` maps each sealed corpus manifest ID to the
family that R2-D8 should use as its primary-family assignment.

## Build

```bash
vn97-r2-corpus-campaign build \
  --definition /data/r2d9-definition.json \
  --output-dir /data/r2d9-campaign
```

Then verify independently:

```bash
vn97-r2-corpus-campaign verify \
  --campaign-dir /data/r2d9-campaign
```

Verification reopens every seal with the R2-D5 verifier and checks:

- campaign identity;
- every sealed manifest ID and SHA;
- split record counts;
- global sealed-record total;
- D6 handoff content and SHA;
- D8 family-map content and SHA;
- unique manifest IDs.

## Scaling toward the 1B data floor

R2-D9 itself does not claim a corpus is large enough.

After D9:

```
vn97-r2-corpus-scale build ...
```

computes canonical VN97TK1 supervised target-token counts and the R2-D6 scale
ratio.

The current production floor remains:

```
8 supervised target tokens / model parameter
```

with a planning target of:

```
20 supervised target tokens / model parameter
```

For a roughly 1B-parameter model, a serious production campaign therefore
requires multi-billion supervised target tokens. D9 is designed to allow that
data to be added incrementally as many small immutable seals rather than one
unmanageable file.

## Relationship to source-specific adapters

Different public or private datasets use different schemas. R2-D9 does not
hide that fact behind a generic scraper.

A source-specific adapter should:

1. pin an upstream release/revision;
2. verify its license separately;
3. convert the upstream schema into canonical VN97 chat JSONL;
4. produce a source receipt;
5. hand the resulting immutable JSONL to R2-D9.

Existing P2 adapters are examples of this pattern. Additional production
adapters can be added family by family without changing the R2 model,
R2-D6/R2-D8 formats, or training runtime.
