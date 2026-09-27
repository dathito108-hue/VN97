# VN97-R2D10 Production Source Adapter Pack

## Purpose

R2-D10 is the source-normalization layer immediately before R2-D9.

It converts pinned raw JSONL sources into one canonical VN97 chat format while
preserving a reproducible source lock and receipt.

The production data path is now:

```
upstream source / exported dataset
-> R2-D10 source lock + adapter
-> canonical chat JSONL + R2-D10 receipt
-> R2-D9 global dedup / split / seal
-> R2-D6 scale evidence
-> R2-D8 curriculum
-> R2-D7 streaming training
```

R2-D10 does not bypass R2-D9. D9 remains the final campaign-wide dedup and
holdout trust boundary.

## Source lock

The lock schema is `VN97R2D10LOCK1`.

Example:

```json
{
  "schema": "VN97R2D10LOCK1",
  "profile_id": "vn97-production-intelligence-v1",
  "shard_target_training_records": 10000,
  "validation_fraction": 0.01,
  "release_fraction": 0.01,
  "sources": [
    {
      "source_id": "reasoning-source-v1",
      "origin": "https://example.org/dataset",
      "revision": "release-or-commit-id",
      "license": "MIT",
      "license_approved": true,
      "family": "reasoning",
      "adapter": "question_answer",
      "path": "raw/reasoning.jsonl",
      "expected_sha256": "<64 lowercase hex>",
      "expected_records": 100000,
      "max_bytes": 1073741824
    }
  ]
}
```

The D9 shard and holdout policy is part of the D10 lock identity. Changing
those values creates a different source-pack fingerprint.

Every source requires:

- explicit source ID;
- origin;
- pinned revision/release identifier;
- explicit license;
- `license_approved=true`;
- exactly one canonical R2-D9 family;
- one declared adapter;
- local relative non-symlink path;
- exact raw SHA-256;
- exact raw JSONL record count;
- maximum input byte bound.

Two source entries may not point to the same physical file.

## Supported adapters

### `messages`

Raw:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

Allowed for any canonical family. The conversation must contain a user turn
and end with an assistant target.

### `dolly`

Raw fields:

```
instruction
context
response
category
```

Allowed for `instruction` or `language`.

### `gsm8k`

Raw fields:

```
question
answer
```

Allowed only for `reasoning`.

### `instruction_io`

Raw fields:

```
instruction
input
output
```

Allowed only for `instruction`.

### `question_answer`

Raw fields:

```
question
answer
```

Allowed for `language` or `reasoning`.

### `prompt_response`

Raw fields:

```
prompt
response
```

Allowed for `language` or `instruction`.

### `tool_trace`

Raw fields:

```
prompt
tool_name
arguments
result
response
```

Allowed only for `tool`.

It becomes:

```
user      -> prompt
assistant -> [TOOL_CALL] ...
system    -> [TOOL_RESULT] ...
assistant -> final response
```

The tool call is therefore an assistant target while the tool result remains
context.

### `action_trace`

Raw fields:

```
observation
action
result
response
```

Allowed only for `action`.

It becomes:

```
user      -> observation
assistant -> [ACTION] ...
system    -> [ACTION_RESULT] ...
assistant -> final response
```

### `capability_demo`

Raw fields:

```
capability
instruction
response
```

Allowed only for `capability`.

The capability label is placed in system context and the response remains the
assistant target.

## Fail-closed conversion

R2-D10 does not silently skip malformed raw examples.

Every locked raw record must be valid for its selected adapter. One malformed
record fails the source-pack build.

This is deliberate: large public datasets that need filtering or schema repair
should have a source-specific export/conversion step with its own receipt,
then feed that immutable normalized export into R2-D10.

## Build

```bash
vn97-r2-source-pack build \
  --lock /data/source-lock.vn97r2d10.json \
  --output-dir /data/r2d10-pack
```

Output:

```
r2d10-pack/
  source-lock.vn97r2d10.json
  r2d10-source-pack.json
  r2d9-definition.json
  normalized/
    <source-id-hash>.chat.jsonl
    ...
```

The normalized filename is derived from the source ID SHA-256, so long or
unusual source names never become unsafe filesystem names.

## Receipt evidence

Each source receipt records:

- source identity;
- origin/revision/license/family;
- adapter;
- raw SHA-256, bytes and records;
- normalized relative path;
- normalized SHA-256, bytes and records.

The pack ID binds:

- canonical lock fingerprint;
- normalized lock file SHA-256;
- all source receipts;
- exact D9 handoff SHA-256.

## Verify

```bash
vn97-r2-source-pack verify \
  --pack-dir /data/r2d10-pack
```

Verification checks:

- pack identity;
- source-lock identity;
- normalized file hashes/sizes/record counts;
- every normalized conversation against the canonical `messages` adapter;
- exact D9 handoff content;
- D9 shard and holdout policy.

Any modification to normalized data or the D9 definition fails verification.

## Adding real production datasets

R2-D10 intentionally separates **adapter schema support** from **dataset
approval**.

Adding a dataset requires:

1. review its license/terms for the intended use;
2. pin an immutable upstream release/revision;
3. obtain or export a raw JSONL snapshot;
4. record exact SHA-256 and record count;
5. select or add a source-specific adapter;
6. add it to a reviewed D10 lock;
7. build and verify the pack;
8. hand the generated definition to R2-D9.

This lets VN97 add language, reasoning, instruction, tool, action and capability
sources incrementally without changing the model architecture or downstream
training formats.
