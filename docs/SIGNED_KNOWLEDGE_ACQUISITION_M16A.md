# M16A — Signed Knowledge Capability Acquisition Foundation

M16A establishes the first production capability-acquisition path that can add
durable knowledge to VN97 without adding executable plugins, a second model, a
parallel planner, or a second memory database.

## Canonical path

```text
user-selected VN97CAP1 + VN97SIG1 + independent Ed25519 public key
  -> existing content-addressed VN97 capability staging
  -> knowledge-only publisher scope verification
  -> strict VN97KN1 data validation
  -> explicit Review
  -> explicit Trust & Acquire
  -> crash-safe VN97KAL1 import ledger
  -> same activated VN97 embedding engine
  -> existing canonical VN97MEM1 semantic records
```

The existing M9 package rule remains authoritative: executable/plugin/script/
native-library/shared-library section roles are forbidden before M16A sees the
payload.

## Narrow knowledge package contract

M16A accepts only:

- `kind = knowledge`;
- capability ID `knowledge` or `knowledge.*`;
- exactly one data section;
- role `knowledge_records`;
- format `VN97KN1`;
- section size at most 8 MiB.

`VN97KN1` is canonical strict UTF-8 JSON:

```json
{"records":[{"content":"...","title":"..."}],"schema":"VN97KN1"}
```

Bounds:

- 1..2048 records;
- title <= 256 UTF-8 bytes;
- content <= 8192 UTF-8 bytes;
- no NUL data.

Each imported VN97MEM1 record is wrapped as canonical `VN97KNMEM1` data and
contains:

- `authority = evidence_only`;
- capability ID/version;
- package SHA-256;
- publisher key ID;
- record index;
- source origin/license;
- title/content.

The memory source is fixed to `vn97.capability.knowledge`.

Imported text is therefore context/evidence, never executable authority.

## Separate knowledge publisher trust

M16A deliberately does not widen the existing model-publisher trust registry.

`VN97KnowledgePublisherTrustRegistry` persists a separate
`VN97KPTR1` registry with a fixed hard-coded policy:

- namespace: `knowledge` / `knowledge.*`;
- kind: `knowledge` only;
- Ed25519 public key exactly 32 bytes;
- key IDs cannot be rebound to different key bytes;
- up to 64 keys;
- app-private atomic replace + directory fsync;
- strict canonical JSON + symlink rejection.

Review only checks whether the independently supplied key is compatible. The key
is not persisted until **Trust & Acquire**.

## Two-phase acquisition

`VN97KnowledgeAcquisitionSession.review(...)`:

1. stages VN97CAP1/VN97SIG1;
2. checks key-ID compatibility;
3. verifies the signature under a temporary knowledge-only policy;
4. revalidates the exact package section;
5. parses strict VN97KN1;
6. returns provenance, publisher fingerprint, package/signature/payload hashes
   and record count.

It does not mutate VN97MEM1 and does not persist trust.

`acquireReviewed(...)`:

1. explicitly enrolls the exact knowledge publisher key;
2. re-verifies the staged package using the durable trust registry;
3. re-parses and compares the exact reviewed payload;
4. resumes or creates the durable import ledger;
5. embeds each `VN97KNMEM1` record using the same activated VN97 cognition
   engine;
6. appends it durably as SEMANTIC data to the caller's existing VN97MEM1.

No M6 grants are created and no dynamic M6 capability handler is registered.

## Crash-safe import / VN97KAL1

The per-package `VN97KAL1` ledger contains immutable acquisition identity,
progress, imported VN97MEM1 record IDs, and one write-ahead pending record.

Before each memory append M16A durably stores:

- pending record index;
- VN97MEM1 last record ID before append;
- expected canonical memory-content SHA-256.

After append it stores the returned VN97MEM1 record ID and advances the index.

If the process dies after the memory append but before the ledger checkpoint, the
next explicit acquisition of the same reviewed package scans only the bounded
post-checkpoint record-ID window and reconciles the exact source/content. It
therefore does not blindly append a duplicate.

The reconciliation window is bounded to 4096 memory IDs. If the process has
advanced beyond that bound, acquisition fails closed rather than guessing.

A completed package is idempotent: the exact signed package re-verifies durable
trust and returns the already imported record IDs without appending again.

## Production binding

`AndroidPlatformRuntime.createProductionKnowledgeAcquisitionSession(...)`
requires:

- one activated VN97 model;
- the existing canonical VN97MEM1 store;
- matching VN97MEM1 vector dimension.

It uses separate app-private stage/trust/ledger directories only for acquisition
metadata. The acquired knowledge itself goes into VN97MEM1; there is no second
memory engine.

## Security boundary

M16A does **not**:

- download packages autonomously;
- execute code from a capability package;
- load plugins/scripts/native libraries;
- mutate the M6 registry;
- grant new external authority;
- replace the VN97 model;
- treat imported text as instructions or policy.

Future M16 milestones can add discovery/download orchestration only through
existing M6-authorized external actions and must preserve the same signed,
data-only review/acquire boundary.

## Regression scope

The isolated M16A host contract covers:

- signed knowledge package review;
- no trust or VN97MEM1 mutation during review;
- explicit trust enrollment at acquisition;
- VN97MEM1 evidence-only wrapping;
- idempotent re-acquisition;
- publisher key-ID rebinding rejection;
- non-knowledge namespace/kind rejection;
- simulated crash after a durable memory append;
- recovery without duplicate memory records.

APK compilation is the production integration gate for the
`AndroidPlatformRuntime` binding.

## Next

M16B should expose a dedicated Android **Capability Acquisition** review surface
and then add bounded M6-authorized acquisition from user-selected/downloaded
artifacts while preserving explicit trust and activation.
