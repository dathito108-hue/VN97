# M16E — Acquisition Provenance Chain

M16E closes M16 by making every completed knowledge acquisition auditable as one
durable app-private provenance record.

## Why this exists

M16A-D already separated four security boundaries:

1. VN97 knowledge-gap proposal;
2. exact-URL M6-approved remote fetch;
3. signed publisher review;
4. explicit Trust & Acquire into VN97MEM1.

Before M16E, those stages were individually auditable but were not bound into one
completed acquisition identity. M16E adds that final link without changing any
authority.

## VN97KPROV1

A completed acquisition is stored as:

`<package_sha256>.vn97kprov1`

under the app-private no-backup knowledge-acquisition provenance directory.

The record binds:

- package SHA-256;
- capability ID/version;
- publisher key ID and public-key SHA-256;
- signature-envelope SHA-256;
- VN97KN1 payload SHA-256;
- source origin/license;
- exact VN97MEM1 record IDs produced by M16A;
- whether M16A reported an already-acquired package;
- optional M16D proposal ID;
- proposal capability namespace and evidence record IDs;
- optional M16C M6 fetch receipt ID;
- exact canonical HTTPS fetch URL;
- completion wall-clock timestamp.

The record itself has a deterministic SHA-256 `provenanceId`.

## Linking rules

Proposal identity is linked only when:

- `needed=true`; and
- the proposal capability ID exactly equals the reviewed signed capability ID.

Fetch identity is linked only when:

- the fetch was explicitly approved by M6; and
- the fetched artifact package SHA-256 exactly equals the reviewed package
  SHA-256.

Therefore an unrelated earlier proposal or remote download cannot be attached to
a later signed acquisition.

Manual/local package acquisition is valid and simply records empty proposal/fetch
fields.

## Storage integrity

VN97KPROV1 uses:

- strict UTF-8;
- strict canonical JSON;
- SHA-256 payload digest in the file header;
- bounded fields;
- non-symlink app-private directory and target validation;
- temp-file write + file fsync;
- atomic create;
- directory fsync;
- immediate post-write parse/verification.

The package SHA-256 is the file key.

Once a provenance file exists, its core completed acquisition identity is
immutable:

- package;
- capability;
- publisher;
- signature;
- payload;
- VN97MEM1 record IDs.

Repeated idempotent acquisition returns the existing provenance rather than
rewriting history.

## Failure semantics

VN97MEM1 import remains controlled by M16A and is committed before the completed
provenance record can know the final memory record IDs.

If provenance persistence fails after a successful M16A import, the app reports
the operation as incomplete rather than claiming a fully audited acquisition.
The user can re-review the same signed package; M16A's completed ledger returns
the existing memory IDs without duplicating VN97MEM1, allowing provenance to be
persisted on the next explicit acquisition attempt.

## Android surface

After Trust & Acquire succeeds, the Capability Acquisition screen now shows:

- provenance ID;
- proposal ID or `none`;
- M6 fetch receipt or `none`;
- package and publisher identity;
- VN97MEM1 record count.

This gives the user one durable identifier for the completed acquisition chain.

## Authority boundary

M16E is metadata-only. It does not:

- create M6 grants;
- fetch a URL;
- trust a publisher;
- write new semantic knowledge;
- execute imported data;
- register dynamic tool handlers;
- load scripts/native libraries/plugins;
- replace the VN97 model;
- add a second planner/model/memory system.

All authority remains in the existing M6 and M16A boundaries.

## Regression coverage

The isolated M16E host contract verifies:

- completed provenance save/load;
- deterministic provenance ID;
- idempotent repeated save;
- immutable core identity;
- digest-tamper rejection;
- proposal/capability mismatch rejection;
- fetch HTTPS requirement;
- valid manual acquisition without proposal/fetch lineage.

APK compilation validates production wiring across M16A/B/C/D and the Android
completion UI.

## M16 closure

With M16A-E the canonical acquisition chain is:

```text
same VN97 + VN97MEM1
  -> bounded knowledge-gap proposal
  -> human source choice
  -> exact-URL M6 approval
  -> bounded content-addressed VN97CAP1 download
  -> independent VN97SIG1 + Ed25519 publisher review
  -> explicit Trust & Acquire
  -> same-model embedding into existing VN97MEM1
  -> durable VN97KPROV1 lineage
```

M16 is architecturally complete at this boundary.

The next roadmap milestone is M17 — Controlled Self-Improvement. M17 must consume
M16 knowledge only as evidence and must not treat acquisition provenance as code
execution authority.
