# M16D — Bounded Knowledge-Gap Acquisition Proposals

M16D lets the canonical VN97 model identify a bounded external-knowledge gap
without granting the model any new acquisition authority.

The milestone is deliberately **proposal-only**.

## Canonical path

```text
user goal
  -> bounded recall from the same open VN97MEM1
  -> same activated VN97 model
  -> strict VN97CAPGAP1 proposal
  -> human review
  -> optional human-chosen exact HTTPS URL
  -> M16C exact-URL M6 approval
  -> remote artifact bytes only
  -> M16A/M16B Review signed knowledge
  -> explicit Trust & Acquire
```

No proposal can skip any later boundary.

## Proposal evidence

Production M16D uses the exact active model and exact already-open VN97MEM1 owned
by the foreground assistant.

For each user goal it retrieves at most four bounded records using:

- semantic weight 0.80;
- recency weight 0.05;
- importance weight 0.15;
- semantic + episodic kinds;
- 30-day recency half-life.

Each evidence item is bounded to:

- source: 128 UTF-8 bytes;
- content: 1024 UTF-8 bytes.

The model receives only this bounded evidence plus the user goal.

## VN97CAPGAP1

The same VN97 inference engine is instructed to return exactly:

- `needed: bool`;
- `capability_id: string`;
- `topic: string`;
- `rationale: string`;
- `source_hint: string`;
- `confidence_bps: integer 0..10000`.

When `needed=false`, capability/topic/source-hint must all be empty.

When `needed=true`:

- capability ID must start with `knowledge.`;
- only lowercase data namespace characters are accepted;
- topic/rationale/source-hint are UTF-8 bounded;
- source hint is advisory source-type text only.

The runtime explicitly rejects source hints containing `://` or beginning with
`http`. Therefore the model cannot turn its proposal into an actionable remote
URL.

## Deterministic proposal identity

Every accepted proposal gets a SHA-256 proposal ID binding:

- user-goal SHA-256;
- recalled VN97MEM1 record IDs;
- canonical structured VN97CAPGAP1 response.

The proposal ID allows the UI/user to refer to an exact proposal without turning
it into a permission token.

## Prompt and output bounds

- user goal: max 4096 UTF-8 bytes;
- recalled evidence: max 4 records;
- prompt: max 16 KiB;
- model output: max 8 KiB;
- topic: max 512 UTF-8 bytes;
- rationale: max 2 KiB;
- source hint: max 512 UTF-8 bytes;
- generation budget: 384 new tokens.

Unknown JSON fields, non-integer confidence, invalid namespaces and malformed
structured output fail closed.

## Android surface

The Capability Acquisition screen now has **Analyze knowledge gap**.

The user enters a goal and sees:

- proposal ID;
- whether VN97 believes more knowledge is needed;
- confidence basis points;
- supporting VN97MEM1 record IDs;
- capability namespace;
- topic;
- rationale;
- source hint.

The screen explicitly states that the result is proposal-only.

M16D never auto-fills the M16C URL field. If the user wants to continue, the
user chooses an exact source URL and M16C still requires explicit M6 approval.

## Authority boundary

M16D does not:

- execute a tool;
- create an M6 request;
- fetch a URL;
- persist publisher trust;
- import a capability;
- write VN97MEM1;
- create executable code/plugins/scripts/native libraries;
- replace the VN97 model;
- add a second planner/model/memory system.

Proposal generation is also blocked while the normal foreground assistant has
an active turn or pending M6 approval, reusing the existing sovereign execution
lock.

## Regression coverage

The isolated M16D contract verifies:

- valid knowledge-gap proposal parsing;
- deterministic proposal identity even with JSON whitespace differences;
- no-need proposal constraints;
- rejection of actionable URLs;
- rejection of non-`knowledge.*` namespaces;
- rejection of unexpected output keys;
- confidence bounds;
- maximum four evidence records;
- goal size bound;
- UTF-8-safe evidence truncation.

APK compilation verifies the same-model/VN97MEM1 production binding and Android
proposal UI.

## Next

After M16D, M16 should be reassessed for architectural closure. A possible M16E
would be justified only if there remains a concrete acquisition gap such as
durable proposal history or explicit acquisition provenance linking. It should
not weaken the existing Review -> M6 approval -> publisher trust -> VN97MEM1
boundaries.
