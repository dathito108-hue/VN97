# M17B — Held-Out Candidate Evaluation

M17B evaluates a reviewed M17A self-improvement candidate against the exact
baseline it was bound to, without activating the candidate.

The canonical architecture remains:

```text
Code 1
  -> Code 2 hardware/mobile-aware
  -> signed VN97MI1 production model
  -> M17A reviewed improvement candidate
  -> M17B non-active held-out evaluation
```

There is still one VN97 model architecture, one native runtime, one planner, and
one sovereign memory path.

## Fixed VN97HELD1 suite

M17B defines a small fixed evaluation suite named `VN97HELD1`.

It contains bounded deterministic prompt/target pairs covering:

- arithmetic;
- sequence continuation;
- basic logic;
- factual completion;
- opposites;
- exact instruction following;
- constrained formatting;
- causal completion;
- numeric comparison;
- language completion;
- the VN97 evidence-vs-authority safety boundary.

The exact canonical suite JSON has a SHA-256 identity stored in every evaluation
record.

M17B does not claim this small suite proves general intelligence. It is a
regression gate for controlled promotion.

## Native metrics

Each model is evaluated with its own signed VN97TK1 tokenizer and a fresh native
runtime session per held-out case.

For each target token M17B measures:

- next-token negative log likelihood;
- top-1 correctness.

Because candidate tokenization can differ from baseline tokenization, quality is
normalized as total negative log likelihood per fixed target UTF-8 byte.

M17B also records:

- target token count;
- fixed target UTF-8 byte count;
- text prefill p95 latency;
- total evaluation time;
- VN97MI1 image bytes;
- audio projection presence;
- vision projection presence.

No model-as-judge or second AI is used.

## Candidate is never activated

The candidate remains a staged signed VN97CAP1 package.

For evaluation:

1. the staged package SHA-256 must equal the M17A candidate package SHA;
2. it must still be `model.language` / `weights`;
3. it must contain exactly one direct `model_image` section in VN97MI1 format;
4. that section is copied into a temporary app-cache VN97MI1;
5. the temporary file is SHA-256 verified against the package section digest;
6. the native runtime opens that file only through the evaluation-only model
   handle path;
7. the file is deleted immediately after evaluation.

The activation backend is not called.

M17B does not:

- call activation `prepare`;
- call activation `commit`;
- write VN97INV1;
- enroll a new publisher;
- create a runtime revision;
- change the active model.

The temporary extraction is necessary because a VN97CAP1 section offset is not
guaranteed to satisfy the native VN97MI1 four-byte offset-alignment contract.

## Exact baseline binding

Before evaluation, the current VN97INV1 model must exactly match the baseline
recorded by M17A:

- activation ID;
- artifact SHA-256;
- package SHA-256;
- capability version.

The same exact inventory identity is checked again after both model evaluations.

If the active baseline changes during the run, no evaluation record is accepted.

## Assistant resource handoff

The app executes evaluation under the existing process-wide sovereign execution
lock.

Before benchmarking, the foreground assistant releases its active native model
and runtime only when:

- no approval is pending;
- no foreground assistant turn is active.

This avoids keeping the production model plus evaluation copies resident at the
same time on memory-constrained phones.

After evaluation, the unchanged active VN97INV1 baseline is reopened.

## Canonical criteria

The initial M17B gate defaults are deliberately regression-oriented:

- candidate NLL/UTF-8-byte <= 102% of baseline;
- candidate top-1 accuracy may drop by at most 2 percentage points;
- candidate VN97MI1 bytes <= 125% of baseline;
- candidate prefill p95 <= 130% of baseline;
- if baseline has audio projection, candidate must retain it;
- if baseline has vision projection, candidate must retain it.

These thresholds make M17B a minimum non-regression/resource gate. They do not
yet authorize promotion and do not by themselves prove that the candidate is an
improvement.

A later M17 promotion milestone can require stronger objective-specific evidence.

## VN97IMPEVAL1

The first completed evaluation is persisted app-private as:

`<candidate_id>.vn97impeval1`

It binds:

- candidate ID;
- exact baseline activation/artifact;
- candidate package SHA-256;
- candidate VN97MI1 artifact SHA-256;
- exact VN97HELD1 suite SHA-256;
- criteria;
- baseline metrics;
- candidate metrics;
- pass/fail decision and reasons;
- completion wall-clock timestamp.

The record has a deterministic SHA-256 `evaluationId`.

Persistence uses:

- strict UTF-8;
- canonical strict JSON;
- SHA-256 file payload header;
- bounded fields;
- non-symlink app-private storage;
- fsync;
- atomic create;
- directory fsync;
- post-write parse verification.

The first evaluation is immutable.

On every load, M17B recomputes the canonical decision from the stored metrics and
criteria. A record whose serialized `passed` or reasons disagree with the
canonical decision is rejected.

It also requires the stored suite hash and total target bytes to match the
built-in VN97HELD1 suite.

## Android surface

The private Controlled Self-Improvement screen now provides:

- Review improvement candidate;
- Evaluate held-out candidate;
- Reject candidate.

The evaluation view shows:

- evaluation ID;
- suite SHA-256;
- pass/fail;
- baseline/candidate NLL per byte;
- baseline/candidate top-1 accuracy;
- baseline/candidate prefill p95;
- failure reasons.

There is still no Promote or Activate button in M17B.

## Authority boundary

M17B adds no new authority.

It does not:

- train weights;
- mutate weights;
- modify source code;
- modify VN97MEM1;
- create M6 grants;
- execute package code;
- activate the candidate;
- promote the candidate;
- replace the canonical planner/model architecture.

## Next

M17C should add a controlled promotion eligibility record that requires:

- a REVIEWED M17A candidate;
- a PASS M17B evaluation bound to the same candidate and exact baseline;
- the exact baseline still active;
- explicit user approval;
- the existing transactional activation coordinator.

Promotion must fail closed if any identity changed, and must retain rollback
evidence for the previous baseline.
