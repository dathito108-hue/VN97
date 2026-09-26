# P4D — Generalization Repair

P4D follows the real P4C evidence:

- P4C optimization succeeded;
- P4C validation loss/top-1 improved strongly;
- P3 language retention stayed inside budget;
- the development diagnostic remained 0/12;
- direct generations exposed curriculum memorization, repeated phrases, textual
  assistant-marker leakage, and weak transfer to unseen prompts.

P4D repairs those failure modes without changing the locked one-model VN97
architecture.

## Parent

The only accepted parent is a complete P4C artifact:

```text
SHA256SUMS
model.vn97ck1
model.vn97mi1
p4c-report.json
tokenizer.vn97tk1
```

The P4C verifier requires exact file hashes, canonical VN97P4C1, ELIGIBLE status,
checkpoint/tokenizer/model-image identity, and parent-P3 corpus identity.

## Completion-aligned targets

Historical P3/P4C chat training supervised textual role-marker bytes, including the
assistant marker.

P4D adds a new training encoder that keeps role markers in context but masks them from
the supervised target. Only assistant content, its terminating newline, and terminal
EOS are trained for new P4D records.

Legacy P3 replay keeps the historical encoding unchanged to preserve the measured P3
language distribution.

## Generalization curriculum

P4D removes all split-revealing target patterns such as:

```text
TRAINING_TOKEN
TRAINING-MEM
Case training-...
```

The new curriculum uses natural values, multiple English/Vietnamese prompt families,
separate training/validation paraphrase families, and exact split disjointness.

Balanced categories remain:

- instruction following;
- bounded arithmetic/reasoning;
- contextual memory use;
- structured JSON;
- tool intent;
- authority behavior.

Defaults:

```text
training:   1,000/category = 6,000
validation:   150/category =   900
P3 replay:  2,000 records
```

## Strict generalization probe

P4D ranks the unseen validation prompts by SHA-256 and selects 30 per category,
180 total, for strict generated-answer measurement.

This probe is never used for gradient updates.

For plain-text tasks, generated text must match exactly after outer whitespace
trimming. For JSON tasks, the entire response must parse as the expected JSON value;
preamble/trailing prose fails.

P4D is eligible only when strict generated passes improve over the immutable P4C
parent, token-level validation does not regress, and P3 language retention remains
inside budget.

The old 12-task P4 file may be supplied as a development diagnostic, but it does not
control promotion because it has already been inspected repeatedly.

## Bounded repetition control

VN97 reference greedy inference now supports optional:

```text
repetition_penalty >= 1.0
no_repeat_ngram_size 0..32
```

Defaults remain backward-compatible:

```text
repetition_penalty = 1.0
no_repeat_ngram_size = 0
```

P4D evaluation uses 1.12 and a 4-token no-repeat n-gram to suppress obvious loops
without replacing model learning with an external backend.

## Fine-tune profile

Default P4D profile:

```text
parent                 exact P4C checkpoint
sequence length        256
logical batch            4
micro-batch               2
epochs                    2
learning rate          5e-5
P3 replay             2,000
progress every           50 steps
checkpoint every        100 steps
```

Training is resumable through the same identity-bound CUDA checkpoint mechanism
validated by P3/P4C.

## Command

```bash
vn97-p4d-generalization \
  --p4c-dir /kaggle/working/p4c-final \
  --p3-corpus-dir /kaggle/working/p3-corpus \
  --dev-suite /kaggle/working/held-out-p4.jsonl \
  --work-dir /kaggle/working/p4d-work \
  --output-dir /kaggle/working/p4d-final \
  --device cuda:0
```

## Output

P4D always preserves measured candidate evidence:

```text
SHA256SUMS
model.vn97ck1
model.vn97mi1
p4d-report.json
tokenizer.vn97tk1
```

Possible statuses:

```text
ELIGIBLE
REJECTED_VALIDATION
REJECTED_RETENTION
REJECTED_GENERALIZATION
```

Only ELIGIBLE may become the next canonical intelligence checkpoint.

P4D does not claim production intelligence. A later blinded, larger held-out suite is
still required before production promotion.
