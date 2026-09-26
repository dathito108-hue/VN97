# P4C — Core Instruction / Cognition Fine-Tuning

P4C is the first measured improvement step after P4A/P4B established that the real P3
winner could model language tokens but did not yet behave as a useful assistant on the
diagnostic task suite.

P4C keeps the locked architecture:

```text
Code 1 -> Code 2 hardware/mobile-aware -> one VN97 model
```

It does not introduce a second model, LLM backend, planner, memory engine or cloud
inference service.

## Parent model

The only accepted parent is the exact verified P3 final artifact:

```text
SHA256SUMS
campaign-report.json
model.vn97ck1
model.vn97mi1
p3-run.vn97p3run1.json
tokenizer.vn97tk1
```

P4C reuses the P4A verifier before any training starts. The P3 winner remains an
immutable baseline.

## Curriculum

P4C generates one deterministic VN97-native chat curriculum with six balanced
categories:

- instruction following;
- bounded arithmetic/reasoning;
- contextual memory use;
- strict structured JSON cognition;
- typed tool intent;
- deny-by-default authority behavior.

Default size:

```text
training   1,000/category = 6,000 records
validation   100/category =   600 records
```

Training and validation are generated from separate seeds and every prompt carries a
split-bound case identity, so the two sets are exactly disjoint.

The optional P4 diagnostic suite is also checked for exact prompt overlap. It is never
used to create gradients or decide the curriculum validation gate.

## Training profile

Default profile:

```text
sequence length        256
logical batch            4
micro-batch               2
epochs                    2
learning rate          1e-4
weight decay           0.01
gradient norm           1.0
progress interval        50 steps
resume checkpoint       100 steps
```

The runner reuses the same CUDA tensor-training implementation that P3 validated. P4C
sets a distinct `VN97 P4C` progress protocol so notebook output is unambiguous.

Resume state is identity-bound to the exact:

- P3 checkpoint;
- curriculum hashes;
- sequence/batch/micro-batch geometry;
- epochs;
- optimizer hyperparameters;
- seed.

A mismatched resume state is rejected.

## Promotion gate

Before training, P4C evaluates the generated validation curriculum on the immutable P3
parent.

After training, it evaluates the exact same validation set.

The candidate is publishable only if:

- validation mean loss does not increase; and
- validation token top-1 does not decrease.

This is a curriculum non-regression gate, not the final production intelligence gate.

If a diagnostic P4 suite is supplied, P4C records task-level pass rate before and after
training, but that diagnostic score does not affect gradient updates or the curriculum
gate.

## Command

```bash
vn97-p4-core-finetune \
  --p3-dir /kaggle/working/p3-fast/final \
  --diagnostic-suite /kaggle/working/held-out-p4.jsonl \
  --work-dir /kaggle/working/p4c-work \
  --output-dir /kaggle/working/p4c-final \
  --device cuda:0
```

The command is resumable when `--work-dir` survives interruption.

## Output

A successful P4C run publishes exactly:

```text
SHA256SUMS
model.vn97ck1
model.vn97mi1
p4c-report.json
tokenizer.vn97tk1
```

`p4c-report.json` uses schema `VN97P4C1` and binds:

- immutable parent P3 run/checkpoint/model image/tokenizer;
- deterministic curriculum profile and hashes;
- baseline/final curriculum validation;
- training geometry and result;
- output checkpoint/model-image identities;
- optional diagnostic before/after measurements.

Status `ELIGIBLE` means the fine-tuned candidate passed its P4C curriculum
non-regression gate. It does **not** mean VN97 is production-ready.

## Honest boundary

The current 12-task P4 file is a diagnostic suite that has already been inspected. It
is useful for measuring whether P4C fixes the observed failure modes, but it is not a
large blinded production benchmark.

A later P4 gate should use a larger independently prepared held-out suite before final
production promotion.

P4C therefore answers one bounded question:

> Can the exact P3 winner acquire core assistant behavior through continued VN97-native
> training without changing the architecture and without regressing its dedicated
> curriculum validation?

Only real training/evaluation evidence may answer yes.
