# P5D2 — Dual-GPU Float-Shadow Behavioral Distillation Pilot

P5D1 produced a sealed 600-record Falcon3-Mamba teacher corpus. P5D2 is the
first training stage for the 309M native VN97-RT target.

This stage is intentionally a **pilot**, not a claim that 600 teacher records
are enough to transfer the full intelligence of a 7B teacher.

## Why a pilot first

A 309M student initialized from scratch is expensive. Before expanding the
teacher corpus, P5D2 tests whether:

- the native 1536 x 32 x 16 VN97 target trains stably;
- float-shadow training produces a held-out teacher-imitation signal;
- P3 replay prevents immediate language collapse;
- any non-zero P4 behavior appears.

If no signal appears, the correct next action is to change initialization or
distillation targets, not to blindly spend more GPU hours.

## Two T4 candidates

Both Kaggle T4 GPUs are used concurrently.

Candidate 0:
- LR 8e-5
- seed 5201

Candidate 1:
- LR 4e-5
- seed 5201

The identical initialization seed isolates learning rate as the experiment.

## Data split

The 600 P5D1 teacher records are split deterministically:

- hold out 10 records from each of the 6 P4 categories;
- hold out 40 general-language records;
- total teacher holdout = 100;
- teacher training records = 500.

P5D2 also replays 600 deterministic P3 training records.

Training is capped at 1200 windows, sequence length 96, logical batch 4,
physical microbatch 1.

## Precision

P5D2 trains in full FP32 with float-shadow TernaryLinear weights.

No ternary STE/QAT is used yet.

This is slower than FP16, but the previous P5C experiment showed that a
premature low-precision alignment path can fail numerically. Ternary QAT stays
deferred until a float student demonstrates held-out learning.

## Output

Each candidate emits:

```text
p5d2-cN-final/
  student-float.pt
  tokenizer.vn97tk1
  p5d2-report.json
  SHA256SUMS
```

`student-float.pt` is a training artifact, not a deployment checkpoint.
Float-shadow mode is required when it is loaded for the next distillation
stage.

The launcher compares both candidates and writes:

```text
/kaggle/working/p5d2-selection.json
```

## Gate

A candidate reports `BEHAVIORAL_SIGNAL` when held-out teacher loss improves
by at least 10% and held-out top-1 accuracy also improves.

`WEAK_SIGNAL` means the held-out loss improved but not enough for the stronger
gate.

A 12-task frozen P4 probe is also recorded, but it is not used as proof of
general capability.

## Kaggle

Fresh:

```bash
bash tools/kaggle_p5d2_distill.sh fresh \
  /kaggle/working/p5d1-final \
  /kaggle/working/p4e-k-final/tokenizer.vn97tk1 \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/p3-corpus
```

Resume:

```bash
bash tools/kaggle_p5d2_distill.sh resume \
  /kaggle/working/p5d1-final \
  /kaggle/working/p4e-k-final/tokenizer.vn97tk1 \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/p3-corpus
```
