# P5D3B — Relational Representation/Dynamics Distillation

P5D3A extracted compact Falcon3-Mamba relational targets for all 600 sealed
P5D1 records. P5D3B trains the selected P5D2 309M VN97 student against those
targets while preserving behavioral imitation and P3 language retention.

## Starting point

P5D3B requires the selected P5D2 float student. It does not initialize a new
309M model.

The P5D2 student remains in FP32 float-shadow mode. Ternary QAT is still
deferred.

## Cross-architecture depth mapping

Teacher relational targets:

```text
Falcon3-Mamba hidden depths: 16, 32, 48, 64
```

Student targets:

```text
VN97 hidden depths: 8, 16, 24, 32
```

The mapped losses are dimension-independent because each depth is represented
by a four-segment cosine Gram matrix and relative segment norm profile.

## Loss

Each optimizer step contains two independent backward passes before one update.

Relational branch:

```text
1.0 * Gram loss
+ 0.5 * depth-to-depth Gram-dynamics loss
+ 0.1 * relative segment norm loss
```

Behavior anchor:

```text
0.25 * teacher-response cross entropy
```

The behavior anchor reduces the risk that representation matching destroys the
behavioral signal already obtained in P5D2.

## Dual T4 pilot

Both candidates start from the exact same selected P5D2 student.

Candidate 0:

```text
LR = 2e-5
```

Candidate 1:

```text
LR = 1e-5
```

Both run for 500 optimizer steps, covering the full 500-record P5D3A training
split once in shuffled order.

Each Kaggle T4 runs one candidate concurrently.

## Evaluation gate

Before and after training, each candidate measures:

- 100-record held-out relational loss;
- P5D1 held-out teacher-response CE/top-1;
- P3 validation CE/top-1;
- 12-task frozen P4 probe.

`RELATIONAL_SIGNAL` requires:

- at least 8% held-out relational-loss improvement;
- teacher held-out CE no more than 5% worse;
- P3 validation CE no more than 10% worse.

`WEAK_RELATIONAL_SIGNAL` allows smaller relational improvement and slightly
larger retention tolerances.

The frozen P4 probe remains diagnostic evidence only.

## Resume

Resume checkpoints are model-only FP16 CPU snapshots. Live training remains
FP32. AdamW moments restart after resume. Candidate checkpoint writes are
staggered by 25 steps so both processes do not write ~600 MB snapshots
simultaneously.

## Output

Each candidate creates:

```text
p5d3b-cN-final/
  student-float.pt
  tokenizer.vn97tk1
  p5d3b-report.json
  SHA256SUMS
```

The launcher writes:

```text
/kaggle/working/p5d3b-selection.json
```

The selected artifact remains a training checkpoint, not a deployment package.

## Kaggle

Fresh:

```bash
bash tools/kaggle_p5d3b_distill.sh fresh \
  /kaggle/working/p5d1-final \
  /kaggle/working/p5d2-c0-final \
  /kaggle/working/p5d3a-final \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/p3-corpus
```

Resume:

```bash
bash tools/kaggle_p5d3b_distill.sh resume \
  /kaggle/working/p5d1-final \
  /kaggle/working/p5d2-c0-final \
  /kaggle/working/p5d3a-final \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/p3-corpus
```
