# P5A — Scaled VN97 Foundation Pilot

P4E-I through P4E-N isolated a capacity bottleneck in the existing small VN97
language core. Decoder changes, arithmetic-coverage expansion, low-LR replay and
sequence-ranking repair did not produce a durable reasoning gain.

P5A scales the **same VN97 architecture** instead of introducing another model:

- Selective SSM remains the only language core;
- ternary training/deployment remains canonical;
- exact-ZOH recurrence remains unchanged;
- the existing VN97 tokenizer remains canonical;
- no Transformer/LLaMA/backend is introduced.

## Frozen scale ladder

Candidate 0:
- d_model 576
- 18 layers
- d_state 32
- embedding rank 288
- about 26.3M parameters with the current tokenizer

Candidate 1:
- d_model 672
- 20 layers
- d_state 32
- embedding rank 336
- about 39.0M parameters

Candidate 2:
- d_model 768
- 20 layers
- d_state 32
- embedding rank 384
- about 50.6M parameters

The candidates are independent from-scratch foundation pilots. A larger
candidate is not run merely because it exists; the smaller stage is measured
first so GPU quota is not spent blindly.

## Training mixture

Each candidate uses 6000 deterministic windows:

- 3000 P3 language windows;
- 3000 category-balanced P4 instruction/cognition windows.

P4 reasoning expressions appearing in the frozen P4 validation set or optional
dev suite are excluded from training. Exact held-out prompts are also excluded.

## T4 execution

The pilot is specifically designed for 16 GB Tesla T4-class GPUs:

- FP16 autocast;
- GradScaler;
- logical batch 8;
- physical microbatch 1;
- gradient accumulation;
- low-memory sequential-reference SSM training path;
- expandable CUDA allocator segments;
- resumable model/optimizer/scaler state every 50 steps.

Production inference remains on VN97's normal parallel scan.

## Evaluation

After training, each candidate is measured on:

- P3 held-out language validation;
- the canonical 180-task P4 probe;
- the 60-task numeric-copy suite;
- optional held-out dev suite.

The pilot never automatically replaces P4E-K. It reports
`PROMISING_SCALE_SIGNAL` only when the candidate reaches all of:

- reasoning >= 5/30;
- canonical P4 >= 60/180;
- numeric copy >= 40/60;
- P3 token top-1 >= 0.08.

This is only a scale signal. Promotion to production requires later P5
validation and release gates.

## Kaggle

Start with candidate 0:

```bash
bash tools/kaggle_p5a_scaled_foundation.sh fresh 0 \
  /kaggle/working/p4e-k-final \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/p3-corpus \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/held-out-p4.jsonl
```

Resume:

```bash
bash tools/kaggle_p5a_scaled_foundation.sh resume 0 \
  /kaggle/working/p4e-k-final \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/p3-corpus \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/held-out-p4.jsonl
```

Do not start candidates 1 or 2 until candidate 0 has been measured.
