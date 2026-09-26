# P5C — Short Alignment / Calibration

P5B successfully produced a native VN97 checkpoint by transplanting a pinned
Mamba-130M selective-SSM checkpoint into VN97. The transplant is structurally
valid and finite, but it is intentionally not lossless.

P5C is the minimum alignment stage required before judging the transplanted
checkpoint against held-out VN97 capability tests.

## What P5C repairs

The bridge must compensate for:

- Mamba inner width 1536 compressed to VN97 width 768;
- omitted source causal conv1d;
- omitted source D skip;
- source tokenizer -> native VN97 tokenizer lexical projection;
- immediate snapping of source projections into VN97 ternary weights.

P5C does not train a new foundation from scratch.

## Frozen training profile

- source: P5B `TRANSPLANT_READY_FOR_ALIGNMENT` checkpoint;
- sequence length: 256;
- 2400 deterministic windows total;
- 1200 P3 language records;
- 1200 category-balanced P4 instruction/cognition records;
- logical batch 8;
- physical microbatch 1;
- 300 optimizer steps;
- FP16 autocast + GradScaler;
- exact low-memory sequential VN97 SSM execution;
- embedding/tokenizer bridge LR: 5e-5;
- recurrent/core LR: 1e-5;
- checkpoint every 50 steps.

The embedding group learns faster than the transplanted recurrent core so the
new tokenizer interface can align without rapidly erasing imported SSM
dynamics.

Frozen P4 validation prompts/expressions and optional dev prompts/expressions
remain excluded from the alignment set.

## Evaluation

P5C records a small pre-alignment probe, then after alignment evaluates:

- full canonical P4 180-task probe;
- reasoning category 30 tasks;
- numeric-copy 60-task suite;
- held-out P3 language validation;
- optional held-out dev suite.

P5C reports `ELIGIBLE_FOR_P5D` only if all of these are reached:

- canonical >= 60/180;
- reasoning >= 5/30;
- numeric copy >= 40/60;
- P3 token top-1 >= 0.08.

Anything below that does not replace P4E-K automatically.

## Kaggle

Fresh:

```bash
bash tools/kaggle_p5c_alignment.sh fresh \
  /kaggle/working/p5b-mamba130m-final \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/p3-corpus \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/held-out-p4.jsonl
```

Resume after a saved checkpoint:

```bash
bash tools/kaggle_p5c_alignment.sh resume \
  /kaggle/working/p5b-mamba130m-final \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/p3-corpus \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/held-out-p4.jsonl
```

P5A stays stopped while P5C is tested. Its existing work/checkpoint should be
kept as a fallback until P5C held-out results are known.


## P5C v2 stability revision

The first real P5C run exposed a numeric issue before the first optimizer step:

- baseline probe: 0/30;
- numeric copy: 0/60;
- P3 loss: about 46.66;
- P3 top-1: about 0.0016;
- FP16 backward produced a non-finite gradient norm.

The transplanted checkpoint itself remained finite, so this is treated as an
alignment-numerics failure rather than a corrupt P5B artifact.

P5C v2 therefore changes only the alignment execution profile:

- sequence length: 256 -> 128;
- FP16 training -> FP32 training;
- first 50 optimizer steps train only the lexical/tokenizer bridge;
- recurrent/core parameters are frozen for those 50 steps;
- the core is then unfrozen at LR 5e-6;
- lexical LR remains 5e-5;
- logical batch remains 8 with physical microbatch 1;
- total optimizer steps remain 300.

This revision intentionally trades speed for stability. The failed v1 run did
not create a valid resume checkpoint, so v2 must start with `fresh`.
