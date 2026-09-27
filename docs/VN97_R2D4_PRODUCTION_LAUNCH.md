# VN97-R2D4 Production Launch Runbook

## Purpose

R2-D4 is the gate between the validated R2-D3 trainer and actual production
dense training. It packages the exact corpus/model recipe, performs a measured
CUDA forward/backward memory probe using the real 1B-class model, and writes a
no-promotion evidence bundle.

A training run is rejected unless it receives a passing bundle whose corpus
manifest and recipe fingerprints exactly match the current invocation.

## Inputs

Required:

- canonical `tokenizer.vn97tk1`;
- disjoint `train.jsonl`;
- disjoint `validation.jsonl`;
- at least one `--task-family`;
- a CUDA device for measured preflight.

For stages after `dense_pretrain`, also provide the exact parent R2 checkpoint.

R2-D4 rejects exact duplicate records inside either corpus split and exact
records shared across train/validation before allocating the production model.

## T4 starting point

For a 16 GB T4, start conservatively:

```bash
vn97-r2-production \
  --mode preflight \
  --stage dense_pretrain \
  --tokenizer /kaggle/working/input/tokenizer.vn97tk1 \
  --train-jsonl /kaggle/working/input/train.jsonl \
  --validation-jsonl /kaggle/working/input/validation.jsonl \
  --task-family language \
  --task-family reasoning \
  --work-dir /kaggle/working/r2d-work \
  --output-dir /kaggle/working/r2d-output \
  --sequence-length 128 \
  --micro-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --precision fp16 \
  --device cuda \
  --safety-fraction 0.90
```

This initializes the exact `r2_mobile_1b_config`, runs one real
forward/backward micro-batch through the memory-efficient scan and activation
checkpointing path, measures CUDA peak allocated/reserved bytes, then returns
the model to CPU.

The preflight writes:

```
r2d-output/r2d4-preflight.json
```

The bundle is explicitly marked `promotion_allowed=false`. Passing preflight
means only that the measured micro-batch fits the configured safety budget. It
is not a model-quality result.

## Production training

Only after preflight passes:

```bash
vn97-r2-production \
  --mode train \
  --stage dense_pretrain \
  --tokenizer /kaggle/working/input/tokenizer.vn97tk1 \
  --train-jsonl /kaggle/working/input/train.jsonl \
  --validation-jsonl /kaggle/working/input/validation.jsonl \
  --task-family language \
  --task-family reasoning \
  --work-dir /kaggle/working/r2d-work \
  --output-dir /kaggle/working/r2d-output \
  --sequence-length 128 \
  --micro-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --precision fp16 \
  --epochs 1 \
  --learning-rate 1e-4 \
  --checkpoint-every 25 \
  --device cuda \
  --preflight-bundle /kaggle/working/r2d-output/r2d4-preflight.json \
  --max-run-seconds 2400
```

The trainer uses:

- full-parameter dense VN97-R2;
- FP32 canonical weights and gradients;
- CUDA FP16/BF16 autocast only for compute;
- CPU-offloaded AdamW moments;
- gradient accumulation;
- dynamic FP16 loss scaling;
- activation checkpointing;
- memory-efficient selective scan;
- resumable state at accumulation boundaries.

Changing the tokenizer, corpus files, parent checkpoint, sequence length,
micro-batch size, precision or other recipe fields invalidates the previous
preflight bundle.

## Quota safety

Use `--max-run-seconds` on quota-limited GPU sessions. The production trainer
pauses only at a complete gradient-accumulation boundary and writes
`production-resume.pt`.

Preserve the complete work and output directories between sessions. Resume
identity is bound to architecture, corpus, recipe and trainer configuration.

## Dense-first rule

R2-D4 does not enable QAT, INT4 or ternary training. Those remain blocked until
production dense intelligence, instruction/reasoning, tool/action, capability
training and fresh multi-axis validation all pass.
