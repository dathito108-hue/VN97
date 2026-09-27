# VN97-R2D7 Deterministic Streaming-Shard Training

## Purpose

R2-D7 is the production dense-pretraining execution bridge between the
R2-D6 sharded corpus and the existing R2-D3 full-parameter optimizer/runtime.

It does **not** introduce a second model or optimizer backend. It reuses:

- the same VN97-R2 model and weights;
- memory-efficient selective scan;
- block activation checkpointing;
- FP32 canonical parameters and gradients;
- CUDA FP16/BF16 autocast where configured;
- CPU-offloaded AdamW moments;
- dynamic FP16 loss scaling;
- measured CUDA preflight evidence.

The change is the data path: training and validation windows are generated
from immutable R2-D6 shards on demand instead of materializing the entire
corpus/window set in RAM.

## Production gates

The CLI refuses to allocate the 1B production model unless all of these hold:

- the R2-D6 package verifies;
- release is still held out;
- sequence length matches the D6 index;
- the D6 scale policy is exactly the canonical 8/20 supervised
  target-token-per-parameter policy;
- the D6 scale floor has passed;
- tokenizer/model architecture fingerprints match;
- an explicit CUDA device is requested;
- a valid R2-D5 measured preflight receipt matches the architecture and recipe.

CPU execution of the core trainer remains available for validation/tests, but
the production CLI does not silently fall back to CPU for a 1B run.

## Streaming order

R2-D7 never shuffles individual records into one giant in-memory permutation.

For each epoch:

1. training shards are deterministically shuffled from `trainer.seed + epoch`;
2. records inside each shard retain their sealed JSONL order;
3. windows inside each record retain canonical VN97 completion-window order;
4. validation shards are read sequentially after a completed epoch;
5. release shards are never included in the production training manifest and
   are never consumed by the training or validation iterators.

The epoch shard-order digest is stored in resume evidence.

## Resume cursor

The exact next training position is represented by:

```
epoch
shard_position
record_index
window_index
```

The resume checkpoint additionally binds:

- R2-D6 index ID;
- architecture fingerprint;
- derived dense-pretrain manifest identity;
- recipe fingerprint;
- trainer configuration;
- epoch shard-order digest;
- model state;
- CPU AdamW state;
- loss-scaling state;
- optimizer/micro-step counters;
- target-token/window counters;
- best checkpoint identity.

Resume checkpoints are only written at complete gradient-accumulation /
optimizer boundaries. `accumulation_open=false` is mandatory.

R2-D7 deliberately does not persist a cursor that has advanced to the next
epoch until validation for the completed epoch has finished. A crash at that
boundary therefore replays some already-trained windows rather than silently
skipping validation/promotion.

## T4 preflight receipt

R2-D7 accepts the no-promotion receipt sealed by R2-D5. It verifies:

- receipt identity;
- `training_allowed=false`;
- `passed=true`;
- architecture fingerprint;
- recipe fingerprint;
- sequence length;
- micro-batch size;
- non-empty device name;
- reason `within_measured_safety_budget`;
- internally consistent free/total/peak memory values;
- peak reserved memory within the recorded safety budget.

At runtime the CUDA device name must match the device used by the measured
preflight. A different GPU requires a new measured preflight.

## Launch

After R2-D5 measured preflight passes and R2-D6 reaches the canonical scale
floor:

```bash
vn97-r2-stream-train \
  --corpus-package /data/r2d6-corpus \
  --work-dir /kaggle/working/r2d7-work \
  --output-dir /kaggle/working/r2d7-output \
  --preflight-receipt /data/r2d5-preflight-receipt.json \
  --sequence-length 128 \
  --micro-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --precision fp16 \
  --epochs 1 \
  --learning-rate 1e-4 \
  --checkpoint-every 25 \
  --max-run-seconds 2400 \
  --device cuda
```

On a quota-limited session, `--max-run-seconds` causes the run to pause only
after a complete optimizer boundary. Preserve:

```
r2d7-work/streaming-production-resume.pt
r2d7-output/model.r2.pt
```

Restart with the exact same corpus package, recipe, trainer settings and
paths. Any identity mismatch fails closed.

## Validation and best checkpoint

Validation is itself streamed shard by shard. It does not materialize the full
validation window set.

A best checkpoint is promoted only after a fully completed epoch and records:

- D6 index ID;
- streaming run identity;
- production manifest identity;
- recipe fingerprint;
- trainer configuration;
- optimizer/micro-step/window counts;
- validation metrics.

A pause in the middle of the first epoch may therefore legitimately have no
best checkpoint yet.

## Determinism evidence

R2-D7 tests include a CPU smoke-model experiment:

```
partial run -> persist resume -> fresh process/model -> resume -> completion
```

The final model tensors are compared with an uninterrupted run using exact
`rtol=0, atol=0` parity.

## Remaining work

R2-D7 makes production-scale data execution resumable, but it does not create
the multi-billion-token corpus itself and it does not spend GPU quota.

The next steps are:

1. run the sealed R2-D5 measured preflight on the intended T4/GPU;
2. expand and seal R2-D6 corpora until the canonical scale floor is reached;
3. launch quota-bounded R2-D7 dense pretraining;
4. continue instruction/reasoning -> tool/action -> capability curriculum;
5. perform fast-path alignment and fresh multi-axis validation;
6. only then enter R2-E QAT/mobile lowering.
