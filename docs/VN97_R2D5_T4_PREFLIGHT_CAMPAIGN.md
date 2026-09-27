# VN97-R2D5 Measured T4 Preflight Campaign

## Status

R2-D5 packages the already-sealed VN97 production corpus and canonical
VN97TK1 tokenizer into a reproducible **memory-preflight-only** artifact for
the 1B-class R2 model.

R2-D5 does **not** authorize production training. Every package and every
sealed preflight receipt carries:

```
training_allowed=false
```

The purpose is to answer one question with measured evidence:

> Does the exact VN97-R2 1B architecture, sequence length, micro-batch and
> precision fit the intended CUDA device with the configured safety margin?

Only after that question is answered can a later milestone decide how to
scale the production corpus and begin dense pretraining.

## Why the existing corpus is useful here

The existing `VN97CORPUS1` production-intelligence corpus is already
license-reviewed, deterministic and split into:

- `training.jsonl`
- `validation.jsonl`
- `release.jsonl`

R2-D5 re-verifies all three split identities, source-license approval flags,
record counts, hashes and exact cross-split isolation.

This corpus is suitable for a real memory preflight because activation memory
depends on the exact architecture/sequence/micro-batch execution path, not on
having billions of training tokens.

It must not be interpreted as proof that the current corpus volume is enough
to train a 1B model to production intelligence.

## Package contents

A successful build produces:

```
r2d5-package/
  corpus.vn97corpus1.json
  training.jsonl
  validation.jsonl
  release.jsonl
  tokenizer.vn97tk1
  r2d5-campaign.json
  run_t4_preflight.sh
```

The campaign identity binds:

- exact repository commit;
- VN97CORPUS1 manifest identity and SHA-256;
- all three split identities;
- tokenizer SHA-256;
- 1B model architecture fingerprint;
- model parameter count;
- R2-D4 production manifest identity;
- sequence length;
- micro-batch;
- gradient accumulation;
- precision;
- production recipe fingerprint;
- safety fraction;
- task families;
- generated preflight script SHA-256.

The release split is packaged only to preserve the complete immutable corpus
identity. It is never passed to preflight or training.

## Build

Use the commit that will also be checked out on Kaggle:

```bash
vn97-r2-preflight-package build \
  --corpus-dir /path/to/VN97CORPUS1 \
  --tokenizer /path/to/tokenizer.vn97tk1 \
  --repository-commit <40-char-main-commit> \
  --output-dir /path/to/r2d5-package \
  --task-family language \
  --task-family reasoning \
  --sequence-length 128 \
  --micro-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --precision fp16 \
  --safety-fraction 0.90
```

Verify before upload:

```bash
vn97-r2-preflight-package verify \
  --package-dir /path/to/r2d5-package
```

Any change to corpus, tokenizer, campaign JSON or generated shell script makes
verification fail.

## Kaggle / T4 execution

Upload or otherwise place the whole R2-D5 package on the Kaggle session.

Check out the exact repository commit recorded in the package, then run:

```bash
export VN97_REPO=/kaggle/working/VN97
export R2D5_WORK_DIR=/kaggle/working/r2d5-work
export R2D5_OUTPUT_DIR=/kaggle/working/r2d5-output

/path/to/r2d5-package/run_t4_preflight.sh
```

The generated script refuses to continue if `git rev-parse HEAD` does not
equal the campaign's locked commit.

The underlying R2-D4 preflight performs one real forward/backward micro-batch
using:

- the exact `r2_mobile_1b_config`;
- FP32 canonical weights/gradients;
- CUDA FP16/BF16 autocast as configured;
- memory-efficient selective scan;
- block activation checkpointing;
- the exact sequence length and micro-batch.

No optimizer moments are allocated and no checkpoint can be promoted.

## Seal the measured result

If R2-D4 writes a passing:

```
r2d4-preflight.json
```

seal it against the package:

```bash
vn97-r2-preflight-package seal-preflight \
  --package-dir /path/to/r2d5-package \
  --preflight-bundle /kaggle/working/r2d5-output/r2d4-preflight.json \
  --output /kaggle/working/r2d5-preflight-receipt.json
```

The receipt is accepted only when:

- D4 schema is correct;
- promotion remains disabled;
- production manifest identity matches;
- recipe fingerprint matches;
- architecture fingerprint matches;
- measured preflight passed.

The receipt records the GPU name, free VRAM before the probe, peak allocated
VRAM and peak reserved VRAM.

## Decision after T4 evidence

R2-D5 itself never starts production training.

A later gate must use the measured receipt to choose one of these outcomes:

1. **T4 fits with useful margin** — continue to production-corpus scale and
   quota-bounded dense pretraining.
2. **T4 only narrowly fits** — reduce sequence length or revise execution
   memory strategy, then perform a new measured preflight with a new recipe
   fingerprint.
3. **T4 OOM / fails safety budget** — do not force the run with quantization;
   use a larger GPU or further exact-memory optimization first.

QAT, INT4 and ternary remain forbidden until dense production intelligence and
fresh multi-axis validation pass.
