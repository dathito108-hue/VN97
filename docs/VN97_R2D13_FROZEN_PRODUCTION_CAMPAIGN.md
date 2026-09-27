# VN97-R2D13 Frozen Production Campaign Package

## Purpose

R2-D13 turns one verified R2-D12 frozen virtual corpus, one exact R2-D8
dense-pretrain curriculum, and one R2-D5 measured-preflight package into a
portable, fail-closed production campaign.

The execution chain becomes:

```
D10/D11/D9/D6
-> D12 frozen virtual corpus
-> D8 dense-pretrain projection/curriculum
-> D13 frozen production campaign
-> measured T4 preflight
-> D13 ready receipt
-> quota-bounded D7 dense training
```

R2-D13 does not train the model and does not claim that a preflight has
occurred.

## Current execution scope

The campaign format binds a stage explicitly, but the executable D13 launcher
currently accepts only:

```
dense_pretrain
```

This matches the current R2-D7 production trainer.

Instruction/reasoning, tool/action and capability stages remain later runtime
work. D13 fails closed instead of pretending those stage launchers already
exist.

## Inputs

A campaign build requires:

- a verified R2-D12 view;
- its runtime workspace containing the mounted D6 batch packages;
- an R2-D8 curriculum compiled against the matching D12 stage projection;
- an R2-D5 preflight package;
- an exact 40-character repository commit;
- quota/trainer settings.

The D5 package must match the D12/D8 campaign on:

- repository commit;
- architecture fingerprint;
- tokenizer SHA-256;
- sequence length;
- production recipe;
- task-family set.

## Why D5 is reused

R2-D13 does not introduce another memory-preflight implementation.

R2-D5 already provides:

- exact production recipe identity;
- CUDA forward/backward memory measurement;
- safety fraction;
- measured peak allocated/reserved bytes;
- device identity;
- a signed-by-hash immutable receipt format.

D13 uses that existing gate and binds its campaign ID into the frozen training
campaign.

## Build

Example:

```bash
vn97-r2-production-campaign build \
  --view-dir /data/r2d12-view \
  --workspace-root /data \
  --curriculum-plan /data/r2d8-dense-plan.json \
  --preflight-package /data/r2d5-preflight-package \
  --repository-commit <40-char-commit> \
  --output-dir /data/r2d13-campaign \
  --learning-rate 0.0001 \
  --weight-decay 0.01 \
  --max-grad-norm 1.0 \
  --checkpoint-every 25 \
  --max-run-seconds 3000
```

The build is CPU-only.

No 1B model is allocated.

## Package contents

```
r2d13-campaign/
  r2d13-campaign.json
  curriculum.json
  run_t4_preflight.sh
  seal_t4_preflight.sh
  run_t4_train.sh
  view/
    r2d12-view.json
    r2d12-mounts.json
    tokenizer.vn97tk1
    r2d11-registry.json
    r2d11-ledger/
      ...
  preflight/
    r2d5-campaign.json
    corpus.vn97corpus1.json
    training.jsonl
    validation.jsonl
    release.jsonl
    tokenizer.vn97tk1
    run_t4_preflight.sh
```

The large production D6 shards are still not copied.

They remain in the D12 runtime workspace and are resolved through the D12
mount file.

## Campaign identity

`VN97R2D13CAMPAIGN1` binds:

- repository commit;
- stage;
- D12 `view_id`;
- D11 registry generation;
- D12 projected corpus `index_id`;
- D8 `plan_id`;
- D5 preflight campaign ID;
- architecture fingerprint;
- tokenizer SHA-256;
- production recipe + recipe fingerprint;
- complete trainer configuration;
- quota limit `max_run_seconds`;
- exact projected scale evidence;
- shell launcher SHA-256 values;
- release-holdout state.

The base campaign always contains:

```
training_launch_allowed = false
```

Training is unlocked only by a separate D13 ready receipt after a successful
measured preflight.

## Stage-scale gate before GPU

D13 intentionally permits a campaign to be built below the data-scale floor.

This allows:

- provenance verification;
- package transport testing;
- mount testing;
- reporting how much data remains missing.

However:

```bash
vn97-r2-production-campaign gate-preflight ...
```

fails unless the selected D12 stage projection has:

```
scale_floor_passed = true
```

Therefore the generated `run_t4_preflight.sh` does not spend GPU quota on a
corpus that the project already knows is below the canonical data-scale floor.

The current production floor remains:

```
8 supervised target tokens / model parameter
```

and the planning target remains:

```
20 supervised target tokens / model parameter
```

These remain project scale gates, not quality guarantees.

## Verify package before T4

```bash
vn97-r2-production-campaign verify \
  --package-dir /data/r2d13-campaign \
  --workspace-root /data
```

Verification reconstructs and checks:

- D12 view + mounted D6 identities;
- stage projection;
- D8 plan identity;
- D5 preflight package identity;
- repository commit binding;
- tokenizer/architecture/recipe binding;
- trainer/epoch/seed binding;
- exact scale evidence;
- all launcher hashes.

## T4 preflight

After the stage projection reaches the floor:

```bash
VN97_REPO=/kaggle/working/VN97 \
VN97_WORKSPACE_ROOT=/kaggle/working \
/path/to/r2d13-campaign/run_t4_preflight.sh
```

The wrapper:

1. checks the exact Git commit;
2. installs the checked-out VN97 package;
3. runs the D13 scale/provenance gate;
4. executes the canonical nested R2-D5 measured CUDA preflight.

The nested D5 package remains preflight-only.

No model training is started.

## Seal measured preflight

After the D5 preflight emits its D4 bundle:

```bash
R2D5_OUTPUT_DIR=/kaggle/working/r2d5-output \
VN97_WORKSPACE_ROOT=/kaggle/working \
/path/to/r2d13-campaign/seal_t4_preflight.sh
```

The wrapper creates:

```
preflight-receipt.json
r2d13-ready.json
```

The D13 ready receipt has schema:

```
VN97R2D13READY1
```

and binds:

- D13 campaign ID;
- exact D5 preflight receipt ID;
- exact D5 preflight campaign ID;
- repository commit;
- D12 view ID;
- projected corpus index ID;
- D8 plan ID;
- production recipe fingerprint;
- measured GPU device name;
- measured peak reserved bytes.

Only this ready receipt contains:

```
training_allowed = true
```

That flag means the frozen launch contract passed all pre-training gates. It
does not mean training has already happened or succeeded.

## Verify ready receipt

```bash
vn97-r2-production-campaign verify-ready \
  --package-dir /data/r2d13-campaign \
  --workspace-root /data \
  --ready-receipt /data/r2d13-campaign/r2d13-ready.json \
  --preflight-receipt /data/r2d13-campaign/preflight-receipt.json
```

A receipt from another D5 campaign is rejected even if its architecture and
recipe happen to match.

This prevents measured evidence from being casually replayed across frozen
campaigns.

## Quota-bounded dense training

After `verify-ready` passes:

```bash
VN97_REPO=/kaggle/working/VN97 \
VN97_WORKSPACE_ROOT=/kaggle/working \
R2D13_WORK_DIR=/kaggle/working/r2d13-work \
R2D13_OUTPUT_DIR=/kaggle/working/r2d13-output \
/path/to/r2d13-campaign/run_t4_train.sh
```

The generated training wrapper:

1. verifies the exact repository commit;
2. verifies the D13 ready receipt and D5 measured receipt;
3. launches `vn97-r2-stream-train` against the D12 virtual view;
4. uses the exact D8 curriculum;
5. uses the exact recipe/trainer settings frozen by D13;
6. applies the frozen `--max-run-seconds` quota bound;
7. preserves D7 optimizer-boundary resume behavior.

## Portability

D12 mount paths are runtime locators, not corpus identity.

After moving the campaign and D6 batch packages to another machine, only the
copied:

```
view/r2d12-mounts.json
```

needs new relative paths under the destination workspace root.

The mounted packages must still reproduce the exact frozen D6 index IDs.

Changing mount locations does not change:

- D12 `view_id`;
- D13 campaign ID;
- D8 curriculum ID;
- D7 resume identity.

## What D13 proves

A verified D13 base campaign proves that a single immutable training launch
contract has been assembled.

A verified D13 ready receipt additionally proves that the exact
architecture/recipe passed the measured CUDA memory gate on the recorded
device.

Neither proves:

- dense training completion;
- model quality;
- capability quality;
- mobile inference quality.

Those require later training and fresh held-out validation.

## No quantization yet

D13 preserves the existing production contract:

- full-parameter dense training;
- activation checkpointing;
- memory-efficient selective scan;
- CPU optimizer-state offload;
- no QAT;
- no INT4;
- no ternary production lowering.

Mobile lowering remains blocked until dense training and fresh validation
succeed.
