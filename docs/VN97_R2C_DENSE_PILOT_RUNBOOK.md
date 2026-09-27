# VN97-R2C Dense Pilot Runbook

## Purpose

R2-C proves the 50-150M dense VN97-R2 architecture with real corpus windows
before production-scale training. It must not be used to introduce QAT,
ternary weights, INT4 lowering, or a second model backend.

The reference `r2_cpu_pilot_config` is about 61.7M parameters with the
byte-base tokenizer and about 64.7M with vocab 4096.

## Required inputs

Keep four immutable inputs for the complete acceptance run:

- `tokenizer.vn97tk1`
- `train.jsonl`
- `validation.jsonl`
- `probes.jsonl`

Train and validation records use the canonical chat JSONL format:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

The pilot rejects exact duplicates inside either split and exact records shared
between train and validation.

Held-out probes use:

```json
{"domain":"natural_language","messages":[{"role":"user","content":"..."}],"expected":"..."}
{"domain":"structured_protocol","messages":[{"role":"user","content":"..."}],"expected":"{\"canonical\":\"json\"}"}
{"domain":"tool_action","messages":[{"role":"user","content":"..."}],"expected":"{\"canonical_tool_contract\":\"...\"}"}
{"domain":"tool_action","messages":[{"role":"user","content":"..."}],"expected":"{\"canonical_external_write_contract\":\"...\"}","requires_external_write":true}
```

The expected JSON must be the same canonical tool/action protocol used by the
training corpus and planner. Do not invent a second R2-only tool schema.

## T4 safety rule

Always run static preflight before spending training time. The eager PyTorch
reference scan retains an autograd graph over logarithmic affine-scan rounds,
so recurrent-state size alone is not a valid VRAM estimate.

For a 16 GB T4, start with sequence length 64 and batch size 1. The fail-closed
preflight may reject larger settings, especially sequence length 128.

Use separate preflight/output directories:

```bash
python -m vn97.r2.dense_pilot_cli \
  --tokenizer /kaggle/working/input/tokenizer.vn97tk1 \
  --train-jsonl /kaggle/working/input/train.jsonl \
  --validation-jsonl /kaggle/working/input/validation.jsonl \
  --probe-jsonl /kaggle/working/input/probes.jsonl \
  --work-dir /kaggle/working/r2c-work \
  --output-dir /kaggle/working/r2c-preflight \
  --profile pilot \
  --sequence-length 64 \
  --stride 64 \
  --batch-size 1 \
  --epochs 1 \
  --checkpoint-every 25 \
  --max-windows 512 \
  --device cuda \
  --require-pilot-gate \
  --preflight-only
```

A successful preflight writes
`r2-dense-pilot-preflight.json` and explicitly records
`model_forward_executed=false` and `training_executed=false`.

Do not use `--allow-low-memory` merely to force a T4 run through a failed
preflight. Reduce sequence length or batch size first.

## Time-bounded training

For a session with roughly one hour of T4 quota remaining, reserve a safety
margin for held-out validation, checkpoint serialization and notebook/session
shutdown. A conservative first run is 35-40 minutes of training budget:

```bash
python -m vn97.r2.dense_pilot_cli \
  --tokenizer /kaggle/working/input/tokenizer.vn97tk1 \
  --train-jsonl /kaggle/working/input/train.jsonl \
  --validation-jsonl /kaggle/working/input/validation.jsonl \
  --probe-jsonl /kaggle/working/input/probes.jsonl \
  --work-dir /kaggle/working/r2c-work \
  --output-dir /kaggle/working/r2c-output \
  --profile pilot \
  --sequence-length 64 \
  --stride 64 \
  --batch-size 1 \
  --epochs 1 \
  --learning-rate 2e-4 \
  --checkpoint-every 25 \
  --max-windows 512 \
  --device cuda \
  --max-run-seconds 2400 \
  --require-pilot-gate
```

The timer is checked at batch boundaries. A single in-flight batch is never
aborted halfway through.

If the training budget expires before the configured training completes, the
run exits cleanly with:

- `status=PAUSED`
- `r2c-work/dense-resume.pt`
- `r2c-work/r2-dense-run.json`
- `r2c-output/model.r2.pt`
- `r2c-output/r2-dense-pilot-progress.json`
- SHA-256 identity for the resume checkpoint.

Final generation/tool probes are intentionally skipped on a paused run so they
do not consume the safety margin.

## Resume

Preserve both `r2c-work` and `r2c-output`. Resume with the exact same model
recipe, tokenizer, train/validation data and paths. The wall-clock budget may
change because it is orchestration, not part of the training identity.

```bash
python -m vn97.r2.dense_pilot_cli \
  --tokenizer /kaggle/working/input/tokenizer.vn97tk1 \
  --train-jsonl /kaggle/working/input/train.jsonl \
  --validation-jsonl /kaggle/working/input/validation.jsonl \
  --probe-jsonl /kaggle/working/input/probes.jsonl \
  --work-dir /kaggle/working/r2c-work \
  --output-dir /kaggle/working/r2c-output \
  --profile pilot \
  --sequence-length 64 \
  --stride 64 \
  --batch-size 1 \
  --epochs 1 \
  --learning-rate 2e-4 \
  --checkpoint-every 25 \
  --max-windows 512 \
  --device cuda \
  --max-run-seconds 2400 \
  --require-pilot-gate \
  --resume
```

Resume fails closed if any of these disagree with the persisted run:

- architecture/config fingerprint
- tokenizer SHA-256
- dataset identity
- training recipe identity
- best checkpoint SHA-256
- optimizer/model resume identity.

The original pre-training held-out baseline is persisted and reused across
resumes. It is never recomputed from a new random model.

## Completion evidence

A complete run writes `r2-dense-pilot-report.json` using schema
`VN97R2DENSEPILOT2`. Acceptance requires:

- the actual model remains in the 50-150M pilot class;
- held-out loss improves over the original pre-training baseline;
- generation probe coverage includes natural language;
- tool/action probes are valid canonical JSON and meet exact protocol targets;
- at least one external-write probe proves R2 routes the request through the
  deep/full-depth cognition path;
- all configured pilot thresholds pass;
- `quantization_used=false`.

R2-C acceptance is evidence that the dense architecture/training path can learn
and generate correctly at pilot scale. It is not permission to start QAT.
Production-scale instruction/reasoning/tool training and fresh multi-axis
validation remain ahead of mobile lowering.
