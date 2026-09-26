# P5D — Large-SSM Teacher → VN97-RT Distillation

P5B/P5C proved that direct tensor transplant is structurally possible but does
not preserve useful intelligence. P5D therefore changes the transfer mechanism,
not the VN97 runtime architecture.

## Locked principle

The final mobile runtime remains a single native VN97 model:

- no teacher model in the APK;
- no Transformer/Mamba backend at runtime;
- no parallel model stack;
- no dependency on a remote AI service.

Large models are temporary teachers used only to produce training targets.

## Teacher

The first frozen teacher is:

`tiiuae/Falcon3-Mamba-7B-Instruct`

It is selected because it is a Mamba1 causal decoder with:

- 64 blocks;
- hidden width 4096;
- SSM state width 16;
- instruction tuning;
- strong language/reasoning/code behavior relative to smaller SSM sources.

The teacher license is the TII Falcon-LLM License 2.0. P5D tooling must never
silently download the teacher; source acquisition is a separate explicit step.

## VN97-RT target

The first realtime target is intentionally much smaller than the teacher:

- d_model: 1536;
- layers: 32;
- d_state: 16;
- embedding rank: 768;
- current VN97 byte-native/hybrid tokenizer;
- recurrent state remains constant with context length.

With the current ~4.1k-token VN97 tokenizer this target is approximately 309M
parameters. A pure 2-bit lower-bound is roughly 77 MB before scales, embedding
storage and metadata. Actual VN97MI1 size must be measured later rather than
inferred from this lower bound.

## Why P5D does not ternarize first

P5B immediately compressed mapped weights into VN97 ternary projections and
then tried to repair the damage. P5C showed that short alignment was not enough.

P5D reverses the order:

1. initialize a native VN97-RT student;
2. enable dense float-shadow execution on every TernaryLinear;
3. distill teacher behavior/representations into VN97;
4. prove held-out capability gain;
5. only then start gradual ternary QAT;
6. pack to VN97T2;
7. benchmark real mobile latency and memory.

The float-shadow switch is runtime-only. It is not serialized as model state,
so normal VN97 inference remains ternary by default.

## Phase sequence

### P5D0 — Distillation Foundation

- freeze teacher identity and license;
- freeze VN97-RT target shape;
- add float-shadow training mode to TernaryLinear;
- add deterministic profile/footprint audit;
- define a sealed teacher-output corpus contract.

### P5D1 — Teacher Corpus Generation

Generate teacher answers offline. The corpus stores only training targets needed
for distillation, not a teacher runtime dependency.

The first corpus should cover:

- instruction following;
- reasoning/math final answers;
- structured JSON;
- tool intent and approval behavior;
- general language;
- code;
- memory-use patterns.

Held-out P4/dev prompts must remain excluded.

### P5D2 — Float-Shadow Behavioral Distillation

Train VN97-RT in float-shadow mode first. Do not run ternary STE during the
initial intelligence-transfer phase.

Primary early success signal:

- substantial P3 held-out improvement;
- non-zero P4 canonical capability;
- reasoning > the failed P5B/P5C baseline;
- no regression in authority/tool behavior.

### P5D3 — Representation/Dynamics Distillation

If behavioral distillation is positive, add compact teacher representation
targets. Cross-tokenizer hidden-state matching must use explicit projection/
pooling rather than pretending teacher and VN97 token positions are identical.

### P5D4 — Byte-Patch Frontend

Only after the language core shows capability gain, introduce MegaByte-style
byte patching as a VN97 frontend optimization. This is not a second model.

### P5D5 — Gradual Ternary QAT

Apply BitNet-inspired gradual ternary training after the float student is
competent:

- dense shadow;
- increasing ternary exposure;
- activation/scaling calibration;
- final VN97T2 packing.

### P5D6 — Dynamic Memory Experiments

RWKV-inspired time-mixing/multi-timescale ideas may be expressed through VN97
dt/B/C parameterization. Do not insert a separate RWKV block into production.

### P5D7 — Realtime Mobile Gate

Benchmark the packed model on the actual Android runtime. Promotion depends on
measured:

- time-to-first-token;
- tokens/second;
- recurrent-state memory;
- peak RAM;
- thermals;
- sustained performance.

No parameter-count estimate is accepted as proof of realtime operation.

## P5D0 audit command

After installing the repository:

```bash
vn97-p5d-plan \
  --vn97-tokenizer /path/to/tokenizer.vn97tk1 \
  --output /tmp/p5d0-plan.json
```

This command does not download the teacher and does not start training.
