# VN97-R2 Mobile-Native Intelligence Core

## Status

VN97-R2 is the canonical successor path for the VN97 intelligence core.

The existing VN97 Android, memory, planner, authority, tool, continuity, voice,
overlay, game-agent, trading-agent, and capability infrastructure is retained.
R2 replaces only the model architecture/training path that became trapped in
repeated P5 decoder/output repair cycles.

The legacy 309M/P5 chain remains evidence and a regression baseline. It is not
deleted and it is not silently promoted into R2.

## Locked architectural principles

1. **One VN97 model.** No Transformer/LLaMA/Mamba runtime backend is added.
2. **Selective SSM only.** R2 adopts the useful Mamba-1 principles (selective
   state update, input-dependent B/C/delta, causal local convolution, gating,
   constant recurrent state) but implements them in VN97-native code/format.
3. **Dense intelligence first.** The model must demonstrate stable generation,
   semantic correctness, reasoning, tool protocol, and authority behavior
   before QAT or ternary/mobile lowering is allowed.
4. **Fast and deep are one backbone.** Fast mode early-exits after a configured
   number of leading R2 blocks. Deep mode uses the full stack. There is no
   router to a second model.
5. **Realtime execution is split by cadence, not by model.** The model updates
   a canonical plan at a low planner frequency. A deterministic native executor
   can replay already-approved plan primitives at high reflex frequency.
6. **Capabilities are merged, not stacked as permanent backends.** A capability
   pack is a training/evidence/delta package. After approval its delta is baked
   into the single VN97-R2 checkpoint and recorded in the capability ledger.
7. **Existing M6 authority remains authoritative.** R2 does not bypass external
   write approval, non-replay side-effect rules, or canonical plan semantics.

## R2 selective SSM block

Reference dataflow:

    x
    |
    RMSNorm
    |
    input projection --------------------------.
    |                                          |
    u                                          gate
    |
    depthwise causal convolution
    |
    SiLU
    |
    x_proj -> [dt_low_rank, B, C]
       |          |       |
       v          |       |
    dt_proj       |       |
       |          |       |
       '---- exact-ZOH selective recurrence ---'
                    |
             recurrent state
                    |
             C readout + D skip
                    |
                gate(SiLU)
                    |
               output proj
                    |
                 residual

For each recurrent channel/state element:

    A = -exp(A_log)
    dA = exp(dt * A)
    zoh = expm1(dt * A) / A
    state_t = dA * state_(t-1) + zoh * B_t * u_t
    y_t = sum(state_t * C_t) + D * u_t

The recurrent state size is independent of context length. The reference
implementation is deliberately simple and auditable. Fused scan, ARM64 NEON,
and NPU kernels are later lowerings of the same equations, not alternate
models.

## Model scale

Three configurations are defined:

- r2_smoke_config: tiny correctness/CI model.
- r2_cpu_pilot_config: locked R2-C 50-150M dense pilot configuration
  (about 61.7M parameters with the byte-base tokenizer and about 64.7M at
  vocab 4096).
- r2_mobile_1b_config: production target around 1B-1.3B parameters depending
  on vocabulary size.

The 1B-class target is intentionally below 7B. Mobile viability is constrained
mainly by memory bandwidth, thermals, and sustained latency even when recurrent
state is small. Scale is increased only after profiling proves the smaller
configuration is insufficient.

## Fast/deep cognition

VN97R2Model exposes the same weights through two execution depths:

- **fast**: config.fast_layers, for low-risk short-deadline cognition.
- **deep**: all layers, for multi-step reasoning, higher risk, or any external
  write request.

A recurrent stream cannot change depth in-place. Switching fast/deep starts a
new recurrent state so state semantics remain deterministic.

External writes are always routed to the deep cognition path before the
existing M6 authority gate is reached.

## Realtime action architecture

The target runtime cadence is:

    perception/events
         |
         v
    VN97-R2 cognition (fast/deep, typically a few Hz)
         |
         v
    canonical VN97 plan + generation
         |
         v
    M6 authority / approvals
         |
         v
    deterministic native reflex executor (30-60 Hz where appropriate)

This lets game/device control remain responsive without running a billion-
parameter model on every frame.

## Capability Pack contract

A capability pack contains:

- stable name/version;
- exact parent checkpoint SHA-256;
- training-recipe SHA-256;
- dataset fingerprints;
- task families;
- tool schemas;
- a parameter delta plus held-out evidence.

The runtime does **not** load multiple capability models. After approval,
merge_capability_delta() bakes the delta into a new single checkpoint.
Promotion/rollback evidence is stored in the capability ledger.

Typical future packs:

- GAME / FPS
- TRADING
- CODING
- VISION
- DEVICE
- SPEECH

## Checkpoint contract

R2 checkpoints use schema VN97R2CP1 and bind:

- architecture id;
- complete architecture config;
- config fingerprint;
- training stage;
- state dict;
- capability ledger;
- metadata.

Loading rejects architecture/config mismatch.

## Canonical training order

R2 locks this order:

    architecture
      -> dense_pretrain
      -> instruction_reasoning
      -> tool_action
      -> capability
      -> fast_path_alignment
      -> fresh_validation
      -> QAT
      -> mobile_lowering

QAT cannot be used as a repair mechanism for a weak dense model.

Fresh validation must independently prove:

- generation stability;
- semantic/task correctness;
- instruction following;
- structured output validity;
- tool-call correctness;
- authority behavior;
- regression bounds.

Exact-match is mandatory for deterministic protocols (JSON/tool/action
contracts) but is not the sole intelligence metric for natural-language tasks.

## Quantization/mobile policy

The dense checkpoint is canonical until fresh validation passes.

After that:

- recurrent/selective dynamics are profiled for FP16/INT8 sensitivity;
- large matrix projections are candidates for INT4 or packed ternary 2-bit;
- sensitive tensors may remain at higher precision;
- all quantized paths must prove parity/regression bounds against the approved
  dense checkpoint;
- ARM64 NEON and NPU paths must implement the same R2 equations.

No mobile compression is allowed to change model semantics by introducing a
second backend.

## Migration from the P5 repair chain

The P5 evidence established:

- latent likelihood improved substantially;
- output expansion improved token metrics;
- runtime sequential/parallel parity was not the failure;
- decoder-policy ablation did not recover generation;
- larger output-only correction still produced zero exact holdout generations.

Therefore R2 does not continue the output-only repair ladder. The useful
lessons are retained as test gates, while the model core returns to a clean,
trainable selective-SSM foundation.

## Milestones

### R2-A — Foundation (this milestone)

- locked config/fingerprint;
- selective SSM reference block;
- constant recurrent state;
- full-sequence/recurrent-step parity;
- single-backbone fast/deep paths;
- realtime cadence contract;
- capability-pack merge contract;
- checkpoint contract;
- dense-first training promotion policy;
- CPU smoke pilot and tests.

### R2-B — Compatibility bridge + dense pipeline

Completed and merged:
- VN97TK1 package loader and exact vocabulary guard;
- canonical chat-completion segmentation/windows reused without a second model;
- weight-sharing inference view compatible with the existing cognition engine;
- retrieval embedding path over R2 hidden states;
- deterministic checkpoint/resume for dense training;
- held-out best-checkpoint selection;
- CPU/GPU-agnostic dense pilot CLI;
- associative selective-SSM scan for training/prefill and recurrent generation;
- explicit fast-path alignment before later fresh validation.

R2-B final validation passed in GitHub Actions run #36312385011. The temporary
validation workflow was removed before merge.

### R2-C — Dense CPU/low-cost pilot

R2-C is a real 50-150M dense intelligence pilot, not another architecture
experiment. Its harness is fail-closed before expensive compute:

- the pilot config must remain inside the locked 50-150M parameter class;
- exact duplicate records inside a split are rejected;
- exact train/validation record overlap is rejected;
- training corpus, validation corpus and tokenizer are fingerprinted;
- CPU runs preflight available host RAM;
- CUDA runs preflight free device VRAM, including Kaggle T4, and budgets the
  logarithmic affine-scan autograd graph rather than only recurrent-state size;
- `--preflight-only` verifies data, windows, probe coverage and memory without
  executing a model forward/backward pass;
- checkpoint/resume binds the run identity, tokenizer, original held-out
  baseline and persisted best-checkpoint SHA-256 before continuing;
- `--max-run-seconds` pauses at a batch boundary, writes a cryptographically
  identified resume state, and skips expensive final probes until training is
  complete;
- initial held-out loss is recorded before training and compared with the best
  dense checkpoint after training;
- optional held-out probes cover natural-language generation, structured
  protocols, tool/action JSON and external-write deep-route behavior;
- acceptance probe suites must contain natural-language, tool-action and
  external-write cases;
- reports store probe output hashes/metrics rather than raw generated content;
- quantization/QAT/ternary remain forbidden.

The R2-C lexical target-F1 is deliberately named as a target-alignment metric.
It is not presented as a general semantic-quality score. Production promotion
still requires the later fresh multi-axis validation gate before QAT.

### R2-D — Production-scale dense training

Production execution is split into fail-closed gates:

- **R2-D1** locks the 0.9B-1.3B full-parameter dense contract, corpus/checkpoint
  identities, curriculum order and static resource lower bounds.
- **R2-D2** adds exact memory-efficient selective scan and per-block activation
  checkpointing with forward/gradient parity.
- **R2-D3** adds the full-parameter trainer: CUDA autocast, gradient
  accumulation, CPU-offloaded AdamW moments, resumable stage identity and
  measured CUDA memory evidence.
- **R2-D4** binds corpus + recipe to measured preflight and exposes the
  production launcher.
- **R2-D5** packages a complete immutable VN97CORPUS1 + VN97TK1 T4 preflight
  campaign, keeps the release split held out, binds the generated Kaggle
  script, and seals passing measured evidence into a no-promotion receipt.
- **R2-D6** composes multiple sealed corpora into deterministic sharded
  training/validation/release packages, preserves source/license provenance,
  rejects cross-corpus leakage, and measures canonical supervised token/window
  scale without allocating the 1B model.
- **R2-D7** streams verified D6 shards directly into the canonical dense
  trainer, resumes from epoch/shard/record/window cursors only at optimizer
  boundaries, keeps release shards out of training/evaluation, and proves
  pause/resume parity against uninterrupted execution.
- **R2-D8** compiles stage-aware task-family weights into immutable per-epoch
  shard schedules, requires an explicit primary family for each training source
  manifest, enforces realized token-share accuracy, and binds the curriculum
  plan ID into D7 run/resume/checkpoint evidence.

A passing R2-D5 receipt proves execution-memory feasibility only. It does not
assert that the current preflight corpus volume is enough to train the 1B
model to production intelligence and it does not authorize training.

After measured feasibility, corpus-scale evidence and a deterministic
curriculum plan are proven, the remaining R2-D work is:

- measured GPU preflight execution and evidence sealing;
- production corpus acquisition/sealing to the canonical scale floor;
- quota-bounded dense pretraining through R2-D7 using an R2-D8 plan;
- instruction/reasoning stage training with its own R2-D8 policy;
- tool/action stage training with its own R2-D8 policy;
- capability stage training with its own R2-D8 policy;
- fast-path alignment;
- fresh multi-axis validation.

### R2-E — Mobile lowering

Only after dense validation:

- QAT;
- hybrid ternary/INT4;
- fused scan/recurrent kernels;
- ARM64 NEON;
- NPU delegate where device support is proven;
- thermal/latency/RAM profiling.

### R2-F — Android production integration

- replace legacy intelligence checkpoint behind the existing VN97 interfaces;
- keep VN97MEM1, planner, M6, tools, continuity, voice, overlay and agents;
- final turnkey APK acceptance.

## Current rule

Do not spend GPU quota on the legacy P5 repair ladder once R2 is canonical.
GPU time should be reserved for dense intelligence training after architecture,
data, evaluation, and resume behavior are already proven on CPU.
