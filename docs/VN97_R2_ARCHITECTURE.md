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
- r2_cpu_pilot_config: roughly 50-100M-class architecture pilot depending on vocab.
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

Implemented:
- VN97TK1 package loader and exact vocabulary guard;
- reuse of canonical chat-completion training segmentation/windows;
- weight-sharing inference view compatible with current cognition engine;
- current retrieval embedding path over R2 hidden states;
- resumable dense trainer with deterministic epoch order;
- best-checkpoint selection by held-out dense loss;
- CPU/GPU-agnostic dense pilot CLI;
- explicit fast-path alignment stage before fresh validation.

Remaining in R2-B:
- structured tool/action end-to-end generation gate against the current planner;
- full memory/planner integration test with a trained R2 checkpoint.

### R2-C — Dense CPU/low-cost pilot

- 50-150M class pilot;
- real corpus windows;
- generation + semantic + tool + authority gates;
- checkpoint/resume;
- no quantization.

### R2-D — Production-scale dense training

- 1B-1.3B initial target;
- pretraining/distillation;
- instruction/reasoning;
- tool/action training;
- capability curriculum.

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
