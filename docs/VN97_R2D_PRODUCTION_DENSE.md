# VN97-R2D Production Dense Training Contract

## Scope

R2-D scales the already-validated VN97-R2 architecture into the canonical
production dense checkpoint. It does not redesign the model and it does not
introduce Transformer/LLaMA/Mamba backends.

The locked target is the existing `r2_mobile_1b_config`, approximately
1.09B parameters at vocabulary size 4096 and constrained to the 0.9B-1.3B
production class.

## Canonical stage order

Production training is strictly ordered:

1. `dense_pretrain`
2. `instruction_reasoning`
3. `tool_action`
4. `capability`

Every stage after dense pretraining must identify the exact parent checkpoint
SHA-256. A later stage cannot skip over an earlier stage.

## Dataset identity

Every R2-D corpus manifest binds:

- tokenizer SHA-256;
- exact train and validation shard SHA-256 values;
- record and byte counts;
- task-family labels;
- parent checkpoint identity where required.

Duplicate shard digests are rejected. Every stage requires both train and
validation shards.

## Dense-first rule

R2-D is full-parameter dense training.

The production contract rejects:

- adapter-only training as a replacement for the canonical model;
- QAT;
- INT4;
- ternary lowering;
- any other quantized rescue path before fresh dense validation.

Mixed precision FP16/BF16 is allowed because it is a numerical execution
choice, not a quantized model format.

## Memory rule

The 1B model must not use the eager affine-scan autograd implementation for
production training. That implementation remains a pilot/reference path.

Before a production run, the recipe must enable:

- memory-efficient selective scan;
- activation checkpointing;
- full-parameter training.

The resource estimator reports a conservative lower bound before activations.
For AdamW it accounts for weights, gradients and optimizer/master-weight
storage. Optimizer-state offload may move the 12-byte/parameter Adam/master
budget to host memory.

For a 16 GB T4, optimizer offload can make the static pre-activation lower
bound fit, but that does **not** prove the run will fit. Activation storage,
temporary kernels, CUDA allocator fragmentation and data buffers still require
runtime profiling. Therefore a T4 must not start 1B production training until
the memory-efficient scan implementation and measured preflight are present.

## GPU use policy

R2-D should spend GPU quota only on work that requires GPU:

- dense pretraining;
- instruction/reasoning training;
- tool/action training;
- capability curriculum;
- measured throughput and memory profiling.

Architecture, corpus identity, resume semantics, stage validation, report
schemas and fail-closed gates should be validated on CPU/CI first.

## Next implementation block

R2-D2 must implement the production execution path behind this contract:

- memory-efficient selective scan suitable for backward;
- activation checkpointing across R2 blocks;
- mixed-precision autocast/scaler policy;
- gradient accumulation;
- optimizer-state offload or another exact full-parameter strategy;
- resumable stage checkpoints bound to corpus/recipe identities;
- measured VRAM preflight rather than static estimates alone.

No production GPU run is accepted until these gates pass.
