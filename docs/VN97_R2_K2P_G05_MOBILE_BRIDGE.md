# VN97 R2-K2P — G0.4 to G0.5 mobile-graph parity bridge

K2O closed the correctness-first G0.4 CUDA same-input parity question with a
predeclared holdout gate. Android G0.6 does not execute G0.4 directly: it
executes the shared-weight G0.5 recurrent graph with a valid_length input.

K2P closes that evidence gap before any physical-device claim is made.

## Real graph

K2P exports one real G0.5 recurrent-8 graph from the exact same G0.3 capsule
and inherited Mamba-2 2.7B weights proven by K2O.

The graph contract is:

    input_ids[1,8]
    valid_length[1]
    conv_state
    ssm_state

with the same explicit recurrent state and one shared copy of the inherited
weights.

## Decode bridge

Six new prompts are run through the PyTorch G0.4 recurrent oracle and the
G0.5 ONNX graph with valid_length=1. Thirty-two teacher-forced continuation
decisions per prompt remain same-input.

## Parallel-prefill bridge

Four additional cases exercise valid lengths 1, 2, 4 and 8. The reference is
repeated exact G0.4 steps. G0.5 performs one parallel recurrent chunk, then
continues for eight same-input decode decisions.

Full conv/SSM state error after the parallel prefill call is recorded, but is
diagnostic only. The acceptance boundary is behavioral because G0.5 changes
the legal accumulation order of the parallel SSD factorization.

## Predeclared acceptance

K2P passes only if:

- all compared logits are finite;
- every top-1 mismatch is at most one local FP16 ULP below the PyTorch
  reference top-1;
- every mismatch retains at least 4/5 top-5 overlap.

These are the already-locked K2O decision semantics, applied to a new graph
and new probe set before Android execution.

A PASS means the real G0.5 graph is a valid mobile-runtime bridge for this
lineage. It still does not claim S21 FE execution, RAM, latency, thermal or
provider qualification. Those require G0.6/G0.7 device evidence.

## Run

    %cd /kaggle/working/VN97
    !git pull --ff-only
    !bash tools/kaggle_r2_k2p_g05_mobile_bridge.sh
