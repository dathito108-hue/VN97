# VN97 R2-K2A — real VN97 native → ONNX Runtime CUDA parity

K2A starts only after the real K1I all-layer semantic gate has passed.

K1I proves that the transferred Mamba-2 2.7B core and the VN97 native
recurrent equations agree layer-by-layer when given identical inputs and
recurrent states.

K2A measures the next boundary:

    VN97 native PyTorch step
            ↓
        ONNX export
            ↓
      ONNX Runtime CUDA

## Why step graph first

The first real ONNX parity gate intentionally uses the G0.4 one-token explicit
state graph rather than the G0.5 parallel-prefill graph.

This isolates export/runtime correctness from parallel-scan reassociation.

The graph contract is:

    input_ids [1]
    conv_state [64,1,5376,4]
    ssm_state [64,1,80,64,128]

and outputs:

    logits [1,50288]
    next_conv_state
    next_ssm_state

Once K2 step parity is understood, K2P can validate the G0.5
parallel-prefill/same-graph path independently.

## Precision correction before export

K2 also aligns G0.4 dBx with the K1 semantic baseline.

The official Mamba-2 fallback and VN97 native path keep:

    dBx = einsum(dt, B, x)

in the inherited activation dtype. Earlier G0.4 lowering force-upcast dt/B/x to
float32. That old behavior is not used by K2.

The dA branch remains float32 through A_log, matching the source semantics.

## Kaggle execution

K2A reuses the existing source checkpoint and passing K1I receipt.

It creates a zero-copy G0.3 capsule if needed, then exports one real dense G0.4
step ONNX bundle with external weight data.

To avoid T4 OOM:

1. run the VN97 PyTorch reference first;
2. copy expected logits/final states back to CPU;
3. delete the PyTorch model and clear CUDA cache;
4. require at least 10 GiB free VRAM;
5. create ONNX Runtime CUDA session;
6. run the same token sequences and explicit recurrent state.

This prevents the 2.7B PyTorch model and the 2.7B ORT session from residing on
the T4 at the same time.

## Measurement-first gate

K2A intentionally emits status MEASURED, not PASS.

It records:

- exact K1I receipt ID;
- G0.4 manifest ID;
- capsule/source-weight lineage;
- active ORT provider;
- PyTorch/ORT logits max/mean absolute error;
- logits max FP16 epsilon units;
- final conv-state max/mean error and epsilon units;
- final SSD-state max/mean error and epsilon units;
- exact greedy argmax equality;
- GPU free memory before ORT session creation.

No production threshold is invented before observing the real runtime result.

K2B will turn the measured evidence into a fail-closed acceptance criterion.

## Run

On the same Kaggle T4 session:

    %cd /kaggle/working/VN97
    !git pull
    !bash tools/kaggle_r2_k2a_real_onnx_parity.sh

The existing 5.4 GB checkpoint and K1I receipt are reused.

Expected outputs:

    /kaggle/working/vn97-k2a-ort-parity.json
    /kaggle/working/VN97-R2-K2A-ORT-parity.zip

The first real export can take substantially longer than K1I because the full
2.7B graph and external data must be serialized once.
