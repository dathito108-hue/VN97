# VN97 R2-K2I — 32-step recurrent trajectory stability

K2H showed that the local in-proj micrograph is already well behaved with the
canonical FP16 linear path, while neither the canonical nor widened conv-affine
micrograph removes the downstream dBx amplification.

This means there is no evidence-backed local rewrite to apply. K2I therefore
stops modifying arithmetic and asks the more important recurrent question:

> Does the observed backend drift remain semantically stable over a longer
> state-carry trajectory?

## Campaign

K2I reuses the unchanged G0.4 step graph and inherited Mamba-2 2.7B weights.
It runs one deterministic 32-token sequence through:

    PyTorch CUDA reference
    ONNX Runtime CUDAExecutionProvider

The exact same token sequence is supplied to both runtimes.

Every step records:

- full-logit max/mean absolute error;
- FP16 epsilon-normalized logit error;
- exact top-1 identity;
- reference top-1 margin;
- whether the observed max-logit perturbation is small enough to certify the
  same top-1 from the margin alone.

Full recurrent state is compared at exponentially spaced checkpoints:

    1, 2, 4, 8, 16, 32

This keeps the 2.7B campaign practical while exposing whether conv/SSM drift
grows, contracts, or oscillates over a materially longer horizon than K2A.

K2I is measurement-only. It does not define an acceptance threshold and does
not authorize production.

## Run

    %cd /kaggle/working/VN97
    !git pull --ff-only
    !bash tools/kaggle_r2_k2i_trajectory.sh

Expected outputs:

    /kaggle/working/vn97-k2i-trajectory.json
    /kaggle/working/VN97-R2-K2I-trajectory.zip
