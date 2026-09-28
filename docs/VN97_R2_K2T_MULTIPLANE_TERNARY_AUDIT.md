# VN97 R2-K2T — multi-plane ternary projection audit

K2S physical-device measurements on the Exynos 2100 established that the
verified FP16 Mamba-2 2.7B graph executes correctly but is not realtime:

- CPU 1 thread: about 24.68 s/token
- CPU 4 threads: about 21.66 s/token
- CPU 8 threads: about 14.83 s/token
- XNNPACK strict cannot own the complete graph
- XNNPACK hybrid caused the isolated worker process to die
- NNAPI strict rejected the model on the Exynos target
- NNAPI hybrid created a session but failed during execution

K2T therefore stops treating execution-provider selection as the main
performance path and audits a VN97-native compressed projection candidate.

## Candidate representation

K2T does not overwrite the verified FP16 baseline.

For every Mamba-2 in_proj and out_proj matrix, K2T applies the canonical
VN97T2 per-row ternary quantizer iteratively to the residual:

    W ~= T1*S1 + T2*S2 + ... + Tn*Sn

Each ternary plane is physically 2 bits/weight plus one FP32 scale per output
row. K2T measures 1 through 4 residual planes, corresponding to 2, 4, 6, and
8 projection bits/weight.

Embedding, norms, recurrent state parameters, convolution weights, and other
non-projection tensors remain exact FP16 in the size accounting.

This is diagnostic only. Weight-space similarity is not a behavior gate.

## Kaggle command

Run on the same T4 / Torch 2.10.0+cu128 environment used by the locked R2
campaign:

    %cd /kaggle/working/VN97
    !git pull --ff-only
    !bash tools/kaggle_r2_k2t_multiplane_ternary_audit.sh

The script downloads the pinned source only when a K2T G0.3 capsule is not
already present. The source directory is then removed while the hardlinked
capsule payload remains once.

Output:

    /kaggle/working/vn97-k2t-multiplane-ternary.json

## What K2T measures

For all 128 large projection matrices:

- global and per-projection-type relative RMSE;
- cosine similarity;
- maximum absolute reconstruction error;
- per-plane sparsity;
- worst matrices by plane count;
- estimated packed projection bytes;
- estimated total model bytes with non-projection tensors retained exact;
- compression ratio versus the unique FP16 source weights.

K2T does not define an acceptance threshold and does not authorize production.
The next gate must measure actual model behavior on held-out prompts before any
compressed candidate can replace the verified FP16 baseline.
