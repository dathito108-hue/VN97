# VN97 R2-K1I — all-layer same-input semantic parity

K1H cannot run on the current Kaggle image because the optional optimized
recurrent CUDA update kernels are not installed. K1I removes that dependency.

K1G already showed near-zero same-input error at the first failing trajectory
layer (layer 2). K1I generalizes that test across all 64 layers and multiple
real token probes.

For every official fallback trajectory token and every layer, K1I:

1. takes the exact official residual input;
2. compares official block RMSNorm with the VN97 RMSNorm reference;
3. clones the exact official recurrent conv/SSD state before the layer step;
4. runs official Mamba-2 fallback step;
5. runs VN97 reference step with the same normalized input, same initial state,
   and same inherited weights;
6. compares block norm, mixer output, conv state and SSD state;
7. advances the trajectory only with the official result.

Therefore earlier VN97 rounding drift cannot contaminate later layer tests.

## Numerical criterion

K1I does not use a hand-widened absolute tolerance.

For each tensor element:

    epsilon_units =
        abs(reference - candidate)
        / (FP16_epsilon * max(1, abs(reference)))

Default PASS requires every tested stage of every layer to stay within
**2 FP16 epsilon units**.

The receipt records all 64 layer maxima, violations, and the worst same-input
result.

## Kaggle T4

Run:

    %cd /kaggle/working/VN97
    !git pull
    !bash tools/kaggle_r2_k1i_all_layer_semantic.sh

The existing 5.4 GB source is reused.

Output:

    /kaggle/working/vn97-k1i-semantic-parity.json

K1I proves local operator semantic parity across all 64 transferred layers. It
does not replace K2 ONNX Runtime parity.
