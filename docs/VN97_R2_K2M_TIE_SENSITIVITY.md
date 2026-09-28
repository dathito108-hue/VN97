# VN97 R2-K2M — near-tie interpolation sensitivity

K2L showed that many distinct one-layer state repairs can flip the exact same
step-29 decision back from ORT token 322 to reference token 1583. This is a
warning against declaring the strongest listed layer to be the unique root
cause: the divergent ORT baseline is already on a near-tie boundary.

K2M measures how much of each influential PyTorch state is actually required
to cross that boundary.

## Candidate selection

K2M takes the two strongest K2L candidates from each mode:

    conv
    SSM
    conv + SSM

For each selected layer/mode, it interpolates only that state slice from the
ORT value toward the PyTorch value:

    state(alpha) = ORT + alpha * (PyTorch - ORT)

with:

    0, 1/16, 1/8, 1/4, 1/2, 3/4, 1

Interpolation is performed in FP32 and cast back to the inherited FP16 state,
so the experiment also records what fraction of state elements actually
change after quantization.

Every point is run through the unchanged ORT CUDA G0.4 step graph at the exact
K2K current token. K2M records the candidate-logit gap and the smallest alpha
that restores the reference top-1.

## Interpretation

If several unrelated layers/modes repair the decision at low alpha, the
evidence supports a distributed near-tie sensitivity rather than one faulty
layer. If one candidate repairs at low alpha while the others require nearly
the full PyTorch state, the boundary is much more localized.

K2M remains diagnostic only. It changes no weights, architecture, production
graph, acceptance threshold, or activation policy.
