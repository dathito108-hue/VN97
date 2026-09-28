# VN97 R2-K2N — ULP-aware same-input decision audit

K2M diagnosed the step-29 disagreement as distributed near-tie sensitivity.
Several unrelated layer/state perturbations can move the result across the
decision boundary, and the interpolation curves are not monotonic after FP16
quantization. That is evidence against treating one listed layer as a unique
repair target.

K2N therefore stops local arithmetic edits and measures how often this kind of
decision disagreement occurs on a broader same-input campaign.

## Campaign

Eight fixed prompts cover English, Vietnamese, code-like, reasoning-like and
state-space-model contexts.

PyTorch CUDA greedily generates 64 reference tokens for each prompt. ONNX
Runtime CUDA is then teacher-forced with exactly those reference tokens, so all
512 decision comparisons retain same-input semantics even if an argmax differs.

For every decision K2N measures:

- exact top-1 identity;
- full-logit PyTorch/ORT error;
- top-5 overlap.

For every top-1 disagreement it additionally measures the PyTorch reference
gap between the reference token and the token chosen by ORT in units of the
local downward FP16 ULP at the reference top-1 logit.

A disagreement at <= 1 reference ULP is labeled diagnostically as a numerical
near-tie. This is **not** an acceptance threshold and does not authorize
production.

## Why this matters

If disagreements remain rare and all occur only inside a one-ULP reference
near-tie with a stable top-5 neighborhood, the evidence supports a numerical
tie-boundary interpretation rather than a semantic model mismatch.

If any disagreement occurs outside that envelope, K2 must continue numerical
localization before parity can be locked.

K2N changes no weights, architecture, production graph, threshold, or
activation policy.
