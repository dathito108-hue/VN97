# VN97 R2-K2K — state-swap divergence attribution

K2J found a real but narrow closed-loop divergence:

- two of three prompts stayed exact for all 32 generated tokens;
- one prompt stayed exact for 28 tokens and first diverged at step 29.

Because both histories are still identical immediately before that prediction,
the divergence can be attributed with a same-input state-swap experiment.

## Eight-way matrix

K2K replays the common history to the state immediately before the token whose
processing produces the divergent step-29 logits. It captures the recurrent
state from both runtimes:

    Pconv, Pssm = PyTorch recurrent state
    Oconv, Ossm = ORT recurrent state

The exact same current token is then evaluated through eight combinations:

    PyTorch(Pconv, Pssm)
    PyTorch(Pconv, Ossm)
    PyTorch(Oconv, Pssm)
    PyTorch(Oconv, Ossm)

    ORT(Pconv, Pssm)
    ORT(Pconv, Ossm)
    ORT(Oconv, Pssm)
    ORT(Oconv, Ossm)

For every result K2K records the selected top-1 token, top-5 candidates,
reference-vs-ORT candidate logit gap, and full-logit error relative to the
canonical PyTorch/PyTorch-state replay.

## Interpretation

- runtime_step_dominant:
  PyTorch chooses the reference token regardless of state source while ORT
  chooses the ORT token regardless of state source.
- ssm_state_dominant:
  the SSM-state source controls the rank flip across both runtimes.
- conv_state_dominant:
  the convolution-state source controls the rank flip across both runtimes.
- accumulated_state_dominant_mixed:
  full PyTorch state produces the reference token and full ORT state produces
  the ORT token across both runtimes, but conv/SSM cross-splices interact.
- mixed_runtime_and_state:
  neither runtime nor one state component alone explains the divergence.

K2K is diagnostic only. It does not change weights, the G0.4 graph, numerical
thresholds, or production activation.

## Run

    %cd /kaggle/working/VN97
    !git pull --ff-only
    !bash tools/kaggle_r2_k2k_state_swap.sh

Expected evidence:

    /kaggle/working/vn97-k2k-state-swap.json
    /kaggle/working/VN97-R2-K2K-state-swap.zip
