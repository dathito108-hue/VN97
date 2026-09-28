# VN97 R2-K2J — closed-loop generation stability

K2I measured a 32-step teacher-forced recurrent trajectory on the unchanged
G0.4 graph. All 32 top-1 decisions matched between PyTorch CUDA and ORT CUDA,
and state error oscillated rather than growing monotonically. However, the
conservative top-1 margin certificate did not hold on every step.

K2J therefore tests a stronger behavioral boundary: each runtime generates its
own continuation.

## Campaign

Three fixed prompts are primed identically. PyTorch CUDA then generates a
32-token greedy continuation for each prompt. After releasing the PyTorch
model, ONNX Runtime CUDA generates its own 32-token greedy continuation from
the same prompts and recurrent zero state.

At every common-prefix step K2J records:

- exact top-1 identity;
- full-logit error;
- top-1 margin certification;
- top-5 overlap;
- recurrent conv/SSM state parity at steps 1, 2, 4, 8, 16, 32.

If a runtime chooses a different token, K2J records the first divergence and
continues the ORT branch independently. Numerical state/logit comparisons are
not reported after the two token histories differ because those values no
longer have same-input semantics.

This directly tests whether observed backend numerical drift changes greedy
generation behavior under feedback.

K2J remains measurement-only. It does not change G0.4, weights, thresholds, or
production activation.

## Run

    %cd /kaggle/working/VN97
    !git pull --ff-only
    !bash tools/kaggle_r2_k2j_closed_loop.sh

Expected outputs:

    /kaggle/working/vn97-k2j-closed-loop.json
    /kaggle/working/VN97-R2-K2J-closed-loop.zip
