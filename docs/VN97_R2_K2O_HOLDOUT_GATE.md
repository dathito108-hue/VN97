# VN97 R2-K2O — predeclared holdout parity gate

K2N measured 512 same-input decisions across eight prompts and found 511 exact
top-1 matches. The single mismatch was the already-known prompt-0 step-29
boundary, with a PyTorch reference gap of exactly one local FP16 ULP and full
top-5 overlap.

K2O turns that diagnostic evidence into a **predeclared holdout gate** before
looking at any new holdout result.

## Holdout set

K2O uses twelve prompts that do not appear in K2N, covering:

- recurrent/model reasoning;
- code-like planning;
- sequential/parallel execution;
- mobile inference;
- game-agent and trading-like contexts;
- English and Vietnamese text.

Each prompt contributes 64 teacher-forced same-input decisions, for 768
holdout decisions total.

## Predeclared gate

The holdout passes only if all of the following are true:

1. all compared logits are finite;
2. every top-1 mismatch, if any, is no more than **one local FP16 ULP** below
   the PyTorch reference top-1 at that exact decision;
3. every mismatch retains at least **4/5** top-5 candidate overlap.

The one-ULP rule is now an acceptance criterion **only for this holdout gate**
because it is fixed in code before the holdout is measured. It does not
retroactively redefine K2N.

A K2O PASS closes the CUDA same-input parity question for this G0.4 lineage.
It still does **not** authorize production or prove Android/mobile-provider
parity. Those remain separate validation steps.

## Run

    %cd /kaggle/working/VN97
    !git pull --ff-only
    !bash tools/kaggle_r2_k2o_holdout_gate.sh
