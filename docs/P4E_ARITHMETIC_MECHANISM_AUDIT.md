# P4E-L — Arithmetic Mechanism Audit

P4E-K found a safe candidate that improved canonical generation from 79/180 to
88/180 and numeric copy from 51/60 to 55/60, but held-out reasoning remained
1/30. That makes another blind arithmetic fine-tune unjustified.

P4E-L performs no training. It asks whether the correct arithmetic answer is
already preferred by the model's sequence likelihood even when greedy decoding
fails.

For each of the 30 held-out reasoning prompts, P4E-L measures:

- canonical greedy solve;
- a normalized-expression solve;
- the mean teacher-forced log-probability of the correct answer;
- the same score for a compact set of arithmetic distractors;
- rank of the correct answer among those candidates;
- first correct answer-token rank/top-5;
- exact train/inference prefix equivalence.

Interpretation:

- many `rank1_but_decode_fail` cases suggest decoding/exposure behavior is the
  bottleneck;
- low correct-answer likelihood ranks suggest the model does not yet represent
  the arithmetic mapping itself;
- normalized prompts outperforming original prompts suggest language parsing is
  part of the bottleneck.

## Kaggle

```bash
bash tools/kaggle_p4e_arithmetic_mechanism_audit.sh \
  /kaggle/working/p4e-k-final \
  /kaggle/working/VN97-P4E-L-mechanism.json
```

No weights are changed.
