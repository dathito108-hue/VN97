# P4E-M — Contract-Constrained Numeric Beam Audit

P4E-L found a mixed failure mode on 30 held-out arithmetic tasks:

- current greedy solve: 1/30;
- correct complete-answer likelihood rank-1: 7/30;
- rank-3: 14/30;
- rank-5: 22/30;
- first correct token top-1: 3/30;
- first correct token top-5: 13/30;
- exact chat prefix match: 30/30;
- rank-1 but greedy-decode failure: 6/30.

This means some correct answers are globally preferred as sequences even though
local greedy decoding leaves the correct path early.

P4E-M performs no training and does not use an arithmetic calculator. It tests
whether a contract-constrained beam search can recover those latent answer
paths using only the same VN97 model logits.

The numeric contract permits only decimal digits plus the normal terminating
newline/EOS. The arithmetic value itself is not computed by the decoder.

The audit compares:

- current production-style greedy decoding;
- raw greedy decoding without repetition constraints;
- numeric beam width 4;
- numeric beam width 8;
- numeric beam width 16.

If beam search materially improves held-out reasoning, the next milestone can
promote a narrowly scoped numeric-output decoding policy. If it does not, the
remaining bottleneck is primarily representation/training rather than search.

## Kaggle

```bash
bash tools/kaggle_p4e_numeric_beam_audit.sh \
  /kaggle/working/p4e-k-final \
  /kaggle/working/VN97-P4E-M-beam.json
```

No model weights are changed.
