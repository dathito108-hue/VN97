# P4E-H — Reasoning Bottleneck Audit

P4E-G improved canonical generalization from 58/180 to 71/180, but
reasoning/planning remained 0/30 despite 1400 dedicated repair examples.

P4E-H performs no training. It tests whether the remaining reasoning failure is
primarily:

- arithmetic computation failure;
- inability to copy numeric answers;
- first-target-token uncertainty;
- or broader teacher-forced answer-token weakness.

For the same 30 held-out reasoning tasks it compares:

- canonical solve generation;
- a numeric copy control using the exact correct answer;
- first target token top-1/top-5;
- teacher-forced answer-token accuracy;
- operation-level breakdown for add/subtract/multiply/parenthesized multiply.

If numeric copying succeeds while solving fails, the next repair should target
computation rather than tokenizer/output formatting.

## Kaggle

```bash
bash tools/kaggle_p4e_reasoning_audit.sh \
  /kaggle/working/p4e-g-final \
  /kaggle/working/VN97-P4E-H-reasoning.json
```

No weights are changed.
