# P4E-J — Held-Out-Safe Arithmetic Coverage Repair

P4E-I repaired numeric transport strongly: held-out numeric copy improved from
0/60 to 51/60. Canonical generalization improved from 71/180 to 79/180, but
reasoning remained only 1/30.

That result isolates the remaining bottleneck to arithmetic computation rather
than numeric output transport.

## Coverage strategy

P4E-J covers nearly the complete arithmetic domain used by the P4 reasoning
task family:

- addition: 11..96 + 11..96;
- subtraction: 30..119 minus valid positive subtrahends;
- multiplication: 3..20 × 2..12;
- parenthesized multiplication: 2..11 × (2..9 + 2..9).

All frozen P4 validation expressions are excluded from training.

If the optional 12-task development suite is provided, any arithmetic
expressions found in that suite are also excluded. This keeps the measured
expressions genuinely unseen even though the surrounding arithmetic domain is
densely covered.

P4E-J also replays non-reasoning anchors from P4E-G and deterministic P3
examples to limit capability regression.

## Gates

Eligibility requires:

- bounded completion-token validation regression;
- bounded P3 retention regression;
- tool-intent and authority non-regression within two passes;
- no material overall canonical regression;
- held-out numeric copy >= 48/60;
- held-out reasoning >= 8/30.

## Kaggle

Fresh:

```bash
bash tools/kaggle_p4e_arithmetic_coverage_repair.sh fresh \
  /kaggle/working/p4e-i-final \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/p3-corpus \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/held-out-p4.jsonl
```

Resume after interruption:

```bash
bash tools/kaggle_p4e_arithmetic_coverage_repair.sh resume \
  /kaggle/working/p4e-i-final \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/p3-corpus \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/held-out-p4.jsonl
```
