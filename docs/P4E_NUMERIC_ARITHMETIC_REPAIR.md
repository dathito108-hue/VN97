# P4E-I — Numeric Representation + Arithmetic Repair

P4E-H showed that the remaining reasoning failure is broader than arithmetic:

- held-out solve: 0/30;
- numeric copy control: 1/30;
- first answer token top-1: 1/30;
- first answer token top-5: 7/30;
- teacher-forced answer tokens: 60/109;
- canonical prefix: 30/30.

The model therefore needs stronger numeric transport before arithmetic-only
training can be expected to generalize.

## Curriculum

P4E-I keeps the VN97 architecture unchanged and trains three numeric layers:

1. **Numeric identity/copy** — 3000 exact integer copy tasks.
2. **Place-value primitives** — 2200 digit, one-column addition, and carry tasks.
3. **Direct arithmetic** — 6200 balanced add/subtract/multiply/parenthesized tasks.

It also includes 1000 non-reasoning anchors from the previous compositional
curriculum and 1000 deterministic P3 replay records.

A separate 60-task numeric-copy suite is held out from training.

## Gates

P4E-I measures:

- canonical P4 generalization;
- 60-task held-out numeric copy;
- optional 12-task development suite;
- completion-token validation;
- P3 retention.

Eligibility requires:

- bounded token-validation regression;
- bounded P3 retention regression;
- tool-intent and authority non-regression within two passes;
- no material overall canonical regression;
- at least 20/60 held-out numeric-copy passes;
- at least 5/30 held-out reasoning passes.

## Kaggle

Fresh:

```bash
bash tools/kaggle_p4e_numeric_arithmetic_repair.sh fresh \
  /kaggle/working/p4e-g-final \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/p3-corpus \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/held-out-p4.jsonl
```

Resume:

```bash
bash tools/kaggle_p4e_numeric_arithmetic_repair.sh resume \
  /kaggle/working/p4e-g-final \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/p3-corpus \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/held-out-p4.jsonl
```

Periodic state is written under `/kaggle/working/p4e-i-work`. Final evidence is
written under `/kaggle/working/p4e-i-final`.
