# P4E-K — Interference-Safe Reasoning Repair

P4E-J demonstrated that simply increasing arithmetic coverage is not safe for
the current VN97 checkpoint. Reasoning rose only from 1/30 to 2/30 while
numeric copy fell from 51/60 to 39/60 and tool intent fell from 30/30 to 24/30.

P4E-K therefore starts again from the P4E-I checkpoint and treats every
reasoning update as a candidate that must pass preservation gates.

## Candidate search

Each candidate is trained independently from the same P4E-I parent.

The default profiles are:

- **balanced** — 1200 reasoning, 1600 numeric-copy, 1600 capability anchors,
  1200 P3 replay records, LR 4e-6;
- **reasoning_cautious** — 1800 reasoning, 1800 numeric-copy, 1800 anchors,
  1200 P3 replay records, LR 3e-6;
- **retention_heavy** — 1000 reasoning, 2000 numeric-copy, 2000 anchors,
  1600 P3 replay records, LR 2e-6.

All candidates use one epoch and completion-aligned supervision.

Tool-intent and authority anchors are always protected first. Remaining anchors
cover instruction following, memory use, and structured cognition.

## Held-out safety

All frozen P4 reasoning validation expressions are excluded from candidate
training. If a dev suite is supplied, its reasoning expressions are excluded as
well.

The 60-task numeric-copy validation suite is never used as candidate training
data.

## Preservation gates

A candidate is considered safe only if it remains within the parent P4E-I
baseline by default:

- overall canonical score: at most 2 passes lower;
- numeric copy: at least 48/60 and at most 3 passes lower than baseline;
- tool intent: at most 1 pass lower;
- authority behavior: at most 1 pass lower;
- instruction, memory, structured cognition: at most 2 passes lower each;
- dev suite: at most 1 pass lower when supplied;
- completion-token loss/top-1 and P3 retention remain inside tight bounds.

Among safe candidates, P4E-K selects lexicographically by:

1. reasoning pass count;
2. overall canonical score;
3. numeric-copy score;
4. dev score;
5. token validation loss.

If a safe candidate reaches at least 5/30 reasoning while maintaining the
baseline overall canonical score, the search stops early to save GPU time.

## Resume behavior

Each candidate has its own resumable optimizer checkpoint. Completed candidates
also have a signed deployment checkpoint plus canonical JSON evidence in
`/kaggle/working/p4e-k-work`.

A resumed run skips completed candidates, resumes an interrupted candidate, and
continues the search without retraining completed candidates.

## Kaggle

Fresh run:

```bash
bash tools/kaggle_p4e_interference_safe_reasoning.sh fresh \
  /kaggle/working/p4e-i-final \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/p3-corpus \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/held-out-p4.jsonl
```

Resume after interruption:

```bash
bash tools/kaggle_p4e_interference_safe_reasoning.sh resume \
  /kaggle/working/p4e-i-final \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/p3-corpus \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/held-out-p4.jsonl
```

P4E-K does not use the rejected P4E-J checkpoint as its parent.
