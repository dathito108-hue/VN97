# P4E-C — Targeted Generation Repair

P4E-A exposed the dominant P4D failure mode. P4E-B then measured a weight-neutral
chat-boundary recovery and improved the sampled score from 1/30 raw to 5/30
recovered, with zero regressions. Boundary recovery therefore helps, but does
not explain or repair the remaining generation failures.

P4E-C is a targeted training repair. It does **not** change VN97 architecture.

## Root cause addressed

P3/P4C historical chat training supervised literal textual role markers.
P4D used completion-aligned supervision for new P4D examples, but its P3 replay
still used the historical marker-supervised encoder. That replay continued to
reinforce the old `<|assistant|>` emission behavior.

P4E-C makes both parts completion-aligned:

- new targeted repair examples use `encode_chat_completion_messages`;
- P3 replay also uses `encode_chat_completion_messages`;
- P3 retention evaluation remains on the historical metric so regression is
  still measured against the existing language baseline.

## Repair curriculum

The repair set contains 600 deterministic records per canonical P4D category,
3600 total. It is generated with a new seed and explicitly filtered so no prompt
overlaps the frozen P4D training or validation sets.

The default run adds 1000 deterministic P3 replay records and performs two
epochs at a reduced learning rate of `2e-5`.

## Gates

P4E-C records both raw and boundary-recovered generation before and after
training. Eligibility requires bounded token-validation regression, bounded P3
retention regression, a strict raw-generation gain on the unseen P4D probe, and
no regression in boundary-recovered generation.

The optional 12-task P4 dev suite remains held out and is measured before/after.

## Kaggle

Fresh run:

```bash
bash tools/kaggle_p4e_targeted_repair.sh fresh \
  /kaggle/working/p4d-final \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/p3-corpus \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/held-out-p4.jsonl
```

Resume after interruption:

```bash
bash tools/kaggle_p4e_targeted_repair.sh resume \
  /kaggle/working/p4d-final \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/p3-corpus \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/held-out-p4.jsonl
```

Output is written to `/kaggle/working/p4e-c-final`. Checkpoints are written
periodically under `/kaggle/working/p4e-c-work`.
