# P4E-G — Compositional Generalization Repair

P4E-F fixed a systemic train/inference tokenization mismatch. On the unchanged
P4E-C checkpoint, canonical segmented chat inference raised strict unseen
generation from 0/180 legacy raw to 58/180 canonical raw.

Measured canonical category scores were:

- instruction following: 5/30;
- reasoning/planning: 0/30;
- memory use: 4/30;
- structured cognition: 5/30;
- tool intent: 25/30;
- authority behavior: 19/30.

P4E-G therefore stops changing the decoder and targets the remaining capability
gap.

## Training policy

P4E-G keeps the VN97 architecture unchanged.

The new repair curriculum contains:

- 800 instruction/copy-binding records;
- 1400 arithmetic/reasoning records;
- 800 contextual binding/memory records;
- 1000 structured JSON composition records;
- 200 tool-intent anchors;
- 200 authority anchors.

Prompts use new templates and are explicitly disjoint from frozen P4D training,
P4D validation, and P4E-C targeted repair prompts.

The run also replays 1000 deterministic P3 chat records using completion-aligned
supervision.

## Canonical generation gate

All P4E-G unseen generation measurements use
`generate_chat_completion(...)`, which encodes role markers, message contents,
and line suffixes with the same segmentation used by training.

The legacy whole-string tokenizer path is not used as the promotion gate.

## Gates

Eligibility requires:

- bounded completion-token validation regression;
- bounded P3 retention regression;
- tool-intent and authority scores must each remain within two passes of their
  measured pre-run values;
- canonical unseen generation must improve by at least 12 tasks overall;
- the four weak categories must contribute at least 12 net gains;
- reasoning/planning must reach at least 5/30.

This is an intermediate capability gate, not a final production claim.

## Kaggle

Fresh run:

```bash
bash tools/kaggle_p4e_compositional_repair.sh fresh \
  /kaggle/working/p4e-c-final \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/p3-corpus \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/held-out-p4.jsonl
```

Resume after interruption:

```bash
bash tools/kaggle_p4e_compositional_repair.sh resume \
  /kaggle/working/p4e-c-final \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/p3-corpus \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/held-out-p4.jsonl
```

Periodic resume state is stored in `/kaggle/working/p4e-g-work`. Final evidence
is stored in `/kaggle/working/p4e-g-final`.
