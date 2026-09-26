# P5D4A — Expanded Falcon3-Mamba Teacher Corpus

P5D3B produced a very strong held-out relational signal, while the frozen P4
probe remained 0/12. That result means the 309M VN97 student has learned useful
teacher-relative representation geometry but has not yet acquired enough
task-level capability to justify ternary QAT.

P5D4A therefore expands the teacher supervision before any additional
compression.

## Why expansion now

The first sealed teacher corpus contains only 600 records. P5D3B nearly
saturated its relational holdout, so repeatedly optimizing the same 500
training records would mostly increase overfitting risk.

P5D4A increases coverage to 2400 records:

```text
P4 curriculum:
  240 records/category x 6 categories = 1440

P3 general language:
  960 records

Total:
  2400 records
```

The original 600 sealed teacher records are copied into the expanded work
directory and reused exactly. Only 1800 new Falcon3-Mamba generations are
performed.

## Teacher identity

The exact teacher revision is read from the parent P5D1 report. P5D4A refuses
to continue if:

- the parent P5D1 hashes do not verify;
- the parent is not exactly the sealed 600-record corpus;
- any seeded record has a different teacher revision;
- the expanded corpus resolves to a different teacher revision.

The Falcon3 tokenizer/runtime stack remains pinned to:

```text
transformers==4.46.1
tokenizers==0.20.3
```

## Format

The expanded artifact intentionally remains **VN97P5D1-compatible**:

```text
/kaggle/working/p5d4a-final/
  teacher-corpus.jsonl
  p5d1-report.json
  SHA256SUMS
```

This avoids introducing a second teacher-corpus format. Downstream distillation
can use the existing sealed-corpus verifier unchanged.

## Storage

The exact Falcon3-Mamba revision is downloaded temporarily. P5D4A requires at
least 15 GiB free before launch and removes only its own teacher cache after
successful completion.

Keep at minimum:

- `p5d1-final` for lineage;
- selected `p5d3b-c0-final`;
- `p5d3a-final` until the next stage is sealed.

## Run

Fresh:

```bash
bash tools/kaggle_p5d4a_teacher_expansion.sh fresh \
  /kaggle/working/p5d1-final \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/p3-corpus \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/held-out-p4.jsonl
```

Resume:

```bash
bash tools/kaggle_p5d4a_teacher_expansion.sh resume \
  /kaggle/working/p5d1-final \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/p3-corpus \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/held-out-p4.jsonl
```

Final marker:

```text
VN97P5D4A status=EXPANDED_TEACHER_CORPUS_READY
records=2400
reused_parent=600
new_teacher_generations=1800
```

P5D4B will then continue from the selected P5D3B student rather than starting
a new model.
