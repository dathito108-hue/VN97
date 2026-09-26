# P5D3A — Relational Teacher Representation Targets

P5D2 produced a real held-out behavioral learning signal in the 309M VN97-RT
student. P5D3 therefore moves to representation/dynamics distillation, but does
so without assuming that Falcon3-Mamba and VN97 share a tokenizer or hidden
dimension.

## Why not direct hidden-vector MSE

The teacher and student differ in:

- tokenizer and token boundaries;
- hidden width (4096 vs 1536);
- number of layers (64 vs 32);
- recurrence implementation.

Direct hidden-vector MSE would therefore impose a false one-to-one alignment.

## Cross-tokenizer relational target

For every P5D1 record, P5D3A builds the same canonical semantic excerpt from
the prompt and teacher response. If the UTF-8 transcript is long, it keeps a
deterministic head+tail excerpt so the prompt and answer are both represented.

The exact same excerpt can later be encoded by the VN97 tokenizer.

Falcon3-Mamba is run once with hidden-state output enabled. Four teacher hidden
depths are sampled:

```text
16, 32, 48, 64
```

At each depth, the sequence is split into four relative segments. Each segment
is mean pooled and normalized, then converted into a 4x4 cosine-similarity Gram
matrix.

The stored target therefore describes **relationships inside the sequence**
rather than teacher-basis coordinates.

This makes it dimension-independent and substantially more robust to
cross-tokenizer boundaries.

Relative segment norms are also stored. P5D3B can use:

- per-layer Gram loss;
- change-in-Gram loss across depth as a representation-dynamics target;
- relative norm-profile loss.

## Data isolation

P5D3A preserves the same deterministic P5D2 split:

- 500 train records;
- 100 holdout records.

Holdout relational targets may be measured during evaluation but must not be
used for gradient updates.

## Artifact

Successful completion creates:

```text
/kaggle/working/p5d3a-final/
  relational-targets.jsonl
  p5d3a-report.json
  SHA256SUMS
```

These files are compact because they store relational matrices rather than 4096
dimensional hidden vectors.

Final marker:

```text
VN97P5D3A status=RELATIONAL_TARGETS_READY records=600 train=500 holdout=100 ...
```

## Kaggle disk plan

P5D3A temporarily downloads the exact Falcon3-Mamba revision recorded in P5D1.
The launcher requires at least 15 GiB free before starting.

Keep:

- `p5d1-final`;
- selected `p5d2-c0-final`.

The unselected candidate and old P5D2 work/log files can be deleted before
starting.

After successful extraction, the P5D3A launcher deletes only its own temporary
teacher cache, so the disk is ready for P5D3B student training.

## Run

Fresh:

```bash
bash tools/kaggle_p5d3a_teacher_features.sh fresh \
  /kaggle/working/p5d1-final
```

Resume:

```bash
bash tools/kaggle_p5d3a_teacher_features.sh resume \
  /kaggle/working/p5d1-final
```

P5D3A does not modify the selected P5D2 student.
