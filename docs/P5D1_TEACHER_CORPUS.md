# P5D1 — Falcon3-Mamba Teacher Corpus

P5D1 is the first stage that transfers intelligence from the large SSM teacher
without trying to remap teacher tensors into VN97.

## Teacher execution

Teacher: `tiiuae/Falcon3-Mamba-7B-Instruct`.

The launcher requires two CUDA GPUs and loads the teacher in FP16 using
Hugging Face Transformers with a balanced two-GPU device map. The model card
uses the standard `AutoModelForCausalLM` / `AutoTokenizer` interface and
supports `device_map` loading.

P5D1 does not install the teacher into VN97 and does not export teacher weights.

## Pilot corpus

The first sealed pilot contains 600 prompts:

- 60 instruction-following;
- 60 reasoning/planning;
- 60 memory-use;
- 60 structured-cognition;
- 60 tool-intent;
- 60 authority-behavior;
- 240 general-language prompts sampled deterministically from P3 training.

Frozen P4 validation/dev prompts are excluded.

The corpus stores:

- the prompt messages;
- teacher response;
- reference response when one already exists;
- teacher repository;
- exact resolved teacher revision;
- deterministic record ID.

This lets later P5D stages distinguish teacher behavior from existing reference
targets instead of silently replacing one with the other.

## Resume semantics

Every generated response is written as an individual atomic record under
`p5d1-work/records`.

If Kaggle disconnects, `resume` reloads the exact same manifest and skips
already completed records. A different prompt manifest or teacher revision is
rejected.

## Output

Successful completion creates:

```text
/kaggle/working/p5d1-final/
  teacher-corpus.jsonl
  p5d1-report.json
  SHA256SUMS
```

The final marker is:

```text
VN97P5D1 status=TEACHER_CORPUS_READY ...
```

## Kaggle

Fresh:

```bash
bash tools/kaggle_p5d1_teacher_corpus.sh fresh \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/p3-corpus \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/held-out-p4.jsonl
```

Resume:

```bash
bash tools/kaggle_p5d1_teacher_corpus.sh resume \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/p3-corpus \
  /kaggle/input/datasets/duongtuan1/vn97-p4c-resume/held-out-p4.jsonl
```

The teacher model is intentionally not part of the resulting VN97 runtime.
