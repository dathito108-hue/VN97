# P3 — Kaggle Free-GPU Sharded Campaign

This path keeps GitHub as the canonical repository/evidence store while splitting P3
training into independent CUDA candidate jobs that can be run in separate Kaggle GPU
sessions.

The sealed P3 corpus is unchanged:

- VN97CORPUS1:
  `77e9142054f82b78c8ca0ff108c5e5297d43292841f9fe1ea69c5dc97a7c303f`
- training records: 21,560
- validation records: 1,219
- sealed release records: 1,186

The frozen candidate set is unchanged:

- candidate 0: 256 / 8 layers / state 16 / rank 128;
- candidate 1: 320 / 10 layers / state 24 / rank 160;
- candidate 2: 384 / 12 layers / state 24 / rank 192;
- candidate 3: 448 / 12 layers / state 32 / rank 224.

## Why the campaign is split

The original P3 runner trains all four candidates in one CUDA process. The Kaggle path
trains one candidate per session and writes one self-contained `VN97P3CAND1` shard.

Each shard:

- uses the exact sealed corpus;
- learns the tokenizer from training data only;
- uses the frozen P3 tokenizer/context/training profile;
- evaluates validation only;
- never opens or encodes `release.jsonl` for model selection;
- writes the candidate checkpoint only when that candidate passes the validation gate.

The four candidate shards can therefore be produced in four separate GPU sessions.
If one session is interrupted, only that candidate needs to be rerun.

The finalizer verifies that all four shards bind the same corpus, tokenizer, dataset
hashes and P3 profile. It then performs deterministic candidate selection and only
after the winner is fixed opens/evaluates the sealed release split.

## Input preparation

Upload the sealed `VN97-P3-Corpus.zip` artifact as a Kaggle dataset or notebook input
and extract it into a readable directory.

The extracted root must contain the canonical `corpus/` directory with:

```text
corpus.vn97corpus1.json
training.jsonl
validation.jsonl
release.jsonl
```

Clone VN97 in the notebook and install the checked-out package without replacing the
CUDA PyTorch runtime already provided by the notebook:

```bash
git clone https://github.com/dathito108-hue/VN97.git
cd VN97
python -m pip install --disable-pip-version-check --no-deps -e .
```

Before training, verify the notebook actually exposes CUDA:

```python
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
```

## Candidate sessions

Run exactly one candidate per session.

Candidate 0:

```bash
bash tools/kaggle_p3.sh candidate 0 /path/to/p3-corpus/corpus
```

Candidate 1:

```bash
bash tools/kaggle_p3.sh candidate 1 /path/to/p3-corpus/corpus
```

Candidate 2:

```bash
bash tools/kaggle_p3.sh candidate 2 /path/to/p3-corpus/corpus
```

Candidate 3:

```bash
bash tools/kaggle_p3.sh candidate 3 /path/to/p3-corpus/corpus
```

Each successful session creates:

```text
candidate-N/
  candidate-report.vn97p3cand1.json
  tokenizer.vn97tk1
  candidate.vn97ck1     # only when validation-eligible
```

and packages it as:

`/kaggle/working/VN97-P3-candidate-N.zip`

Persist/download that ZIP before ending the session.

## Finalization session

Create one writable directory containing the four extracted candidate directories:

```text
candidate-root/
  candidate-0/
  candidate-1/
  candidate-2/
  candidate-3/
```

Then run:

```bash
bash tools/kaggle_p3.sh finalize \
  /path/to/p3-corpus/corpus \
  /path/to/candidate-root
```

The finalizer:

1. verifies the exact four frozen candidate IDs/indexes;
2. verifies every VN97P3CAND1 identity;
3. verifies all tokenizer SHA-256 values are identical;
4. verifies training/validation dataset identities match across shards;
5. verifies the same sealed-release manifest hash is bound by every shard;
6. reloads every validation-eligible VN97CK1 and checks its architecture/hash;
7. deterministically selects the validation winner;
8. only then loads and evaluates `release.jsonl`;
9. applies the frozen sealed-release gate;
10. exports the exact winner to VN97MI1 and VN97P3RUN1.

Successful final output:

```text
campaign-report.json
model.vn97ck1
model.vn97mi1
p3-run.vn97p3run1.json
tokenizer.vn97tk1
SHA256SUMS
```

The launcher packages it as:

`/kaggle/working/VN97-P3-final.zip`

## Evidence rule

A Kaggle session, screenshot or notebook status is not sufficient proof of P3 success.

P3 is considered complete only after the final ZIP is retrieved and its
`VN97P3RUN1`, `VN97CAMP2`, checkpoint, tokenizer, VN97MI1 and `SHA256SUMS` are
verified.

This sharded path does not alter VN97 architecture, candidate definitions, validation
ranking or sealed-release policy. It changes only the compute scheduling boundary so
the four candidates can be trained across separate GPU sessions.
