# P3 — Production Language Campaign

P3 is the first bounded multi-candidate GPU campaign after the successful real P2
pipeline proof.

The canonical P3 corpus artifact was prepared by GitHub Actions run
`35992987070`.

Corpus identity:

- VN97CORPUS1:
  `77e9142054f82b78c8ca0ff108c5e5297d43292841f9fe1ea69c5dc97a7c303f`
- training records: 21,560
- validation records: 1,219
- sealed release records: 1,186
- training split SHA-256:
  `90c9624bdd4d0f5ce22e634ce3948799f68f1e65559742b4ab2066c55cc7cf56`
- validation split SHA-256:
  `f867e592803dcd935dd4d251f3a81039ae8e0c35dbdcf5464d8aa8c4293ce901`
- release split SHA-256:
  `5468ac3e108d9e88d3ef6a1603dd584c5d205b517dd9bea4be570c934cd9e22f`
- GitHub artifact ID: `10805231734`
- artifact digest:
  `sha256:fe909240a1a62eb35dbe03f8ce427681bf6033c1f116b8e25f7ca4f14085e92c`

The artifact SHA256SUMS file was independently rechecked after download and all
contained files matched.

## Candidate freeze

The candidate set is:

`configs/p3-language-production.vn97campdef1.json`

It contains four candidates, all using the same canonical VN97 architecture:

1. d_model 256, 8 layers, d_state 16, rank 128;
2. d_model 320, 10 layers, d_state 24, rank 160;
3. d_model 384, 12 layers, d_state 24, rank 192;
4. d_model 448, 12 layers, d_state 32, rank 224.

Candidate identity remains canonical SHA-256 over candidate JSON. There is no manual
winner override.

## Frozen P3 profile

The canonical runner is:

`vn97-p3-language-campaign`

Training profile:

- tokenizer training sample: deterministic SHA-256-ranked 2,048 records from the sealed training split only;
- tokenizer learned-token ceiling: 4,096;
- sequence length: 512;
- batch size: 8;
- epochs: 2;
- train window safety ceiling: 100,000;
- validation/release window ceilings: 10,000 each;
- parameter ceiling: 50,000,000;
- VN97MI1 ceiling: 128 MiB;
- recurrent-state ceiling: 8 MiB;
- VN97T2 deployment tiling: 16 x 16.

The large window values are safety ceilings, not requested truncation counts. The
existing canonical campaign materializes all model-training windows and fails if the
configured bound is exceeded. The tokenizer sample bound applies only to tokenizer
learning; all 21,560 sealed training records remain available for model training.

## Admission quality gate

P3 uses P2 only as a non-regression reference. It does not claim these thresholds are
the final intelligence standard.

Validation admission:

- mean loss <= 6.76;
- token top-1 >= 0.052;
- target tokens >= 512.

Sealed release admission for the validation-selected winner:

- mean loss <= 6.79;
- token top-1 >= 0.039;
- target tokens >= 512.

The P2 reference run is:

`e0509769f3f819516c7e41c28769053673f634e41051d8e469b95bd8421db5a2`

P4 remains responsible for actual assistant-task quality: instruction following,
reasoning/planning, memory use, structured cognition, tool intent and authority
behavior.

## GPU boundary

The P3 runner deliberately refuses CPU and `auto`.

It requires an explicit CUDA device and verifies `torch.cuda.is_available()` before
starting the multi-candidate campaign. This prevents accidental consumption of large
CPU/GitHub Actions budgets.

Example on a rented GPU host:

```bash
vn97-p3-language-campaign \
  --corpus-dir VN97-P3-Corpus/corpus \
  --output-dir p3-language-output \
  --device cuda
```

The runner validates the exact VN97CORPUS1 manifest and minimum split sizes before
training.

## Output

A successful campaign produces:

- `tokenizer.vn97tk1`;
- `model.vn97ck1`;
- `campaign-report.json` / VN97CAMP2;
- `model.vn97mi1`;
- `p3-run.vn97p3run1.json` / VN97P3RUN1.

VN97P3RUN1 binds:

- corpus manifest ID and SHA-256;
- campaign report SHA-256;
- selected candidate ID;
- checkpoint SHA-256;
- tokenizer SHA-256;
- VN97MI1 SHA-256 and byte count;
- validation and sealed-release loss/top-1 measurements;
- CUDA device;
- immutable P3 profile SHA-256;
- P2 baseline run identity.

Only a successful VN97P3RUN1 proceeds to P4 task-level intelligence evaluation.
