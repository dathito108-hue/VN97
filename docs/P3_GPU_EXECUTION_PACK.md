# P3 — GPU Execution Pack

P3 is now packaged so the production-language campaign can be launched on any
user-controlled CUDA host without changing the frozen candidate set or corpus.

The execution path is:

`sealed VN97CORPUS1
-> VN97GPUENV1 preflight
-> frozen VN97 P3 campaign
-> VN97CAMP2
-> selected VN97CK1 + VN97TK1
-> VN97MI1
-> VN97P3RUN1
-> SHA256SUMS`

## Required inputs

Use the already prepared P3 corpus artifact from GitHub Actions run
`35992987070`.

Canonical corpus identity:

- VN97CORPUS1:
  `77e9142054f82b78c8ca0ff108c5e5297d43292841f9fe1ea69c5dc97a7c303f`
- artifact ID: `10805231734`
- artifact digest:
  `sha256:fe909240a1a62eb35dbe03f8ce427681bf6033c1f116b8e25f7ca4f14085e92c`

The GitHub Actions artifact expires after its configured retention period, so it
should be copied to the CUDA host before expiry.

## Host requirements

The host must provide:

- Linux shell;
- Git;
- Python >= 3.10;
- a CUDA-capable PyTorch build;
- an NVIDIA CUDA device visible to PyTorch;
- enough free disk for the corpus, editable package and model outputs.

VN97 does not auto-install a CUDA PyTorch wheel because the correct wheel depends on
the host driver/runtime. The launcher deliberately uses the already working
CUDA-enabled PyTorch environment supplied by the GPU host.

## GPU preflight

`vn97-p3-gpu-preflight` verifies:

- CUDA is visible through PyTorch;
- the requested CUDA device exists;
- exact VN97CORPUS1 manifest identity;
- train/validation/release split hashes and minimum record counts;
- current repository commit;
- P3 profile identity;
- GPU name;
- compute capability;
- total/free VRAM at launch;
- CUDA runtime exposed by PyTorch;
- PyTorch version;
- Python version;
- cuDNN version when available;
- BF16 support.

It writes canonical `VN97GPUENV1`.

Optional memory floors can be supplied with:

- `--min-total-vram-bytes`;
- `--min-free-vram-bytes`.

They default to zero because P3 does not claim an unmeasured universal minimum VRAM
requirement.

## One-command launcher

From the exact VN97 repository revision on the CUDA host:

```bash
tools/run_p3_gpu.sh /absolute/path/to/VN97-P3-Corpus/corpus
```

Optional output directory:

```bash
tools/run_p3_gpu.sh \
  /absolute/path/to/VN97-P3-Corpus/corpus \
  /absolute/path/to/p3-language-output
```

Optional controls:

```bash
export VN97_CUDA_DEVICE=cuda:0
export VN97_EXPECTED_REPOSITORY_COMMIT=<exact-40-hex-VN97-commit>
export VN97_MIN_TOTAL_VRAM_BYTES=<optional-floor>
export VN97_MIN_FREE_VRAM_BYTES=<optional-floor>
```

The launcher refuses a dirty tracked Git tree. It validates CUDA before training,
installs the checked-out VN97 package editable with `--no-deps`, runs preflight,
executes the frozen P3 campaign and seals all final hashes.

## Final output set

A successful run must contain exactly:

```text
campaign-report.json
gpu-environment.vn97gpuenv1.json
model.vn97ck1
model.vn97mi1
p3-run.vn97p3run1.json
tokenizer.vn97tk1
SHA256SUMS
```

The launcher rejects an unexpected output set.

## Remote execution from ChatGPT

If a user-authorized remote-computer connector is available, the CUDA host can be
connected to ChatGPT. Once connected, the same execution pack can be driven directly
from chat: inspect the GPU, clone/check out the exact VN97 commit, transfer or retrieve
the sealed corpus, launch the campaign, inspect progress and collect the final
VN97P3RUN1 evidence.

No GPU result is considered real until it exists on the connected host and its
VN97P3RUN1/SHA256SUMS evidence has been read back.
