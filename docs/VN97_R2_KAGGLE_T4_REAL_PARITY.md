# VN97 R2-K1 — Kaggle T4 real Mamba-2 → VN97 parity

This campaign uses Kaggle T4 only after the real G0.3 1:1 transfer has already
been proven.

Canonical real-transfer evidence:

- source: `state-spaces/mamba2-2.7b`
- source revision:
  `99b226cc377d131cccc610ed4346db564f381f1e`
- source weight SHA-256:
  `254d89bfef6dd9f44c8e3f54e0730af3f10c482969117cb37fb33486348b00be`
- unique core parameters: 2,702,599,680
- real G0.3 capsule:
  `8e1aff6c6b6d2947cb25111e48febae6850100fd782a2f112adcab2e047f674e`

K1 verifies numerical execution, not just file identity.

## Kaggle settings

Create a Kaggle notebook with:

- Accelerator: **GPU T4**
- Internet: **On**
- persistence/output enabled if you want to retain the final ZIP

The runner intentionally uses only `cuda:0` even if Kaggle exposes two T4s.
This keeps source and VN97 execution on one deterministic device for parity.

## One-cell start

Run this in a Kaggle notebook cell:

    !git clone https://github.com/dathito108-hue/VN97.git /kaggle/working/VN97
    %cd /kaggle/working/VN97
    !bash tools/kaggle_r2_k1_real_parity.sh

The launcher:

1. verifies CUDA and requires an NVIDIA T4;
2. requires at least 16 GiB free under `/kaggle/working`;
3. installs only the small Python dependencies needed by the oracle;
4. clones `state-spaces/mamba` and checks out the pinned oracle commit
   `e9594ce1c732d97440f0332fdc43170a2294dbfa`;
5. downloads the exact pinned 5.4 GB Mamba-2 source and GPT-NeoX tokenizer;
6. verifies the source SHA-256 and G0.3 transfer evidence;
7. loads the official 2.7B model once in FP16 on T4;
8. executes official Mamba-2 recurrent steps and VN97-native recurrent steps
   against the same inherited weights;
9. compares logits, layer-0 hidden state, convolution state and SSD state;
10. requires greedy generated token sequences to remain exact;
11. writes a content-addressed `VN97M2K1PARITY1` receipt.

## Default probe profile

Default workload:

- 8 fixed text probes;
- at most 12 prompt tokens per probe;
- 4 greedy continuation tokens per probe;
- FP16 on T4;
- maximum absolute error thresholds:
  - logits: 0.005
  - hidden state: 0.005
  - recurrent states: 0.005

The thresholds are intentionally explicit and fail closed.

To tighten them later:

    %env VN97_K1_MAX_LOGIT_ERROR=0.002
    %env VN97_K1_MAX_HIDDEN_ERROR=0.002
    %env VN97_K1_MAX_STATE_ERROR=0.002
    !bash tools/kaggle_r2_k1_real_parity.sh

To extend probe length:

    %env VN97_K1_MAX_PROMPT_TOKENS=24
    %env VN97_K1_GENERATION_TOKENS=8
    !bash tools/kaggle_r2_k1_real_parity.sh

## Outputs

Successful K1 creates:

    /kaggle/working/vn97-r2-k1/vn97-k1-real-parity.json
    /kaggle/working/vn97-r2-k1/K1_SHA256SUMS
    /kaggle/working/VN97-R2-K1-real-parity.zip

The JSON binds:

- real G0.3 capsule ID;
- source checkpoint revision/hash;
- pinned official Mamba implementation commit;
- GPU name;
- Torch/CUDA runtime;
- probe counts;
- numerical error metrics;
- generated-token equality;
- PASS/FAIL status.

## What K1 proves

If K1 passes, we can say the real transferred Mamba-2 core is executing through
the VN97-native recurrent equations within the locked numerical tolerance and
produces the same greedy continuation tokens on the probe suite.

K1 does **not** yet prove ONNX Runtime parity.

The next Kaggle stage is K2:

    VN97 native
      -> real G0.5 ONNX export
      -> ONNX Runtime
      -> logits/state parity

Only after K1 and K2 both pass should G0.10 model-parity evidence be promoted.

## T4 quota usage

The 5.4 GB download/hash work is CPU/network-bound, but it is performed in the
same Kaggle session so the exact source is immediately available for T4 parity.

Do not start G1 augmentation training until K1 has passed. Preserving the
pretrained intelligence baseline has priority over adding new VN97 modules.

## K1D kernel diagnosis

If the default optimized run reports `generation_exact=true` but numerical
thresholds fail, do not relax the tolerances. First rerun the same exact model
and probes with the official Mamba-2 optional recurrent CUDA update kernels
disabled so both sides use the unfused fallback recurrence:

    %env VN97_K1_OFFICIAL_KERNEL_MODE=fallback
    !bash tools/kaggle_r2_k1_real_parity.sh

The source weights, tokenizer, probe set and thresholds remain unchanged.

Interpretation:

- fallback PASS + optimized FAIL: the architecture/equations are aligned and the
  observed gap is backend numerical drift from the optimized FP16 kernels;
- fallback FAIL: treat it as a real implementation-parity defect and diagnose the
  first divergent operation before K2 or G1 training.

This diagnostic is intentionally preferred over simply widening the error gate.
