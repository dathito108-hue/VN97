# M11B — Production Intelligence Package

M11B turns the M11A multimodal runtime contract into a reproducible production
intelligence packaging pipeline for VN97.

The architecture remains one model:

`Code 1 -> Code 2 hardware/mobile-aware -> VN97 production`

There is no speech sidecar, Transformer/LLaMA backend, cloud recognizer or
secondary inference runtime.

## 1. Unified deployment checkpoint

VN97CK1 remains the canonical non-pickle deployment checkpoint.

Language-only checkpoints remain byte-compatible with the existing VN97CK1
contract and keep header flags equal to zero.

A speech-enabled checkpoint sets optional header flag bit 0 and adds:

- `audio_adapter.projection.weight`;
- `audio_adapter.norm.weight`;
- canonical `VN97AUDIO1` metadata:
  - frame size 320;
  - hop size 320;
  - preprocessing eps 1e-5;
  - RMSNorm eps equal to the language core;
  - ternary threshold equal to the language core.

The loader reconstructs `AudioFrameAdapter`, verifies tensor names/shapes and
SHA-256 values, and rejects any audio quantization/norm configuration that does
not match the language core.

This means one VN97CK1 can now preserve the complete language + speech ingress
parameter state needed to reproduce a speech-enabled VN97MI1.

## 2. Native speech supervision

The new `vn97-speech-train` CLI consumes:

- an already trained canonical VN97CK1 language checkpoint;
- its exact VN97TK1 tokenizer;
- local JSONL speech metadata;
- local 16 kHz, mono, uncompressed PCM16 WAV files.

A speech JSONL record is exactly:

`{"audio":"relative.wav","text":"transcript"}`

Input paths are bounded to the manifest directory, regular-file reads are
bounded, the WAV format is checked, duration is limited to 20 ms..30 s, and no
network/download path exists.

Training sequence semantics exactly match the M11A native inference path:

`AUDIO control -> normalized audio frames -> TEXT control -> transcript -> EOS`

The language core is frozen during this stage. Gradients flow through that same
recurrent core into the AudioFrameAdapter, so speech alignment is learned
without silently changing the already validated language parameters.

The resulting checkpoint contains both the unchanged language core and the
trained audio adapter. `speech-training-report.json` binds:

- base checkpoint SHA-256;
- resulting checkpoint SHA-256;
- speech dataset SHA-256;
- tokenizer SHA-256;
- examples / target tokens / steps;
- mean/final loss;
- exact training bounds and optimizer settings.

## 3. Held-out speech quality gate

`vn97-bootstrap-release` now detects whether the loaded VN97CK1 contains an
audio adapter.

For a speech-enabled checkpoint it requires both:

- `--speech-training-report`;
- `--speech-validation-input`.

Before private signing material is opened, release verifies:

- the training report schema and canonical JSON;
- report checkpoint identity equals the release VN97CK1;
- report tokenizer identity equals the release VN97TK1;
- held-out speech dataset digest differs from the recorded training dataset;
- teacher-forced speech validation loss is below the configured ceiling;
- speech top-1 token accuracy is above the configured floor;
- validation example and target-token minimums are satisfied.

Language validation still runs independently. A speech release must therefore
pass both language and speech quality gates.

The aggregate training/validation digest inequality prevents accidental reuse of
the exact same dataset artifact. It is not a claim that two separately assembled
datasets cannot contain overlapping utterances; stronger corpus-level
deduplication belongs in later data-governance work.

## 4. Mobile release budget

The existing mobile footprint estimator now includes the two speech sections:

- packed VN97T2 `d_model x 320` audio projection;
- float32 `d_model` audio RMSNorm weights.

The release CLI evaluates the exact projected VN97MI1 footprint before signing
and rejects model-image or recurrent-state budget violations.

For speech releases it also records a deterministic runtime work bound:

`1 AUDIO prefix step + max audio frames + max generated transcript tokens`.

This is a bounded work contract, not a hardware latency claim. True on-device
p50/p95 latency, thermal behavior and energy measurements require execution on
representative Android devices and are intentionally reserved for the next
mobile evidence slice.

## 5. Signed turnkey artifact path

A speech-enabled release now flows as:

`language VN97CK1`
`-> vn97-speech-train`
`-> unified language+speech VN97CK1`
`-> language held-out gate + speech held-out gate + mobile budget gate`
`-> build_model_image(language + audio_adapter + tokenizer)`
`-> one speech-enabled VN97MI1`
`-> VN97CAP1 model.language/weights`
`-> VN97SIG1 Ed25519`
`-> existing M10J three-asset APK bootstrap slot`.

The Android bootstrap contract does not change. It still receives exactly:

- `model.vn97cap1`;
- `model.vn97sig1`;
- `publisher.ed25519`.

Therefore M11B adds real speech-weight packaging without adding user-installed
files or changing the final turnkey activation mechanism.

## 6. Honest production boundary

M11B provides the complete native training/checkpoint/quality/package path for
speech intelligence.

It does **not** claim that the repository already contains a large, high-quality
trained production corpus or a finished assistant checkpoint. Real capability
still depends on the data, compute, model geometry and validation thresholds used
to produce the release weights.

The next production-intelligence slice should focus on representative training
campaigns, corpus governance, and on-device quality/latency evidence rather than
adding another model architecture.
