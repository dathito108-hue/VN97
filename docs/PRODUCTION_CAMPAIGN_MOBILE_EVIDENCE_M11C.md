# M11C — Production Campaign and On-device Evidence

M11C connects the existing language campaign, M11B speech training, and the
actual Android native runtime into one evidence-gated production promotion path.

The architecture remains:

`Code 1 -> Code 2 hardware/mobile-aware -> VN97 production`

No Transformer/LLaMA/Whisper/cloud model or parallel inference backend is added.

## Production campaign finalization

M10P/VN97CAMP2 remains responsible for comparing multiple language-core
candidates and selecting one winner using the sealed language validation/release
contract.

M11C intentionally does **not** train a speech adapter for every language
candidate. That would multiply training cost without changing the underlying
language architecture.

Instead:

1. `vn97-campaign` selects one language winner.
2. `vn97-production-campaign` verifies the selected VN97CK1 and VN97TK1
   identities against the canonical VN97CAMP2 report.
3. Only that fixed winner receives one AudioFrameAdapter training run.
4. Speech train/validation/release splits must be record-disjoint.
5. Speech validation must pass before a unified checkpoint is materialized.
6. The serialized unified checkpoint is reloaded.
7. The sealed speech release split is evaluated only after the winner and speech
   adapter are fixed.
8. A sealed speech failure aborts promotion. It never falls back to another
   language candidate.

This keeps the release split an evaluator rather than a hidden selector.

## Speech corpus governance

Each loaded speech record now retains the SHA-256 of its original WAV bytes.
The record fingerprint binds:

- audio content identity;
- UTF-8 transcript.

The production campaign rejects overlap across:

- speech training / validation;
- speech training / release;
- speech validation / release.

Different filenames containing identical audio+transcript therefore do not
bypass the split guard.

The aggregate dataset SHA-256 values are also retained in the production report
and in VN97SPEECHTRAIN1 provenance.

## Unified production artifacts

After all language and speech gates pass, M11C writes:

- `model.vn97ck1` — language + speech adapter;
- `tokenizer.vn97tk1`;
- `speech-training-report.json` — M11B-compatible provenance;
- `production-campaign-report.json` — VN97PRODCAMP1.

VN97PRODCAMP1 binds:

- selected language candidate;
- source language campaign report SHA-256;
- source and unified checkpoint identities;
- tokenizer identity;
- train/validation/release speech dataset identities;
- speech validation/release results and thresholds;
- exact mobile footprint/budget;
- deployment tile geometry;
- preview speech-enabled VN97MI1 SHA-256.

The preview VN97MI1 uses the exact same exporter/tiling as final release.

## Android on-device evidence

`VN97OnDeviceEvidenceCollector` runs against the already activated native
VN97MI1. It creates fresh canonical runtime sessions and collects:

- text prefill p50/p95 latency;
- recurrent decode p50/p95 milliseconds per token;
- speech ingress/prefill p50/p95 when speech weights exist;
- process PSS high-water observation during the benchmark;
- maximum Android thermal status observed;
- device battery energy-counter delta when the platform exposes it;
- device manufacturer/model/API/ABI;
- exact model-image SHA-256.

The Android record serializes as canonical `VN97MOBEVID1` JSON.

The battery energy counter is a device battery property, not an isolated
per-process energy meter. It is retained as evidence and can be required by a
release policy when a target device supports it, but M11C does not mislabel it
as exact app-only energy consumption.

## Developer evidence capture

Non-turnkey/developer APKs expose a **Collect VN97 mobile evidence** button after
a model is READY.

The operation:

- is disabled while a turn/approval is active;
- runs on the app worker thread;
- benchmarks the activated model through the same native runtime;
- atomically publishes `vn97-mobile-evidence.json` in the app external-files
  area;
- reports the measured model SHA-256 and key p95 values in the UI.

The turnkey release build hides this developer control.

## Release evidence gate

`vn97-bootstrap-release` now supports:

- `--device-evidence <vn97-mobile-evidence.json>`;
- `--require-device-evidence`;
- minimum measured runs;
- max text prefill p95;
- max decode/token p95;
- max peak PSS;
- max thermal status;
- max speech prefill p95;
- optional required battery energy counter and delta limit.

Before the private Ed25519 signing key is opened, release:

1. builds the unsigned preview VN97MI1;
2. computes its SHA-256;
3. parses canonical VN97MOBEVID1;
4. requires the evidence model SHA-256 to match the preview exactly;
5. applies configured mobile evidence thresholds.

After signing, the bundle VN97MI1 SHA-256 is asserted to be unchanged from the
evidence-gated preview.

The release report schema is now `VN97BOOTREL4` and records the accepted
device evidence digest and observed mobile metrics.

## Honest boundary

M11C provides the machinery to generate and gate **real** device measurements.
Repository CI cannot manufacture real phone thermal/energy evidence. A
production release must run the developer evidence capture on representative
physical target devices and feed those records into the release gate.

Likewise, the repository contains the production campaign pipeline, not a claim
that a large production corpus has already been trained.

## Next slice

After a real production checkpoint and representative device evidence exist,
the next roadmap block is M12 — Perception/Action. That work should extend VN97
with camera/screen/device perception and action loops while preserving the same
single recurrent intelligence core and M6 authority boundary.
