# M12A — Native Visual Perception

M12A extends the single VN97 production intelligence core with native visual
perception. It does not add a vision-language sidecar, cloud vision API,
Transformer, OCR engine or second inference backend.

The locked architecture remains:

`Code 1 -> Code 2 hardware/mobile-aware -> VN97 production`

## Canonical visual ingress

The production profile is fixed to:

- RGB, 3 channels;
- width/height <= 224;
- both dimensions divisible by 16;
- non-overlapping 16x16 patches;
- per-patch DC removal and RMS normalization;
- two explicit patch coordinates appended to each patch.

Each patch therefore has:

`3 * 16 * 16 + 2 = 770`

input features.

The native path is:

`RGB888 -> M3B vision patch preparation -> VN97T2 770->d_model projection
-> RMSNorm -> VISION control token -> same VN97 recurrent core
-> tied vocabulary logits -> native sampler -> UTF-8 perception`.

## One model image

VN97MI1 remains the only activated model image. M12A adds optional flag bit 3
and two global sections:

- type 8: packed VN97T2 visual projection;
- type 9: float32 visual RMSNorm weights.

Language-only and language+speech images remain backward compatible. Visual
perception is exposed only when the activated image contains valid signed vision
sections.

The native loader requires the canonical RGB/16x16 geometry and rejects invalid
projection shapes before exposing the model.

## Android ingress

Android runtime exposes `NativeVisionModality.prepareRgb888()`.

It accepts only bounded RGB888 buffers and calls the existing native M3B patch
preprocessor through JNI. The Kotlin layer does not reinterpret images as text
and does not call external OCR/vision services.

`NativeRuntimeSession.prefillVision()` then consumes the prepared patches using
the activated model's signed visual projection and the same recurrent session
used by text/audio.

`NativeCognitionInferenceEngine.perceiveVision()` continues generation from
that recurrent visual state using the existing native sampler.

## VN97CK1 and training

VN97CK1 remains the canonical non-pickle deployment checkpoint. Optional vision
flag bit 1 stores:

- `vision_adapter.projection.weight`;
- `vision_adapter.norm.weight`;
- VN97VISION1 metadata binding RGB channels, patch size, preprocessing epsilon,
  RMSNorm epsilon and ternary threshold.

The vision adapter threshold and normalization semantics must exactly match the
language core.

`vn97-vision-train` trains only the canonical VisionPatchAdapter against a
frozen VN97LanguageCore. Training semantics match runtime inference:

`VISION -> visual patches -> TEXT -> supervised description -> EOS`.

The CLI consumes local raw RGB888 files plus JSONL metadata. No on-device Python
runtime is introduced; this is an offline production-weight pipeline.

## Release quality gate

A vision-enabled checkpoint requires:

- VN97VISIONTRAIN1 provenance matching checkpoint + tokenizer;
- a separate held-out visual validation manifest;
- configured visual loss/accuracy/example/token gates.

The release CLI includes visual weights in the same VN97MI1, includes their
footprint in the mobile budget, and advances the signed release report to
VN97BOOTREL5.

## Authority boundary

M12A is perception only. It does not grant device-control authority.

Screen capture, camera acquisition and device actions belong to M12B and must
use explicit Android permission/consent paths plus the existing M6
deny-by-default authority/approval fabric.

This separation is intentional: seeing a screen must never implicitly grant
permission to act on it.
