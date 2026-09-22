# VN97 M3B modality adapter contract

M3B defines how non-text inputs enter the shared recurrent language core without turning raw
audio bytes or image pixels into pseudo-text tokens.

## Shared ingress

VN97LanguageCore.forward_embeddings() accepts dense [B,L,D] embeddings and uses the same
recurrent layers, state semantics, final normalization and tied vocabulary head as the normal
token-ID path.

For text:

    VN97TK1 token IDs -> tied embedding -> recurrent core

For audio/vision:

    native/reference preprocessing
      -> ternary projection to d_model
      -> RMSNorm
      -> prepend reserved modality embedding
      -> recurrent core

The prefix is the exact vocabulary embedding for the reserved control ID:

    text   = 3
    audio  = 4
    vision = 5

There is no independent modality-ID embedding table.

## Audio reference frontend

Input:

    waveform [B,S] float PCM

Each frame:

1. takes frame_size samples with hop_size stride;
2. removes the frame mean (DC component);
3. RMS-normalizes with epsilon;
4. projects frame_size -> d_model using TernaryLinear;
5. applies RMSNorm.

Default mobile profile:

    frame_size = 320
    hop_size   = 320

At 16 kHz this corresponds to 20 ms non-overlapping frames, or 50 frame embeddings per second.
The rate is configurable and this document does not claim any trained speech-recognition
capability by itself.

## Vision reference frontend

Input:

    image [B,C,H,W]

Height and width must be divisible by patch_size. Each non-overlap patch is flattened in
channel-major order, centered and RMS-normalized. Two explicit spatial features are appended:

    normalized_y in [-1,1]
    normalized_x in [-1,1]

The resulting C*P*P+2 row is projected through TernaryLinear to d_model and RMS-normalized.

Default mobile profile:

    channels   = 3
    patch_size = 16

A 224x224 image therefore produces 14x14 = 196 patch embeddings before the modality prefix.

## Native preprocessing

libvn97_modality.a provides C++17 and C-linkage functions for the deterministic preprocessing
portion:

Audio:
- AudioFrameCount
- PrepareAudioFramesF32
- vn97_audio_frame_count
- vn97_prepare_audio_frames_f32

Vision:
- VisionPatchCount
- PrepareVisionPatchesF32
- vn97_vision_patch_count
- vn97_prepare_vision_patches_f32

Python/native equivalence tests require the prepared feature rows to match within floating-point
tolerance.

The native preprocessing output is deliberately compatible with the existing VN97T2 packed
matvec projection path. M3B does not create a second weight format.

## Invariants

- raw text bytes remain lossless through VN97TK1;
- raw audio/image inputs are not silently reinterpreted as text bytes;
- modality identity is explicit before the shared recurrent core;
- audio/vision and text share the same recurrent state/update semantics after d_model ingress;
- full dense-sequence execution matches repeated one-step execution;
- preprocessing rejects invalid shapes and undersized native output buffers;
- M3B defines an adapter/runtime contract, not a claim that speech or vision intelligence is
  already trained.

## Future capability boundary

Speech recognition, speaker understanding, visual recognition and higher-level multimodal
reasoning weights can be trained or imported later through VN97 capability acquisition. Those
capabilities must conform to this stable frontend contract instead of changing the core runtime
format.
