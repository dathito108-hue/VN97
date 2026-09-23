# M11A — Unified Production Intelligence Ingress

M11A is the first production-intelligence slice after M10V. It closes the architectural gap
between the existing M3B audio frontend and the canonical VN97 recurrent language core without
introducing a second AI model.

## One model, one recurrent core

The runtime now supports two equivalent ingress forms:

- token IDs -> canonical VN97 embedding table -> recurrent core;
- dense d_model embeddings -> the same recurrent core.

The dense path uses the same recurrent state, layers, final normalization, tied vocabulary head,
backend dispatch and model-identity binding as text inference. It is not a parallel backend.

Native regression coverage compares token ingress with the exact corresponding embedding ingress
and requires matching state/logits within floating-point tolerance.

## Speech weights in VN97MI1

VN97MI1 remains the single activated production model image. M11A adds one optional flag:

- bit 2: signed audio projection is present.

When set, two global sections appear immediately after the language embedding section(s):

- type 6: VN97T2 packed ternary audio projection, exactly 320 -> d_model;
- type 7: d_model float32 RMSNorm weights.

The projection therefore uses the same physical VN97T2 format and scalar/ARM64 dispatch already
used by the language core. No Whisper/Transformer/STT sidecar is embedded.

The native loader validates that:

- projection rows equal d_model;
- projection columns equal the canonical M3B frame size 320;
- normalization weights are finite;
- the adapter shares the language core rms_eps;
- all sections remain inside the SHA-256-bound VN97MI1 image.

Legacy VN97MI1 images without bit 2 remain valid language-only models.

## Canonical exporter

build_model_image() accepts an optional AudioFrameAdapter. Production export requires:

- frame_size = 320;
- hop_size = 320;
- preprocessing eps = 1e-5;
- projection geometry 320 -> d_model;
- audio RMSNorm eps equal to the core rms_eps.

The audio projection and norm are serialized into the same VN97MI1 bytes before the image is
packaged, signed, reviewed and activated by the existing M9/M10 capability pipeline.

## End-to-end native voice path

For a speech-enabled activated model the production path is now:

Android AudioRecord
-> PCM16 mono 16 kHz
-> M3B native frame normalization
-> signed VN97T2 audio projection from VN97MI1
-> RMSNorm
-> prepend reserved AUDIO token id 4
-> shared VN97 recurrent state
-> tied vocabulary logits
-> native VN97 sampler
-> UTF-8 transcription
-> canonical VN97 assistant planner/memory/tool turn

The transcription step uses a fresh cognition runtime session exactly like existing text cognition,
then feeds the resulting user text through the same VN97AppAssistant turn path. External actions
remain behind the existing M6 approval/authority boundary.

## Android behavior

The floating assistant enables Mic only when the activated model advertises a valid signed audio
projection. Language-only models remain fully usable for text and do not pretend to support
speech.

Voice capture remains explicit push-to-talk, bounded to 30 seconds, local-only and memory-only.
No Android SpeechRecognizer, Google Speech, cloud service, downloaded voice model, Python or
Termux dependency is introduced.

## Architecture invariant

The locked architecture remains:

Code 1 -> Code 2 hardware/mobile-aware -> VN97 production

M11A extends Code 2/VN97 production directly. It does not add a second inference architecture.

## Next M11 slice

M11B should harden the production-intelligence package contract and training/import path so
turnkey release assets contain trained language + speech parameters with capability/evidence
tests, followed by reasoning-quality and latency evaluation on representative mobile profiles.
