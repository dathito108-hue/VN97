# M10V — Native Voice Ingress

M10V adds a self-contained Android voice-input path to the floating VN97 assistant
without introducing Google Speech, cloud STT/TTS, Whisper, Transformer/LLaMA, a
second model, Python, Termux, or an external runtime.

## Runtime path

The production ingress path is:

`Android microphone -> PCM16 mono 16 kHz -> JNI -> native M3B audio frontend
-> normalized 20 ms VN97 audio frames`.

The JNI bridge calls the existing C++ M3B functions from `vn97_modality`; raw
audio is never reinterpreted as text tokens.

The default profile remains the canonical M3B mobile profile:

- sample rate: 16 kHz;
- frame size: 320 samples;
- hop size: 320 samples;
- one native feature row per 20 ms frame;
- utterance memory bound: 30 seconds.

Captured PCM is held only in bounded process memory for the current utterance.
M10V does not persist microphone audio to disk and does not send it over the
network.

## Floating interaction

The M10U panel now includes a Mic / Stop control.

When recording:

- the foreground service promotes itself to Android's microphone foreground
  service type only for the active capture;
- the avatar receives real PCM RMS levels through the existing M8B speech
  synchronizer and displays LISTENING state;
- text input is single-flight disabled until capture finishes;
- stopping capture preprocesses the utterance through the native M3B frontend.

If Android has not granted `RECORD_AUDIO`, the overlay opens VN97 so the user can
make the one-time runtime permission decision. VN97MainActivity also exposes an
explicit microphone-permission button.

## Android lifecycle

The manifest declares:

- `RECORD_AUDIO`;
- `FOREGROUND_SERVICE_MICROPHONE`;
- the existing `FOREGROUND_SERVICE_SPECIAL_USE`;
- service types `microphone|specialUse`.

The service starts in special-use mode. It requests the microphone foreground
type only while capture is active and demotes back after capture. If Android
rejects background microphone access, VN97 fails closed and asks the user to open
the app rather than bypassing the platform restriction.

## Intelligence boundary

The currently activated VN97 model-image format contains the language model and
tokenizer but does not yet contain trained audio-projection/speech-understanding
weights. M3B explicitly defined the audio frontend without claiming trained ASR.

Therefore M10V does not fabricate speech recognition by calling an external
engine. It completes the sovereign capture/preprocessing path and makes the
missing semantic component explicit.

M11 Production Intelligence is responsible for adding trained speech capability
to the same VN97 model/capability package so the future path is:

`PCM -> M3B frontend -> VN97 audio projection -> same recurrent VN97 core
-> same planner/memory/tool runtime`.

No parallel STT model is permitted.

## Output speech boundary

M8B already provides speech timing, output-energy and viseme synchronization for
the 3D assistant. M10V does not use Android TextToSpeech because device TTS engines
or downloaded voice data would violate the final turnkey APK requirement.

Native VN97 speech synthesis/output intelligence remains part of the production
intelligence work that follows this I/O milestone.

## Architecture invariant

The locked architecture remains:

`Code 1 -> Code 2 hardware/mobile-aware -> VN97 production`

and the voice path remains part of the one sovereign model/runtime rather than a
sidecar AI backend.
