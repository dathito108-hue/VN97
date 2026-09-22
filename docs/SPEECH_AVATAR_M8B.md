# M8B — Speech / Listening / Viseme Synchronization

M8B defines how speech timing and audio energy become avatar presentation state. It does not
choose a speech recognition or synthesis engine.

## Speech input contract

`SpeechAvatarInput` contains:

- monotonic avatar source sequence;
- presentation-only cognition state;
- normalized input/listening level;
- normalized output/speaking level;
- playback position in milliseconds;
- optional bounded viseme timeline;
- optional gesture, blink, gaze and energy.

All continuous values are finite and range checked.

`CognitionPresentationState` deliberately mirrors only display-relevant states such as
LISTENING, REASONING and RESPONDING. The host maps its real planner/cognition state into this
enum. M8B never mutates M5 state.

## Audio level path

`AudioLevelMeter.rmsPcm16()` calculates normalized RMS from caller-owned PCM16. The window must
be non-empty and at most 262,144 samples. It opens no device and requires no Android permission.

`AudioLevelEnvelope` applies separate exponential attack/release constants and bounds a single
update delta to one second. Listening and speaking envelopes are gated by the current assistant
mode before update, preventing out-of-mode audio energy from pre-charging later animation.

## Viseme timeline

A `SpeechTimeline` may contain at most 4096 cues and is capped at one hour. Each cue has:

- non-negative start time;
- 1..10,000 ms duration;
- generic VN97 viseme;
- 0..1 weight.

Cues must be sorted and non-overlapping. End-time overflow is rejected. Runtime lookup uses a
binary search and returns REST outside an active cue.

The generic viseme set is converted to three continuous mouth controls:

- jaw open;
- mouth width;
- lip round.

This avoids coupling the renderer to a specific TTS vendor's phoneme/viseme IDs.

## Synchronization

`SpeechAvatarSynchronizer` owns independent listening/speaking envelopes and serializes each
publication. For SPEAKING, the current timeline pose is multiplied by the speaking envelope and
published through the existing `AvatarStateBridge`.

The M8A monotonic source-sequence rule still applies, so stale speech frames cannot overwrite a
newer visual state.

## Renderer integration

M8B extends avatar state with:

- listeningLevel;
- jawOpen;
- mouthWide;
- lipRound.

The motion filter smooths all four continuous values. The renderer uses listening level for a
small attentive head lift and uses viseme parameters for mouth height/width/depth. Speaking level
remains a fallback jaw driver when no timeline is available.

`VN97AvatarView` exposes `speechSynchronizer` and `publishSpeech()`, which only update visual
state and request a frame.

## Authority and privacy boundary

M8B does not:

- request RECORD_AUDIO;
- capture microphone input;
- select or call an STT/TTS provider;
- create tool requests;
- approve M6 actions;
- launch apps or write clipboard;
- access network or arbitrary files.

Future speech I/O can supply PCM levels and viseme timing to M8B, but side effects remain behind
M6C/M6A and the renderer remains presentation-only.

## Verification

The avatar host regression now runs two Kotlin `-Werror` suites:

1. M8A compatibility: original state/filter/interaction/frame-policy behavior;
2. M8B speech sync: timeline validation/lookup, attack-release envelopes, PCM16 RMS bounds,
   cognition presentation mapping, listening/speaking publication, rounded/open visemes,
   stale-sequence rejection, mode-gated envelope behavior and motion-filter bounds.

Both suites run without Android SDK or a third-party speech/graphics dependency.
