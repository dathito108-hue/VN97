# M10U — Floating Assistant Interaction

M10U turns the persistent M10T 3D overlay into an interactive VN97 assistant
surface without introducing another model, runtime, backend, or inference path.

## User interaction

When the user taps the floating 3D avatar, VN97 opens a compact focusable overlay
panel next to the avatar. The panel provides:

- native VN97 chat input and Send control;
- bounded conversation transcript for the current overlay-service lifetime;
- live READY / THINKING / WAITING_APPROVAL / ERROR presentation;
- explicit Approve and Reject controls when the canonical M6 authority gate yields
  an approval request;
- a close control that collapses the panel back to the 3D avatar;
- normal Android IME input without opening VN97MainActivity.

Dragging the avatar continues to work. If the panel is open, it follows the avatar
and is clamped to the current display bounds.

## Runtime path

The floating controller is deliberately thin. It calls the same application-level
singleton already used by VN97MainActivity:

`VN97Application.assistant -> VN97AppAssistant -> activated native VN97 model`

There is no overlay-specific assistant session implementation and no cloud or
secondary-model fallback.

The controller opens the currently activated model through
`VN97AppAssistant.openIfActivated()`. On a turnkey build, this is the same signed
model inventory already provisioned by VN97. M10U adds no user-installed model,
tokenizer, Python, Termux, or external runtime requirement.

## Authority boundary

External actions remain governed by the existing M6 approval protocol.

If a turn returns `APPROVAL_REQUIRED`, the floating panel renders the canonical
approval presentation and disables new input until the user explicitly chooses
Approve or Reject. Resolution is delegated to
`VN97AppAssistant.resolvePendingApproval()`.

A tap on the avatar is never interpreted as permission to perform an external
action.

## Android window model

M10T keeps the avatar window non-focusable so it can float unobtrusively over
other apps. M10U creates a second window only while interaction is expanded.

The interaction window:

- is `TYPE_APPLICATION_OVERLAY`;
- is focusable so the Android keyboard can target its EditText;
- is sized conservatively and clamped to display bounds;
- is removed when collapsed or when the foreground service is destroyed.

The persistent notification now tells the user to tap the avatar to interact.

## Avatar state

The floating controller feeds the existing typed avatar state with:

- SLEEPING when no trusted activated model can be opened;
- IDLE when ready;
- THINKING while a turn is running;
- WAITING_APPROVAL while an external action awaits explicit authorization;
- ERROR on an interaction failure.

This preserves the canonical path:

`Code 1 -> Code 2 hardware/mobile-aware -> VN97 production`

and:

`one activated VN97 model -> one sovereign assistant runtime -> typed avatar state
-> floating interaction surface`.

## M10V boundary

M10U intentionally does not add microphone capture, speech recognition, wake-word
logic, or speech synthesis. Those belong to M10V Voice and will bind to this same
floating interaction/runtime path rather than creating a parallel assistant.
