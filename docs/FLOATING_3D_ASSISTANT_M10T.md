# M10T — Floating 3D Assistant Overlay

M10T turns the existing sovereign VN97 3D avatar into an Android system overlay
that can remain visible while the VN97 Activity is closed and while the user is
inside another app.

This milestone does not create a second assistant/runtime. It reuses
`VN97AvatarView`, the existing OpenGL ES renderer and the same application
process.

## User contract

The user enables the floating assistant once from VN97.

Android requires explicit user authorization for drawing over other apps. VN97
opens the system overlay-permission screen; it never attempts to bypass this
system decision.

After permission is granted and the user enables the feature:

- a `TYPE_APPLICATION_OVERLAY` hosts the existing 3D avatar;
- the avatar is draggable and its last position is persisted;
- closing the VN97 Activity does not remove the avatar;
- a foreground service keeps the user-enabled overlay lifecycle explicit;
- a persistent notification provides a direct Hide action;
- the preference is restored after normal reboot/package replacement when the
  system permits the foreground-service start.

## Android lifecycle

The service is declared as foreground-service type `specialUse` because a
persistent user-visible assistant overlay is not covered by the standard
media/location/etc. foreground-service categories.

The manifest declares:

- `SYSTEM_ALERT_WINDOW`;
- `FOREGROUND_SERVICE`;
- `FOREGROUND_SERVICE_SPECIAL_USE`;
- `RECEIVE_BOOT_COMPLETED`.

The `specialUse` manifest property explains the exact user-visible overlay
purpose for platform/store review.

M10T returns `START_STICKY` while the feature remains enabled, but does not
claim that Android or an OEM can never terminate the process. VN97 persists the
enable flag/position and can reconstruct the overlay through allowed lifecycle
entry points.

## Rendering

`VN97AvatarView` now requests an RGBA EGL config and exposes
`enableTransparentOverlaySurface()`, which must be called before attachment.

The overlay keeps the existing adaptive avatar frame policy, so an idle avatar
does not force permanent 60 FPS rendering.

## Interaction boundary

M10T provides the persistent floating 3D presence and drag interaction.

The next integration layer can bind tap/voice/compact-chat interactions directly
to `VN97AppAssistant` so normal conversations and task control can happen from
the overlay without opening `VN97MainActivity`.

No overlay gesture is an M6 authority grant. External actions continue to use
the existing explicit approval/authority path.

## Architecture boundary

M10T adds no accessibility-control bypass, hidden input injection, alternate AI
backend, cloud inference path or secondary model.

The path remains:

`one activated VN97 model -> one sovereign assistant runtime -> typed avatar
state -> floating VN97AvatarView`.
