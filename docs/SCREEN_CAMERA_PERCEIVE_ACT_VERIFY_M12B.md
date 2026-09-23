# M12B — Screen/Camera Acquisition + Perceive → Reason → Act → Verify

M12B connects real Android visual sources to the M12A native visual
intelligence path while keeping all external actions under the existing M6
deny-by-default authority fabric.

The canonical architecture remains:

`Code 1 -> Code 2 hardware/mobile-aware -> VN97 production`

No cloud vision API, OCR service, Transformer/LLaMA, external VLM, automation
sidecar or hidden accessibility executor is introduced.

## Screen perception

Screen perception uses Android MediaProjection only after the user accepts the
system screen-sharing consent dialog.

The runtime path is:

`MediaProjection -> 224x224 RGBA frame -> bounded RGB888
-> NativeVisionModality -> M12A native patches/projection
-> same VN97 recurrent core`.

A dedicated foreground service owns the projection token and shows a persistent
notification with a Stop action while sharing is active.

The service:

- is non-exported;
- declares `mediaProjection` foreground-service type;
- never persists raw screen frames;
- keeps only the newest prepared visual frame in process memory;
- stops immediately when Android revokes the projection token or the user stops
  the service.

A live MediaProjection session is intentionally retained long enough for a
post-action frame to be collected. This enables visual verification without a
second AI backend.

## Camera perception

Camera perception uses Camera2 and requires the normal Android CAMERA runtime
permission.

Each request captures exactly one local JPEG frame, center-crops/scales it to
224x224 RGB, converts it to the same `NativePreparedVision` contract, and
closes the camera/session immediately.

Raw camera frames are not written to disk or sent to a network service.

Camera hardware is optional; devices without a camera keep all non-camera VN97
functions.

## Perception/action coordinator

`VN97VisualActionCoordinator` owns the visual action context.

For a visual user goal:

1. M12A perceives the initial frame.
2. The observation is added as local evidence to the user's goal.
3. Visual content is explicitly treated as untrusted data, not instructions.
4. The normal VN97 assistant builds/reasons over the plan.
5. Any EXTERNAL step goes through the same typed M6 capability registry.
6. M6 requires an exact policy grant and explicit user approval where required.
7. After a successful screen-side action, M12B waits for a frame newer than the
   completed action turn.
8. M12A perceives the new frame.
9. The same native VN97 model performs `VN97VISVERIFY1` before/after
   verification.

A missing/revoked post-action frame does not erase or rewrite a successful
action receipt. The user sees verification as unavailable while the M6 action
result remains authoritative.

## Action authority

M12B does not add a wildcard device-control authority.

Production actions remain the existing typed capabilities:

- `device.clipboard.write`
- `app.launch`

Clipboard access keeps its existing exact system-clipboard scope.

For app launch, VN97 enumerates only launcher activities visible through the
declared Android launcher query and creates one exact package-scope M6 grant per
visible package, capped at 512. Every descriptor still requires explicit
approval and one-use short leases. Invalid package identifiers are skipped
rather than broadening authority.

No tap/swipe/accessibility action is added in M12B.

## UI and consent

The main app adds three explicit controls:

- Start/Stop screen perception
- Analyze shared screen
- Analyze camera

If the message field contains a goal, the visual frame becomes evidence for a
normal VN97 assistant turn. If the field is empty, VN97 performs perception
only and no external action plan is started.

Screen sharing and camera permission are never requested silently.

## Security boundary

M12B deliberately separates perception from authority:

`seeing != permission to act`

Text rendered inside a screen/camera image is treated as untrusted evidence so
visual prompt injection cannot itself grant tool authority.

The M6 request digest, exact scope digest, approval token, one-use lease and
durable action receipt remain unchanged.

## Honest boundary

M12B provides real screen/camera acquisition and the first production
perceive-reason-act-verify loop for the currently available typed Android
actions. It does not yet provide generalized touch/swipe/game-control
automation. Those higher-frequency action primitives belong to later
autonomous/game-agent milestones and must remain governed rather than bypassing
M6.
