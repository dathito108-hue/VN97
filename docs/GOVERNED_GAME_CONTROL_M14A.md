# M14A — Governed Game Control Foundation

M14A begins the Game Agent milestone by extending the existing VN97 production action path with governed Android game-control primitives.

It does not introduce a game-specific model, planner, reinforcement-learning sidecar, Transformer/LLaMA backend, cloud controller or automation daemon.

The locked architecture remains:

`Code 1 -> Code 2 hardware/mobile-aware -> VN97 production`

with one VN97 model, one canonical planner, VN97MEM1 and M6 authority.

## Scope of M14A

M14A provides the controlled action substrate required by a later closed-loop game agent:

- exact-package game-control authorization;
- Android AccessibilityService gesture execution;
- typed M6 tap, swipe and back capabilities;
- normalized resolution-independent coordinates;
- capability payload schemas exposed to the canonical VN97 cognition adapter;
- foreground-package enforcement at execution time;
- short one-use M6 leases and durable M6 audit receipts;
- explicit user enable/revoke UI.

M14A is intentionally not yet the full continuous game-playing loop. A later M14 block must bind each game action to a fresh post-action frame before choosing the next action.

## Explicit user game session

Game controls are unavailable by default.

The user must:

1. explicitly enable the VN97 Game Control AccessibilityService in Android settings;
2. open the target game so VN97 observes its package as an external foreground app;
3. return to VN97 and authorize that exact package;
4. keep the session within a finite duration.

The durable local authorization is:

`VN97GameControlSession(packageName, authorizedAt, expiresAt)`

Bounds:

- minimum session: 5 minutes;
- default session: 2 hours;
- maximum session: 4 hours;
- one exact Android package per session.

The user can revoke the session at any time.

The action adapter checks the durable authorization again at execution time. Therefore revocation or expiry takes effect even if an already-open assistant session still contains older in-memory M6 grants.

## Accessibility boundary

The AccessibilityService is used only as a gesture execution bridge.

Its XML configuration:

- permits gesture dispatch;
- listens only for window-state/window-list changes needed to identify the foreground package;
- does not request window-content retrieval;
- is protected by Android `BIND_ACCESSIBILITY_SERVICE`;
- requires the user to enable the service through Android system UI.

M14A does not use Accessibility to scrape UI text, inspect view trees, capture passwords or bypass Android screen-capture consent.

Visual perception remains the existing M12B path:

`user-approved MediaProjection
-> bounded RGB frame
-> native VN97 vision adapter
-> same VN97 recurrent core`

## Exact foreground-package enforcement

Every game action carries scope:

`{"package":"com.example.game"}`

Immediately before dispatch, VN97 requires:

- an active non-expired user game session;
- session package equals the action scope package;
- AccessibilityService is connected;
- current foreground package equals the exact scoped package.

A stale plan cannot tap another app after the user switches away from the game.

## Typed M6 game capabilities

M14A adds:

- `device.game.tap`
- `device.game.swipe`
- `device.game.back`

All three use the existing:

`canonical planner
-> EXTERNAL step
-> NativeTypedCognitionAdapter
-> M6ExternalIntentBinder
-> M6 deny-by-default authority
-> one-use lease
-> typed handler
-> durable action receipt`

No alternate action path exists.

### Tap

Scope:

`package`

Canonical payload schema:

`{"duration_ms":80,"x_bps":5000,"y_bps":5000}`

Bounds:

- x/y: 0..10000 basis points;
- duration: 20..1500 ms.

### Swipe

Scope:

`package`

Canonical payload schema:

`{"duration_ms":300,"end_x_bps":8000,"end_y_bps":5000,"start_x_bps":2000,"start_y_bps":5000}`

Bounds:

- all coordinates: 0..10000 basis points;
- duration: 50..3000 ms;
- start and end must differ.

### Back

Scope:

`package`

Payload:

`{}`

It maps to Android's global Back action only after exact-package/session validation.

## Resolution-independent coordinate contract

Game coordinates are represented in basis points instead of raw pixels.

`0 = left/top`

`10000 = right/bottom`

The Accessibility bridge maps these values to the current display dimensions immediately before gesture dispatch.

This keeps the VN97 action representation independent of individual phone resolutions while preserving bounded deterministic validation.

## Capability payload schemas

M14A extends the existing `NativeExternalCapabilityView` and `M6CapabilityDescriptor` with a bounded canonical `payloadSchemaJson`.

The same `NativeTypedCognitionAdapter.proposeExternalIntent()` now receives, for every allowed capability:

- capability ID;
- required/optional scope keys;
- approval requirement;
- maximum payload bytes;
- canonical payload schema.

This is part of the same VN97 cognition protocol. It is not a tool-routing model or second planner.

The game schemas therefore tell VN97 exactly which canonical JSON fields it must emit rather than relying on prompt guesswork.

## Authority model

Outside an active game session, no M6 game grants are produced.

During an active exact-package session, M14A generates grants only for that package and only for the three game capabilities.

Each action still receives:

- an exact scope digest;
- a maximum 10-second lease;
- a one-use lease budget;
- a durable M6 execution receipt.

The game-session authorization is the user's explicit bounded consent for repeated game primitives within the selected package, so the per-action grant does not require another approval prompt.

This does not weaken existing approval behavior for `app.launch` or `device.clipboard.write`.

## Existing M12B visual verification

The current screen visual action path already supports:

`screen frame
-> same-model perception
-> canonical reasoning/planning
-> M6 external execution
-> fresh post-action screen frame
-> same-model visual verification`

M14A makes game gestures available to that governed action fabric.

However, the generic foreground assistant can advance more than one external handoff inside a turn. Therefore M14A does not claim that repeated gameplay is already a strict one-action/one-frame closed loop.

The next Game Agent block should enforce:

`fresh frame
-> one canonical decision
-> at most one game action
-> fresh frame
-> outcome verification
-> next decision`

with bounded episode/action budgets and recovery.

## Regression coverage

M14A extends production-capability host coverage to verify:

- the three game capabilities exist in the sealed registry;
- game descriptors are one-use and short-lived;
- exact-package grants are generated only for the requested package;
- valid tap/swipe/back requests execute through M6;
- out-of-range tap coordinates fail validation;
- zero-distance swipe fails validation;
- capability payload schemas are propagated into the canonical VN97 external-intent request.

Full Android APK compilation remains the integration gate for:

- AccessibilityService APIs;
- manifest/resource wiring;
- SharedPreferences game-session policy;
- app/runtime grant integration;
- Activity consent controls.

## Result

After M14A, VN97 has a governed native Android game-control substrate:

`user consent
-> exact package session
-> user-approved screen perception
-> VN97 perception/reasoning
-> typed game intent
-> M6 exact-scope lease
-> Accessibility gesture
-> durable audit`

The architecture remains one VN97 production model/planner/memory/authority path.
