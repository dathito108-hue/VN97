# M14B — Strict Fresh-Frame Closed-Loop Game Agent

M14B turns the governed M14A game-control substrate into a bounded closed-loop game agent while preserving the locked VN97 architecture:

`Code 1 -> Code 2 hardware/mobile-aware -> VN97 production`

There is still one VN97 model, one canonical native cognition/planner path, VN97MEM1 and M6 authority.

No game-specific model, reinforcement-learning sidecar, Transformer/LLaMA backend, cloud controller, second planner or alternate action fabric is added.

## Core invariant

The M14B loop is:

`fresh screen frame
-> same VN97 vision perception
-> canonical VN97 plan
-> at most one game M6 action
-> fresh post-action frame
-> same VN97 visual verification
-> discard old frame plan
-> next frame plan`

The next game action is never selected from a pre-action frame.

## Why one coordinator advance is the action boundary

M14A exposed governed game actions to the normal assistant, whose generic session can advance multiple external handoffs across one logical turn.

M14B does not use that generic multi-handoff loop for gameplay.

Instead each frame builds a fresh canonical plan and calls the existing `M6EndToEndExternalCoordinator.advance()` directly.

One coordinator advance can:

- reason until one WAITING_EXTERNAL boundary;
- bind exactly one typed external request;
- issue/consume one M6 lease;
- execute exactly one capability;
- run cognition again only until the next boundary;
- return to the caller.

It cannot execute a second external request in the same call.

If the plan contains another EXTERNAL step, the returned cognition boundary is WAITING_EXTERNAL. M14B deliberately abandons that old-frame plan and captures a new post-action frame before creating the next plan.

Host regression coverage locks this property using a plan containing two consecutive EXTERNAL steps and asserts that the first coordinator advance produces exactly one side effect.

## Game-only capability surface

M14B adds a game-only M6 binder and sealed registry containing only:

- `device.game.tap`
- `device.game.swipe`
- `device.game.back`

The frame planner cannot bind app launch, clipboard or another production capability through the M14B coordinator.

The same M14A exact-package authorization, foreground-package check, AccessibilityService bridge, typed payload validation, short one-use lease and durable receipt remain authoritative.

## Episode start contract

A user must explicitly prepare all of the following:

1. a trusted activated VN97 model with production vision weights;
2. user-approved MediaProjection screen sharing;
3. user-enabled VN97 Game Control AccessibilityService;
4. a non-expired exact-package M14A game-control session;
5. a non-empty game episode goal;
6. explicit Start Game Agent action in the VN97 UI.

The foreground game-agent service then gives the user up to 30 seconds to open the exact authorized game.

It never launches the game behind the user's back.

## Foreground service

`VN97GameAgentService` is a user-started Android foreground service with a visible ongoing notification and Stop action.

It is:

- non-exported;
- special-use foreground service;
- START_NOT_STICKY;
- bounded to the current process/session;
- cancelled explicitly by the user, authorization revocation, accessibility loss, screen-share loss, target-app loss, action/time budget exhaustion or execution failure.

A game episode is not silently resurrected after process death because MediaProjection consent itself is process-bound.

## Single sovereign execution owner

Once the authorized game becomes foreground, M14B acquires the existing process-wide sovereign execution lock.

The normal foreground assistant releases its production model/memory resources before the game loop opens them.

The game loop then owns:

- the exact activated VN97 model;
- the same canonical VN97MEM1 store;
- the same native cognition adapter;
- the same M6 action/audit stack.

When the episode ends, these resources are closed and the normal assistant is reopened if it was previously active.

There is no parallel model or memory owner.

## Per-frame decision

For every decision frame M14B builds a bounded `VN97GAME2` goal containing:

- exact authorized package;
- action index;
- original user episode goal;
- fresh VN97 visual observation;
- previous post-action verification when available;
- the rule to select at most one game EXTERNAL action;
- the rule to respond with no EXTERNAL action when the episode goal is visibly complete or no safe action is needed.

The prompt is planning context for the same canonical VN97 planner, not a second policy engine.

## Fresh-frame verification

After a successful M6 game receipt:

1. M14B records an elapsed-realtime boundary after gesture completion;
2. waits for a screen frame captured strictly after that boundary;
3. runs the same activated VN97 model's production vision path on the new frame;
4. runs bounded `VN97GAMEVERIFY2` text verification over:
   - user goal;
   - action capability;
   - durable action result;
   - before observation;
   - after observation;
5. uses the post-action observation and verification as context for the next fresh plan.

The action handler itself already waits for Accessibility gesture completion before returning, so the fresh-frame timestamp boundary cannot precede gesture completion.

## Completion

If a fresh-frame canonical plan completes without producing a game action, M14B treats the planner's non-empty final response as episode completion.

If a game action was executed, M14B always captures and verifies a new frame before any next decision, even if the old-frame planner could otherwise have continued.

## Hard bounds

Current M14B production limits are:

- maximum game actions per episode: 64;
- maximum episode wall duration: 10 minutes;
- maximum reasoning advances for one frame: 4;
- maximum cognition cycles per advance: 8;
- wait for authorized game to become foreground: 30 seconds;
- foreground package poll: 100 ms;
- screen fresh-frame wait: existing bounded ScreenCaptureBroker timeout;
- game authorization duration: inherited M14A 5 minutes to 4 hours, default 2 hours.

The episode fails closed when a bound is exhausted.

## Continuous authority checks

Before every frame decision M14B rechecks:

- game authorization still exists;
- exact package has not changed;
- screen sharing is still active;
- AccessibilityService is still connected;
- exact authorized package is still foreground.

The M14A action adapter repeats exact authorization and foreground checks immediately before every actual gesture.

This creates both orchestration-time and execution-time enforcement.

## UI

The VN97 app adds:

- Start game agent;
- Stop game agent;
- current agent state;
- governed action count;
- bounded episode detail.

Start is enabled only when:

- exact game authorization exists;
- AccessibilityService is connected;
- user-approved screen sharing is active;
- no game episode is already active.

The user enters the episode goal in the existing VN97 input box and then explicitly starts the game agent.

## Notification privacy

The ongoing foreground notification does not include the user goal or visual observations.

It shows only generic game-agent state/action count and exposes a Stop action.

Terminal notification reports only the number of governed actions and terminal category.

## Regression coverage

M14B adds/extends host contracts for:

- game-only binder contains exactly tap/swipe/back;
- game-only registry is sealed;
- one coordinator advance executes at most one external side effect even when a canonical plan contains two consecutive external steps;
- after that first effect, the second step remains WAITING_EXTERNAL and therefore requires a new caller decision/frame boundary.

Full APK compilation remains the integration gate for:

- foreground service lifecycle;
- Android manifest special-use service declaration;
- screen capture broker integration;
- Accessibility foreground package integration;
- direct native VN97 vision/cognition use;
- VN97MEM1 access;
- game-only M6 coordinator creation;
- activity episode controls.

## Result

After M14B, VN97 has the first real game-agent execution loop:

`user goal
-> user-authorized game + screen
-> fresh frame
-> VN97 perceive
-> VN97 plan
-> one governed gesture
-> fresh frame
-> VN97 verify
-> repeat or finish`

The action sequence is bounded, auditable, exact-package constrained and fresh-observation driven.
