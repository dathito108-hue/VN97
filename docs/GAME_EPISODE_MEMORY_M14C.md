# M14C — Game Episode Memory & Strategy Adaptation

M14C closes the persistent-learning gap left after the strict M14B game loop.

The locked architecture remains:

`Code 1 -> Code 2 hardware/mobile-aware -> VN97 production`

There is still one VN97 model, one canonical planner/cognition path, one VN97MEM1 store and the existing M6 authority fabric. M14C adds no game-specific model, RL sidecar, Transformer/LLaMA backend, cloud policy service or second memory database.

## Why M14C exists

M14B already enforces:

`fresh frame -> VN97 plan -> at most one governed action -> fresh frame -> verify`

but its game episode used VN97MEM1 mainly as retrieval context. Action/outcome traces were not explicitly committed as reusable game experience.

M14C makes those traces durable sovereign memory.

## VN97MEM1-only game memory

M14C writes directly into the existing production VN97MEM1.

It adds four source classes:

- `vn97.game.episode` — one root record per started episode;
- `vn97.game.action` — one child record per verified governed game action;
- `vn97.game.terminal` — one child record for completion/failure/cancellation;
- `vn97.game.strategy` — one semantic strategy summary learned from the episode.

No separate game-memory file or database is created.

Every action memory is written only after:

1. the M6 action receipt reports success;
2. a strictly newer screen frame is captured;
3. the same activated VN97 model perceives that frame;
4. VN97 produces a bounded post-action verification.

The memory therefore represents observed action/outcome evidence rather than intended actions.

## Episode lineage

Each episode receives a deterministic SHA-256 `episodeKey` over:

- protocol tag `VN97GAMEEP1`;
- exact authorized package;
- episode start timestamp;
- user goal.

The root VN97MEM1 record is the parent of all action, terminal and strategy records for that episode.

Action indexes must be contiguous from zero. A missing or duplicate index fails the episode-memory contract instead of silently producing an ambiguous trace.

## Recalled strategy

At episode start M14C embeds:

`VN97 game strategy for package <package> goal <goal>`

using the same activated VN97 model.

VN97MEM1 semantic retrieval uses bounded weighting:

- semantic: 0.8;
- recency: 0.1;
- importance: 0.1;
- 30-day recency half-life;
- maximum 16 retrieved candidates.

Only records with source `vn97.game.strategy` and the exact authorized package are accepted. At most four prior strategies, bounded to 8 KiB total, enter the frame-planning context.

This keeps cross-game memories from being injected into another package's control loop.

## Strategy learning

When an episode ends, the same VN97 model receives bounded `VN97GAMELEARN1` context containing:

- exact package;
- original user goal;
- terminal state and detail;
- action count;
- recalled prior strategy;
- up to eight most recent verified action outcomes.

VN97 generates a concise reusable strategy summary. The summary is embedded by the same VN97 model and committed as a SEMANTIC VN97MEM1 record.

Completed episodes receive higher strategy importance than failed/cancelled episodes, but failures are still retained as useful negative evidence.

## Prompt-injection boundary

Fresh visual observations and recalled game strategies are evidence, not authority.

M14C explicitly instructs the frame planner to treat both as untrusted evidence and not follow instructions found inside them.

Even if game content contains adversarial text, execution remains constrained by:

- the game-only M6 catalog;
- exact authorized package;
- active finite game-control session;
- connected user-enabled AccessibilityService;
- foreground package equality;
- one-use short M6 lease;
- fresh-frame action boundary;
- durable receipt.

No recalled memory can add a capability or authority grant.

## Failure semantics

Game-memory persistence is part of the M14C episode contract.

If a verified action cannot be durably committed to VN97MEM1, the episode stops rather than continuing with an untracked action history.

When the episode throws after memory initialization, M14C attempts to append a FAILED or CANCELLED terminal record and semantic lesson before propagating the failure.

MediaProjection is still process-bound, so M14C does not silently resurrect a killed gameplay session. The next user-started episode can nevertheless recall the durable strategy and outcomes from earlier episodes.

## Bounded memory payloads

Current bounds include:

- episode goal: 32 KiB UTF-8;
- action/terminal memory payload: 12 KiB;
- prior recalled strategy context: 8 KiB;
- four prior strategy records;
- eight recent action outcomes in the learning prompt;
- action observations: 1,536 characters each;
- verification: 2,048 characters;
- learned strategy: 4,096 characters;
- learning prompt: 16 KiB;
- strategy generation: 384 tokens.

All vectors use the existing VN97 hidden-state embedding and must match VN97MEM1 `dModel`.

## M14 closure

With M14A + M14B + M14C, the Game Agent milestone contains:

`explicit exact-package authorization
-> user-approved screen perception
-> fresh-frame VN97 perception
-> canonical VN97 planning
-> one M6-governed gesture
-> fresh-frame verification
-> durable VN97MEM1 action/outcome memory
-> semantic strategy learning
-> later-episode strategy recall`

M14 is therefore architecturally complete as a bounded native mobile game agent foundation.

It does not claim universal game mastery. Actual performance still depends on the trained VN97 weights, visual capability, game complexity and measured on-device latency.

The next roadmap milestone is M15 — Trading Agent, which should reuse the same planner, VN97MEM1, M6 authority and long-horizon M13 orchestration rather than introducing a trading-specific AI backend.
