# M7Y — Cold-Process Turn Memory Recovery

M7Y closes the process-death gap between M7X completed-turn orchestration and the existing M7W crash-safe VN97MEM1 transaction.

## Architecture

The production path remains one canonical VN97 path:

`NativeActivatedModel -> VN97COG1 -> planner -> M7V VN97MEM1 retrieval -> M6 authority/execution -> final response -> M7Y recovery envelope -> M7W VN97TWJ1 -> VN97MEM1`

M7Y does not add a model, planner, authority system, memory engine, or alternative assistant session.

## VN97TMR1 recovery envelope

Before a completed turn enters M7W, M7Y atomically persists a bounded `VN97TMR1` envelope containing only continuity metadata required to reconstruct the exact completed turn after process death:

- M7W turn key;
- original completion timestamp;
- bounded principal;
- canonical completed `VN97PLN1` planner checkpoint.

The envelope has a versioned header and SHA-256 payload digest. It is written by fsync + atomic replace + directory fsync and rejects symlinks, invalid sizes, invalid UTF-8 principal data, and corrupt planner checkpoints.

The actual long-term memory content remains exclusively in native `VN97MEM1`. VN97TMR1 is not a second memory database.

## Crash windows

Recovery handles the important durable states:

1. Envelope persisted, M7W PREPARED not yet written: cold start performs the normal idempotent M7W commit.
2. PREPARED written, VN97MEM1 append not yet durable: cold start resumes the expected append.
3. VN97MEM1 append durable, COMMITTED not yet written: M7W verifies the exact expected record and writes COMMITTED without appending again.
4. COMMITTED durable, envelope deletion interrupted: cold start resolves the already committed record and removes the stale envelope.

If VN97TWJ1 has an unresolved PREPARED entry but VN97TMR1 is missing, or their turn keys disagree, recovery fails closed rather than guessing the completed turn.

## Timestamp identity

The first durable VN97TMR1 envelope owns the completed-turn timestamp. Later in-process retries or cold-process recovery reuse that original timestamp rather than replacing it with the retry time.

## Production startup

`AndroidPlatformRuntime.createProductionMemoryBackedAssistant(...)` now:

1. opens the canonical VN97MEM1 store;
2. opens the M7W VN97TWJ1 writer;
3. opens/validates VN97TMR1;
4. reconciles any pending completed-turn transaction before exposing the assistant session;
5. binds the same recovery object to M7X completion/retry orchestration.

`VN97ProductionAssistantResources.recoveredTurnMemoryCommit` reports a startup recovery when one occurred.

This milestone recovers completed-turn memory durability across process death. Broader Android reboot/background scheduling and resumed nonterminal assistant work remain separate continuity work.
