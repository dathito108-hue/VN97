# VN97 R2 F2-F7 integration truth record

This branch moves production text cognition to the F1 ONNX Runtime executor without a
legacy intelligence fallback.

## Implemented code gates

- F2: VN97TK1 identity is bound to the exact R2 runtime/bundle; generation uses F1
  prefill/step logits with deterministic/top-k/top-p sampling.
- F3: assistant, canonical planner, M6 authority, VN97MEM1 retrieval/write-back and
  tool coordination are rooted in R2. ORT sessions are closed with assistant lifecycle.
- F4: existing persisted planner/continuation, reboot/process-death recovery,
  WAITING_APPROVAL restoration and no-replay M6 receipts are reused unchanged; every
  reopened cognition engine revalidates the R2 binding.
- F5: paper trading and game reasoning no longer construct the legacy cognition
  engine. Realtime cadence explicitly separates 1-4 Hz cognition from 30-60 Hz
  deterministic reflex scheduling. Game visual input currently uses deterministic
  prepared-feature summaries, not a hidden vision model.
- F6: release builds require an APK-bundled R2 runtime directory. Installation is
  app-private, bounded, crash-safe and migration-aware through staging/backup/tree ID.
- F7: static truth-gate plus Android compile/APK build run in one temporary CI workflow.

## Hard gates intentionally not claimed

1. No Samsung Galaxy S21 FE latency/thermal/RAM PASS is claimed until the APK/runtime
   is exercised on that physical device and E5 evidence is collected.
2. No 1B model or production dense-training completion is claimed without the actual
   validated checkpoint and matching R2 ONNX assets.
3. Voice transcription and semantic vision remain fail-closed until identity-bound
   R2 ONNX audio/vision graphs exist. The legacy native intelligence path is not used
   as a fallback.
4. A true turnkey release APK additionally requires the signed VN97 bootstrap,
   release signing secrets, and the matching R2 runtime assets. CI can build a debug
   APK without pretending those release-only assets exist.
