# VN97 R2-F5 — Realtime Agent Integration

F5 keeps one VN97 R2 ONNX cognition path and separates realtime work into two
clocks:

- R2 cognition: 1-5 Hz, selected by the Android thermal/memory resource policy;
- deterministic native reflex/safety tick: approximately 30-60 Hz.

The reflex tick performs only cancellation, package authorization, foreground
and capture safety checks plus already-authorized Android gesture execution. It
never invokes the model. Multi-action remains one governed M6 action through
the existing 2-4 overlapping-stroke game multi-touch contract.

Game cognition uses VN97R2CognitionInference and the same E4-controlled F1
executor. Until a signed R2 vision ONNX graph is packaged, frame evidence is a
bounded deterministic VN97R2VISIONFEATURE1 summary derived from native VN97
vision preprocessing. It is explicitly marked semantic_vision=false. No legacy
native model is used as a hidden vision backend.

Paper trading already uses R2 cognition per market episode and deterministic
risk/execution logic; it never runs inference at market/reflex frame rate.

Physical S21 FE 30/60 Hz sustain, semantic game vision quality and thermal
performance remain F7 device-evidence gates.
