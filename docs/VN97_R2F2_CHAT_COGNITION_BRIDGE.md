# VN97 R2-F2 — Chat/Cognition Bridge

R2-F2 binds the existing VN97TK1 tokenizer and cognition contracts to the
R2-F1 ONNX Runtime executor without introducing a second inference backend.

Production identity chain:

```
validated checkpoint SHA
 -> E2 bundle ID
 -> F1 runtime ID
 -> F2 binding ID
 -> exact activated VN97TK1 artifact SHA
```

Schema `VN97R2F2BIND1` fails closed when any identity, vocabulary, geometry,
profile or runtime differs. The descriptor explicitly records
`legacy_inference_fallback=false`.

Android `VN97R2CognitionInference`:
- uses `NativeActivatedModel` only for VN97TK1 encode/decode and signed artifact
  identity;
- uses `VN97OrtProductionExecutor.prefill/step` for every model inference;
- samples only from ONNX logits;
- resets explicit recurrent state at cognition-operation boundaries;
- supports FAST/DEEP latency/budget policy over the same bound model;
- never calls `NativeRuntimeSession.inferStep/prefill/generateGreedy`.

For VN97MEM1 retrieval, F2 defines versioned `VN97R2LOGITEMB1`, a normalized
deterministic projection of the same ONNX logits into `d_model`. This avoids a
hidden native hidden-state backend. Memory migration is bound to the R2
checkpoint identity in later integration.

Build the identity descriptor only after E2/F1 exist:

```bash
vn97-r2-assistant-bind build \
  --bundle-dir /data/vn97-r2 \
  --tokenizer-model-sha256 <activated-vn97tk1-artifact-sha256>
```

No S21 FE performance claim is made by F2; provider/thermal performance remains
an E3/E4/E5 physical-device evidence gate.
