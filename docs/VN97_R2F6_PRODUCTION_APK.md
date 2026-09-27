# VN97 R2-F6 — Production APK Packaging

F6 extends the existing M19 turnkey release path instead of creating a second
APK builder.

A production release now requires:
- the signed VN97CAP1/VN97TK1 bootstrap;
- one validated R2 E2/F1 runtime bundle;
- one E4 target-device tuning profile;
- Android ONNX Runtime dependency already carried by the runtime module.

The release builder materializes only the F1-declared graph set into
`assets/vn97-r2/`, adds an F2 binding keyed to the release candidate's exact
VN97MI1 model-image SHA, and writes `VN97R2APK1` with SHA-256/byte identities
for every bundled runtime asset. The ONNX checkpoint SHA must equal the release
candidate checkpoint SHA.

First use is crash-safe:
1. copy APK assets to an app-private staging directory while hashing;
2. verify the index, F2 binding and activated model identity;
3. preserve the previous runtime;
4. atomically rename staging to `current`;
5. delete the previous runtime only after activation;
6. on an interrupted update, the old current runtime remains usable.

`VN97R2CognitionInference.open` then performs the existing F1/E4/device/graph
integrity verification before inference. A release APK fails closed when the R2
asset set is missing or does not match the activated model.

No GPU/T4 work is part of F6. It packages only already validated artifacts.
