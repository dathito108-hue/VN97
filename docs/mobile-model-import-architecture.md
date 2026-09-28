# VN97 mobile model import audit — 2026-09-29

## Architecture actually wired into the Android assistant

One canonical planner, VN97MEM1 and M6 remain unchanged. Code 1 → hardware-aware
Code 2 is the project lineage. The current text execution path uses
`VN97R2CognitionInference` → `VN97OrtProductionExecutor`. Its F2 binding matches
runtime ID, ONNX bundle ID, tokenizer model ID, vocabulary and layer geometry.
Inherited Mamba2 weight/runtime provenance is not erased by VN97 packaging.

The fast/slow adaptive-state research and trained ~10M checkpoint remain an
experimental candidate. They have not been lowered, bound and promoted to the
production Android executor. Training is paused. Selecting a research ZIP,
GGUF, G03 capsule, standalone ONNX graph or checkpoint does not convert it.

## Mismatch found

The old import UI presented CAP/SIG/public-key review as sufficient preparation
for chat. CAP1 activates an MI1 model image using the canonical inventory. It
does not install the R2 runtime, graph weights, tuning or F2 binding. Previously
runtime mismatch was discovered only when reopening the assistant, after the
inventory could already have changed.

The app now injects an R2 candidate check into the same activation backend.
Both prepare and commit inspect the MI1 candidate, load/validate the installed
ORT package and require F2 compatibility **before inventory activation**.
The existing open-time checks remain. This validates installed package metadata,
hashes and identities; it does not prove task quality, numerical parity or
successful inference on the phone. Trust/review and M6 are not bypassed.

## Revised import experience

The primary picker accepts one ZIP whose root contains exactly:

- `model.vn97cap`: existing signed capability package carrying the model image;
- `model.vn97sig`: the matching existing VN97SIG1 signature envelope;
- `publisher.ed25519`: public key (raw 32 bytes or 64 lowercase hex characters).

ZIP is transport, not a new model architecture or trust root. Its extracted
contents still pass through the canonical staging/signature/compatibility
review. A publisher key included in an archive is not automatically trusted;
the fingerprint/source/license review and explicit activation remain.
The three-file flow is retained under Advanced. The file picker no longer
hides valid signature/key files whose providers use a different MIME type.

Extraction allows only the three fixed names, rejects duplicates/paths/extras,
counts decompressed bytes, checks CRC via ZIP streaming and removes temporary
files on success/failure. The core remains bounded to the canonical 512 MiB
limit, signature to 16 KiB, public key to 256 bytes. The package is streamed;
large model bytes are not loaded as a ZIP-sized byte array.

## Remaining distribution blocker

This ZIP **does not install R2**. The existing APK-bundled R2 path is still the
supported distributor route, including its existing 768 MiB aggregate bound.
The recovered ~5.4 GB G03 capsule is not directly accepted by these paths.
Do not lift limits or relabel that capsule to claim compatibility. A complete
mobile release still needs compatible, identity-bound runtime assets,
verified payloads/parity, and the required release evidence. No available
production model package or completed AGI is established by this UI change.
