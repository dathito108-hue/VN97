# M10L — VN97 Deployment Checkpoint (VN97CK1)

M10L defines a deterministic, non-pickle deployment checkpoint for trained VN97 weights and
connects it directly to the M10K signed bootstrap bundle builder.

This is not an optimizer/training-resume file and does not change the VN97 architecture.

## Wire format

`VN97CK1` consists of:

- fixed 80-byte little-endian header;
- canonical UTF-8 JSON manifest;
- contiguous float32 tensor payload.

The header binds:

- schema version;
- manifest byte size;
- tensor count;
- payload byte size;
- total byte size;
- SHA-256 of manifest + payload;
- CRC32 of the immutable header prefix.

The manifest contains exact VN97Config fields and a lexicographically sorted tensor table. Each
tensor entry binds name, shape, float32 dtype, contiguous offset/size and SHA-256.

Limits:

- manifest <= 4 MiB;
- <= 10,000 tensors;
- checkpoint <= 2 GiB;
- tensor names <= 256 safe ASCII characters.

## Safety boundary

VN97CK1 contains only numeric configuration and raw tensor bytes. Loading does not invoke pickle,
Python object reconstruction, import hooks or package-provided code.

The loader:

1. validates header CRC and whole-content SHA-256;
2. requires canonical JSON and exact keys;
3. reconstructs `VN97Config`;
4. instantiates a fresh canonical `VN97LanguageCore`;
5. requires the checkpoint tensor-name set and shapes to match that model exactly;
6. verifies every tensor SHA-256;
7. requires tied embedding/head checkpoint values to be identical;
8. loads state strictly and verifies tied Parameter identity remains shared;
9. returns the model in eval mode.

Non-finite deployment tensors are rejected during checkpoint creation.

## File I/O

`save_deployment_checkpoint(...)` uses a same-directory temporary file, file fsync, atomic replace,
directory fsync and byte-for-byte post-write verification. Symlink source/target paths are denied.

## Direct bootstrap bridge

`build_bootstrap_bundle_from_checkpoint(...)` computes the checkpoint SHA-256 as source
provenance, loads the canonical VN97 model, requires the supplied VN97TK1 vocabulary to match, then
calls M10K:

`VN97CK1 -> VN97LanguageCore -> VN97MI1 -> VN97CAP1 -> VN97SIG1 -> M10J assets`

No alternate backend/model format is introduced.

## Remaining boundary

M10L makes trained weights portable and safely deployable, but it does not train them. A chat-ready
turnkey APK still requires an actually trained VN97CK1 checkpoint with an assistant-capable
tokenizer/model.
