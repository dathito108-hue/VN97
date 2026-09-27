# VN97 R2-G0.3 — Zero-copy real Mamba-2 2.7B capsule

R2-G0.3 is the real-artifact bridge between the pinned Mamba-2 2.7B source
and the later VN97-native ONNX lowering.

The key design decision is **zero-copy transfer at G0**.

The inherited tensors are already the exact intelligence that G0 wants to
preserve. Re-serializing all 2.7B parameters merely to rename keys would:

- double peak storage from roughly 5.4 GB to roughly 10.8 GB;
- add another full read/write pass;
- create a second large file whose only semantic difference is namespace;
- introduce an unnecessary corruption/failure surface.

Instead G0.3 adopts the verified source payload as an immutable VN97 capsule
payload and applies the 1:1 source->VN97 namespace through a logical tensor view.

## Capsule layout

A materialized capsule contains capsule.vn97m2g03.json, source and transfer
receipts, weights/pytorch_model.bin, source/config.json, and the five pinned
GPT-NeoX tokenizer files under tokenizer/.

The weight file is hardlinked from the already verified pinned source during
materialization. Config and tokenizer files are hardlinked as well. Once the
capsule is sealed, the acquisition/source directory may be removed. The capsule
hardlinks keep the underlying inodes alive, so deleting the source path does
not delete capsule data.

The resulting directory is therefore self-contained while materialization never
stores a second 5.4 GB copy on the same filesystem.

## VN97 identity

VN97M2G03CAP1 binds the pinned Mamba-2 model-release revision, exact
5,405,424,282-byte weight identity, source receipt hash, transfer manifest hash,
source config hash, every tokenizer file name/size/hash, payload format,
logical namespace vn97.core, tensor_value_identity_1to1 semantics,
zero_copy_materialization=true, and source_runtime_required=false.

The capsule is Mamba-derived in lineage, but loading it does not require
mamba_ssm, Triton, CUDA, or a Mamba runtime backend.

## Logical tensor view

The source file keeps its original physical state-dict keys because changing
those bytes provides no intelligence benefit at G0.

VN97Mamba2TensorView exposes those exact mmap-backed tensors under the VN97
namespace. Embedding, all 64 layer tensors, final norm and tied LM head are
logically renamed without cloning tensor values.

G1/G2 training may later produce a physically VN97-native checkpoint. G0.3 is
the lossless seed artifact.

## Structural verification vs tensor loading

verify_g03_capsule_structure() validates manifest identity, sidecar hashes,
config identity, tokenizer hashes, exact weight byte size, and optionally the
full 5.4 GB weight SHA-256. It does not construct the model or load tensors.

load_g03_capsule() first runs that structural gate and then mmap-loads the
weight payload and validates the full official Mamba-2 2.7B tensor contract
before exposing the logical VN97 tensor view.

This separation lets CI test capsule logic with a sparse 5.4 GB fixture without
allocating or downloading a real 2.7B model.

## Commands

Materialize after the pinned source has been downloaded and verified:

    vn97-r2-mamba2-g03 materialize --source-root /path/to/mamba2-source --output-root /path/to/vn97-g03-capsule

Inspect structure without mmap-loading 2.7B tensors:

    vn97-r2-mamba2-g03 inspect --capsule-root /path/to/vn97-g03-capsule

Production-grade verification, including full payload SHA and state-dict
validation:

    vn97-r2-mamba2-g03 verify --capsule-root /path/to/vn97-g03-capsule

--skip-large-weight-sha256 exists only for development diagnostics.

## Real campaign

tools/r2_g03_real_capsule_campaign.sh performs the complete local/runner
sequence: fetch exact pinned source, verify source hash/tensor contract,
materialize the hardlinked VN97 capsule, confirm source/capsule weight paths are
the same inode, delete the acquisition/source directory, confirm the capsule
remains valid, and emit VN97M2G03RUN1 evidence.

The GitHub workflow .github/workflows/r2-g03-real-mamba2-capsule.yml is
workflow_dispatch only. It never runs on push. It requires explicit execution
and Apache-2.0 acknowledgements before downloading the 5.4 GB source.

If artifact upload is enabled, the workflow retains the large capsule only
three days to limit Actions storage use. A much smaller evidence bundle is
retained seven days.

## Production boundary

A real G0.3 capsule is still not permission to switch Android production
inference. Required later gates remain:

real G0.3 capsule -> real official Mamba-2 vs VN97 numerical parity -> G0.4
explicit-state ONNX lowering -> source/VN97/ONNX parity -> quantization study
only after dense correctness -> physical Galaxy S21 FE RAM/latency/thermal
qualification -> release packaging.

The dense 2.7B payload is not assumed to fit or perform acceptably on the
Galaxy S21 FE.
