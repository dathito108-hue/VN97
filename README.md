# VN97

VN97 is a mobile-first sovereign intelligence project built around a recurrent selective
state-space core, a physically packed ternary runtime and a lossless mobile-native token
transport.

## Completed foundations

- M0: stable selective SSM reference with constant recurrent state;
- M1: physical VN97T2 2-bit ternary weights, scalar kernel, ARM64 NEON backend and dispatch;
- M2A: associative parallel affine scan for full-sequence training/reference execution;
- M2B: native fused recurrent state update/readout for prefill and token-step;
- M2C: native fused selective ZOH dynamics without expanded decay/drive tensors;
- M3A: VN97TK1 lossless tokenizer + native codec + optional factorized tied embeddings.

## M3A mobile token / embedding foundation

VN97TK1 reserves eight stable control IDs, maps every raw byte to a guaranteed token ID and
places learned multi-byte tokens after the byte range. Text is UTF-8 over this lossless byte
transport, so unsupported text can never become an unknown-token data loss problem.

The serialized tokenizer contains learned-token offsets plus a 256-way first-byte bucket index.
Native encoding therefore searches only the candidate bucket for the current byte, with
candidates stored longest-first.

Embedding compression is opt-in. The canonical legacy path remains unchanged when
embedding_rank is None. When embedding_rank R is enabled, the tied vocabulary matrix is
represented as VxR token factors plus an RxD projection; input embedding and output logits use
the same two Parameter objects.

## Native execution libraries

- libvn97_packed_ternary.a
- libvn97_recurrent.a
- libvn97_selective.a
- libvn97_tokenizer.a

Native tokenizer C-linkage entry points are provided for later Android NDK/JNI integration.

See:

- docs/ARCHITECTURE.md
- docs/PACKED_TERNARY_V1.md
- docs/ARM64_NEON_BACKEND.md
- docs/PARALLEL_SCAN.md
- docs/NATIVE_SELECTIVE_M2C.md
- docs/TOKENIZER_V1.md
- docs/EMBEDDING_COMPRESSION_M3A.md

## Local verification

    python -m pip install -e '.[dev]'
    pytest

    cmake -S native -B native/build
    cmake --build native/build
    ctest --test-dir native/build --output-on-failure

Production-only native build:

    cmake -S native -B native/build-prod -DVN97_BUILD_TESTS=OFF
    cmake --build native/build-prod
