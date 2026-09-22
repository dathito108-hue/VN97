# VN97

VN97 is a mobile-first sovereign intelligence project built around a recurrent selective
state-space core, physically packed ternary execution, a lossless native tokenizer and compact
multimodal frontends.

## Completed foundations

- M0: stable selective SSM reference with constant recurrent state;
- M1: physical VN97T2 2-bit ternary weights, scalar kernel, ARM64 NEON backend and dispatch;
- M2A: associative parallel affine scan for full-sequence training/reference execution;
- M2B: native fused recurrent state update/readout for prefill and token-step;
- M2C: native fused selective ZOH dynamics without expanded decay/drive tensors;
- M3A: VN97TK1 lossless tokenizer + native codec + optional factorized tied embeddings;
- M3B: audio/vision modality adapters + native preprocessing + dense embedding ingress.

## M3 multimodal contract

Text remains VN97TK1 token IDs. Audio and vision do not enter the recurrent core as raw
waveform/pixels. They are converted to compact frame/patch feature rows, projected with the same
ternary projection machinery used elsewhere, normalized to d_model, prefixed by the exact
reserved modality embedding, then passed through VN97LanguageCore.forward_embeddings().

Default reference profiles:

- audio: mono PCM, 320-sample frame / 320-sample hop;
- vision: RGB NCHW, non-overlap 16x16 patches;
- vision patch features include explicit normalized y/x coordinates.

The native runtime provides equivalent audio framing and vision patch extraction in
libvn97_modality.a. Projection weights can reuse VN97T2 packed matvec rather than introducing a
second modality-specific weight format.

## Native execution libraries

- libvn97_packed_ternary.a
- libvn97_recurrent.a
- libvn97_selective.a
- libvn97_tokenizer.a
- libvn97_modality.a

See:

- docs/ARCHITECTURE.md
- docs/TOKENIZER_V1.md
- docs/EMBEDDING_COMPRESSION_M3A.md
- docs/MODALITY_ADAPTERS_M3B.md
- docs/NATIVE_SELECTIVE_M2C.md

## Local verification

    python -m pip install -e '.[dev]'
    pytest

    cmake -S native -B native/build
    cmake --build native/build
    ctest --test-dir native/build --output-on-failure

Production-only native build:

    cmake -S native -B native/build-prod -DVN97_BUILD_TESTS=OFF
    cmake --build native/build-prod
