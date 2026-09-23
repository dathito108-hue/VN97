import hashlib
import struct

import pytest
import torch

from vn97 import (
    VN97Config,
    VN97LanguageCore,
    VN97TokenizerPackage,
    build_model_image,
)
from vn97.modality import AudioFrameAdapter
from vn97.model_image import (
    ENTRY_SIZE,
    FLAG_AUDIO_PROJECTION,
    FLAG_FACTORIZED,
    FLAG_TOKENIZER,
    GLOBAL_LAYER,
    HEADER_SIZE,
    MAGIC,
    SECTION_A_LOG,
    SECTION_AUDIO_NORM,
    SECTION_AUDIO_PROJECTION,
    SECTION_B_PROJ,
    SECTION_C_PROJ,
    SECTION_DT_BIAS,
    SECTION_DT_PROJ,
    SECTION_EMBEDDING,
    SECTION_EMBEDDING_PROJECTION,
    SECTION_FINAL_NORM,
    SECTION_IN_PROJ,
    SECTION_LAYER_NORM,
    SECTION_OUT_PROJ,
    SECTION_TOKEN_FACTORS,
    SECTION_TOKENIZER,
)


def _model(*, rank=None, vocab=264):
    torch.manual_seed(97)
    return VN97LanguageCore(
        VN97Config(
            vocab_size=vocab,
            d_model=12,
            n_layers=2,
            d_state=4,
            embedding_rank=rank,
        )
    ).eval()


def _entries(blob):
    count = struct.unpack_from("<I", blob, 20)[0]
    return [
        struct.unpack_from("<IIQQII", blob, HEADER_SIZE + i * ENTRY_SIZE)
        for i in range(count)
    ]


def test_full_image_is_deterministic_canonical_and_digest_bound():
    model = _model()
    first = build_model_image(model, tile_rows=4, tile_cols=4)
    second = build_model_image(model, tile_rows=4, tile_cols=4)
    assert first.data == second.data
    assert first.data[:8] == MAGIC
    assert first.model_id == hashlib.sha256(first.data).digest()
    assert len(first.model_id) == 32

    flags = struct.unpack_from("<I", first.data, 16)[0]
    assert flags == 0
    entries = _entries(first.data)
    expected = [(SECTION_EMBEDDING, GLOBAL_LAYER)]
    for layer in range(2):
        expected.extend(
            [
                (SECTION_LAYER_NORM, layer),
                (SECTION_IN_PROJ, layer),
                (SECTION_DT_PROJ, layer),
                (SECTION_DT_BIAS, layer),
                (SECTION_B_PROJ, layer),
                (SECTION_C_PROJ, layer),
                (SECTION_OUT_PROJ, layer),
                (SECTION_A_LOG, layer),
            ]
        )
    expected.append((SECTION_FINAL_NORM, GLOBAL_LAYER))
    assert [(entry[0], entry[1]) for entry in entries] == expected
    assert all(entry[4] == 0 and entry[5] == 0 for entry in entries)
    assert all(entry[2] % 4 == 0 and entry[3] > 0 for entry in entries)
    for entry in entries:
        if entry[0] in {
            SECTION_IN_PROJ,
            SECTION_DT_PROJ,
            SECTION_B_PROJ,
            SECTION_C_PROJ,
            SECTION_OUT_PROJ,
        }:
            start = entry[2]
            assert first.data[start : start + 8] == b"VN97T2\0\0"


def test_factorized_and_tokenizer_sections_preserve_single_vn97_identity():
    tokenizer = VN97TokenizerPackage()
    image = build_model_image(
        _model(rank=4),
        tokenizer=tokenizer,
        tile_rows=8,
        tile_cols=8,
    )
    flags = struct.unpack_from("<I", image.data, 16)[0]
    assert flags == FLAG_FACTORIZED | FLAG_TOKENIZER
    entries = _entries(image.data)
    assert (entries[0][0], entries[0][1]) == (SECTION_TOKEN_FACTORS, GLOBAL_LAYER)
    assert (entries[1][0], entries[1][1]) == (SECTION_EMBEDDING_PROJECTION, GLOBAL_LAYER)
    assert (entries[-1][0], entries[-1][1]) == (SECTION_TOKENIZER, GLOBAL_LAYER)
    tokenizer_offset = entries[-1][2]
    assert image.data[tokenizer_offset : tokenizer_offset + 8] == b"VN97TK1\0"


def test_audio_projection_is_signed_inside_the_same_vn97_image():
    model = _model()
    audio = AudioFrameAdapter(
        model.config.d_model,
        rms_eps=model.config.rms_eps,
    ).eval()
    image = build_model_image(
        model,
        audio_adapter=audio,
        tile_rows=8,
        tile_cols=8,
    )
    flags = struct.unpack_from("<I", image.data, 16)[0]
    assert flags == FLAG_AUDIO_PROJECTION

    entries = _entries(image.data)
    assert (entries[0][0], entries[0][1]) == (
        SECTION_EMBEDDING,
        GLOBAL_LAYER,
    )
    assert (entries[1][0], entries[1][1]) == (
        SECTION_AUDIO_PROJECTION,
        GLOBAL_LAYER,
    )
    assert (entries[2][0], entries[2][1]) == (
        SECTION_AUDIO_NORM,
        GLOBAL_LAYER,
    )
    audio_offset = entries[1][2]
    assert image.data[audio_offset : audio_offset + 8] == b"VN97T2\0\0"


def test_export_rejects_tokenizer_identity_and_nonfinite_parameters():
    with pytest.raises(ValueError, match="vocabulary"):
        build_model_image(
            _model(vocab=300),
            tokenizer=VN97TokenizerPackage(),
        )

    model = _model()
    with torch.no_grad():
        model.layers[0].core.a_log[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        build_model_image(model)


def test_export_rejects_noncanonical_tile_geometry():
    with pytest.raises(ValueError, match="tile"):
        build_model_image(_model(), tile_rows=0)
    with pytest.raises(ValueError, match="tile"):
        build_model_image(_model(), tile_cols=257)
