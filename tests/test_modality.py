import pytest
import torch

from vn97 import (
    AudioAdapterConfig,
    AudioFrameAdapter,
    Modality,
    VN97Config,
    VN97LanguageCore,
    VisionAdapterConfig,
    VisionPatchAdapter,
    prepare_audio_frames,
    prepare_vision_patches,
    prepend_modality_identity,
)


def _core(rank: int | None = None) -> VN97LanguageCore:
    return VN97LanguageCore(
        VN97Config(
            vocab_size=320,
            d_model=24,
            n_layers=2,
            d_state=6,
            embedding_rank=rank,
        )
    )


def test_forward_embeddings_matches_token_forward_exactly():
    torch.manual_seed(31)
    for rank in (None, 8):
        model = _core(rank).eval()
        tokens = torch.randint(
            0, model.config.vocab_size, (2, 7)
        )
        with torch.no_grad():
            expected_logits, expected_states = model(tokens)
            actual_logits, actual_states = model.forward_embeddings(
                model.embedding(tokens)
            )
        torch.testing.assert_close(
            actual_logits, expected_logits, rtol=0, atol=0
        )
        for actual, expected in zip(
            actual_states, expected_states
        ):
            torch.testing.assert_close(
                actual, expected, rtol=0, atol=0
            )


def test_audio_preparation_shape_zero_mean_and_rms():
    torch.manual_seed(32)
    waveform = torch.randn(2, 40)
    frames = prepare_audio_frames(
        waveform, frame_size=16, hop_size=8, eps=1e-5
    )
    assert frames.shape == (2, 4, 16)
    torch.testing.assert_close(
        frames.mean(dim=-1),
        torch.zeros(2, 4),
        atol=2e-6,
        rtol=0,
    )
    mean_square = frames.square().mean(dim=-1)
    assert torch.all(mean_square <= 1.00001)
    assert torch.all(mean_square > 0.99)


def test_vision_preparation_order_and_coordinates():
    image = torch.arange(
        16, dtype=torch.float32
    ).reshape(1, 1, 4, 4)
    patches = prepare_vision_patches(
        image, channels=1, patch_size=2, eps=1e-5
    )
    assert patches.shape == (1, 4, 6)
    expected_coords = torch.tensor(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ]
    )
    torch.testing.assert_close(
        patches[0, :, -2:], expected_coords
    )
    torch.testing.assert_close(
        patches[..., :-2].mean(dim=-1),
        torch.zeros(1, 4),
        atol=2e-6,
        rtol=0,
    )


def test_modality_identity_is_exact_reserved_embedding():
    torch.manual_seed(33)
    for rank in (None, 8):
        model = _core(rank)
        frames = torch.randn(
            2, 3, model.config.d_model
        )
        packed = prepend_modality_identity(
            model, frames, Modality.AUDIO
        )
        expected = model.embedding(
            torch.full((2, 1), 4, dtype=torch.long)
        )
        torch.testing.assert_close(
            packed[:, :1], expected
        )
        torch.testing.assert_close(
            packed[:, 1:], frames
        )


def test_audio_and_vision_adapters_feed_same_recurrent_core():
    torch.manual_seed(34)
    model = _core(8).eval()
    audio = AudioFrameAdapter(
        model.config.d_model,
        config=AudioAdapterConfig(
            frame_size=16, hop_size=8
        ),
    ).eval()
    vision = VisionPatchAdapter(
        model.config.d_model,
        config=VisionAdapterConfig(
            channels=3, patch_size=2
        ),
    ).eval()

    waveform = torch.randn(2, 40)
    image = torch.randn(2, 3, 4, 6)
    audio_seq = prepend_modality_identity(
        model, audio(waveform), Modality.AUDIO
    )
    vision_seq = prepend_modality_identity(
        model, vision(image), Modality.VISION
    )

    with torch.no_grad():
        audio_logits, audio_states = model.forward_embeddings(
            audio_seq
        )
        vision_logits, vision_states = model.forward_embeddings(
            vision_seq
        )

    assert audio_logits.shape[:2] == audio_seq.shape[:2]
    assert vision_logits.shape[:2] == vision_seq.shape[:2]
    assert len(audio_states) == model.config.n_layers
    assert len(vision_states) == model.config.n_layers


def test_modality_sequence_matches_recurrent_embedding_steps():
    torch.manual_seed(35)
    model = _core().eval()
    adapter = AudioFrameAdapter(
        model.config.d_model,
        config=AudioAdapterConfig(
            frame_size=12, hop_size=6
        ),
    ).eval()
    sequence = prepend_modality_identity(
        model,
        adapter(torch.randn(1, 30)),
        Modality.AUDIO,
    )

    with torch.no_grad():
        full_logits, full_states = model.forward_embeddings(
            sequence
        )
        step_states = None
        outputs = []
        for index in range(sequence.shape[1]):
            logits, step_states = model.forward_embeddings(
                sequence[:, index : index + 1],
                step_states,
            )
            outputs.append(logits)
        step_logits = torch.cat(outputs, dim=1)

    torch.testing.assert_close(
        full_logits, step_logits, rtol=1e-4, atol=1e-5
    )
    assert step_states is not None
    for full, step in zip(
        full_states, step_states
    ):
        torch.testing.assert_close(
            full, step, rtol=1e-4, atol=1e-5
        )


def test_adapter_gradients_are_finite():
    torch.manual_seed(36)
    audio = AudioFrameAdapter(
        24,
        config=AudioAdapterConfig(
            frame_size=16, hop_size=8
        ),
    )
    vision = VisionPatchAdapter(
        24,
        config=VisionAdapterConfig(
            channels=3, patch_size=2
        ),
    )
    loss = (
        audio(torch.randn(2, 40)).square().mean()
        + vision(torch.randn(2, 3, 4, 4)).square().mean()
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None
        or torch.isfinite(parameter.grad).all()
        for module in (audio, vision)
        for parameter in module.parameters()
    )


def test_invalid_modality_shapes_fail_closed():
    with pytest.raises(ValueError):
        prepare_audio_frames(
            torch.randn(8),
            frame_size=4,
            hop_size=4,
            eps=1e-5,
        )
    with pytest.raises(ValueError):
        prepare_vision_patches(
            torch.randn(1, 3, 5, 4),
            channels=3,
            patch_size=2,
            eps=1e-5,
        )
