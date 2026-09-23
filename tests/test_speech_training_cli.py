import json
import math
import struct
import wave

import torch

from vn97 import (
    VN97Config,
    VN97LanguageCore,
    VN97TokenizerPackage,
    load_deployment_checkpoint_file,
    save_deployment_checkpoint,
)
from vn97.speech_training_cli import main


def test_speech_training_cli_produces_unified_vn97ck1(tmp_path, capsys):
    tokenizer = VN97TokenizerPackage()
    torch.manual_seed(97)
    model = VN97LanguageCore(
        VN97Config(
            vocab_size=tokenizer.vocab_size,
            d_model=8,
            n_layers=1,
            d_state=2,
        )
    ).eval()

    checkpoint = tmp_path / "base.vn97ck1"
    save_deployment_checkpoint(model, checkpoint)
    tokenizer_path = tmp_path / "tokenizer.vn97tk1"
    tokenizer_path.write_bytes(tokenizer.to_bytes())

    wav_path = tmp_path / "sample.wav"
    samples = [
        int(4000 * math.sin(index * 0.05))
        for index in range(640)
    ]
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(
            b"".join(struct.pack("<h", value) for value in samples)
        )

    manifest = tmp_path / "speech.jsonl"
    manifest.write_text(
        '{"audio":"sample.wav","text":"a"}\n',
        encoding="utf-8",
    )
    output = tmp_path / "out"

    assert main(
        [
            "--checkpoint", str(checkpoint),
            "--tokenizer", str(tokenizer_path),
            "--input", str(manifest),
            "--output-dir", str(output),
            "--max-examples", "4",
            "--epochs", "1",
            "--learning-rate", "0.001",
            "--max-frames", "8",
            "--max-target-tokens", "16",
            "--device", "cpu",
        ]
    ) == 0

    loaded = load_deployment_checkpoint_file(
        output / "model.vn97ck1"
    )
    assert loaded.audio_adapter is not None
    assert loaded.audio_adapter.projection.out_features == model.config.d_model
    assert loaded.audio_adapter.config.frame_size == 320
    assert loaded.audio_adapter.config.hop_size == 320

    report = json.loads(
        (output / "speech-training-report.json").read_text("utf-8")
    )
    printed = json.loads(capsys.readouterr().out)
    assert report == printed
    assert report["schema"] == "VN97SPEECHTRAIN1"
    assert report["examples"] == 1
    assert report["steps"] == 1
    assert report["target_tokens"] > 0
    assert len(report["dataset_sha256"]) == 64
    assert len(report["checkpoint_sha256"]) == 64
