import hashlib
import json
import math
import struct
import wave

import pytest
import torch

from vn97 import (
    AudioAdapterConfig,
    AudioFrameAdapter,
    VN97Config,
    VN97ReleaseQualityError,
    VN97LanguageCore,
    VN97TokenizerPackage,
    save_deployment_checkpoint,
)
from vn97.bootstrap_release_cli import main


cryptography = pytest.importorskip(
    "cryptography.hazmat.primitives.asymmetric.ed25519"
)
serialization = pytest.importorskip(
    "cryptography.hazmat.primitives.serialization"
)


def _checkpoint_and_tokenizer(tmp_path):
    tokenizer = VN97TokenizerPackage((b" the", b"ing", b"VN97"))
    torch.manual_seed(97)
    model = VN97LanguageCore(
        VN97Config(
            vocab_size=tokenizer.vocab_size,
            d_model=12,
            n_layers=2,
            d_state=4,
        )
    ).eval()

    checkpoint = tmp_path / "model.vn97ck1"
    save_deployment_checkpoint(model, checkpoint)
    tokenizer_path = tmp_path / "tokenizer.vn97tk1"
    tokenizer_path.write_bytes(tokenizer.to_bytes())
    validation = tmp_path / "validation.jsonl"
    validation.write_text(
        '{"text":"VN97 validation sample for release quality."}\n',
        encoding="utf-8",
    )
    return checkpoint, tokenizer_path, validation


def test_release_cli_writes_exact_m10j_assets(tmp_path, capsys):
    checkpoint, tokenizer, validation = _checkpoint_and_tokenizer(tmp_path)
    private = cryptography.Ed25519PrivateKey.generate()
    raw_private = private.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    private_path = tmp_path / "publisher.private"
    private_path.write_bytes(raw_private)

    assets = tmp_path / "assets"
    assert main(
        [
            "--checkpoint", str(checkpoint),
            "--tokenizer", str(tokenizer),
            "--private-key", str(private_path),
            "--key-id", "publisher.main",
            "--capability-version", "1",
            "--source-origin", "vn97-training",
            "--source-license", "proprietary",
            "--assets-dir", str(assets),
            "--validation-input", str(validation),
            "--validation-format", "text",
            "--validation-sequence-length", "32",
            "--validation-batch-size", "1",
            "--min-validation-target-tokens", "1",
            "--max-validation-loss", "100",
            "--tile-rows", "4",
            "--tile-cols", "4",
        ]
    ) == 0

    assert {path.name for path in assets.iterdir()} == {
        "model.vn97cap1",
        "model.vn97sig1",
        "publisher.ed25519",
    }
    assert (assets / "publisher.ed25519").read_bytes() == (
        private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    )
    assert all(
        path.read_bytes() != raw_private
        for path in assets.iterdir()
    )

    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == "VN97BOOTREL4"
    assert report["device_evidence"] is None
    assert report["speech_enabled"] is False
    assert report["speech_validation"] is None
    assert report["publisher_key_id"] == "publisher.main"
    assert len(report["checkpoint_sha256"]) == 64
    assert len(report["package_sha256"]) == 64
    assert report["validation"]["target_tokens"] > 0
    assert report["validation"]["mean_loss"] <= 100
    assert len(report["validation"]["dataset_sha256"]) == 64


def test_release_cli_requires_and_reports_speech_quality_for_speech_checkpoint(
    tmp_path,
    capsys,
):
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
    adapter = AudioFrameAdapter(
        model.config.d_model,
        ternary_threshold=model.config.ternary_threshold,
        config=AudioAdapterConfig(),
        rms_eps=model.config.rms_eps,
    ).eval()

    checkpoint = tmp_path / "speech-model.vn97ck1"
    save_deployment_checkpoint(
        model,
        checkpoint,
        audio_adapter=adapter,
    )
    tokenizer_path = tmp_path / "tokenizer.vn97tk1"
    tokenizer_path.write_bytes(tokenizer.to_bytes())
    validation = tmp_path / "validation.jsonl"
    validation.write_text('{"text":"language validation"}\n', encoding="utf-8")

    wav_path = tmp_path / "sample.wav"
    samples = [
        int(5000 * math.sin(index * 0.08))
        for index in range(640)
    ]
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(
            b"".join(struct.pack("<h", value) for value in samples)
        )
    speech_validation = tmp_path / "speech-validation.jsonl"
    speech_validation.write_text(
        '{"audio":"sample.wav","text":"a"}\n',
        encoding="utf-8",
    )

    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    tokenizer_sha256 = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    speech_training_report = tmp_path / "speech-training-report.json"
    speech_training_report.write_text(
        json.dumps(
            {
                "base_checkpoint_sha256": "0" * 64,
                "checkpoint_sha256": checkpoint_sha256,
                "dataset_sha256": "1" * 64,
                "examples": 1,
                "final_loss": 1.0,
                "mean_loss": 1.0,
                "schema": "VN97SPEECHTRAIN1",
                "steps": 1,
                "target_tokens": 3,
                "tokenizer_sha256": tokenizer_sha256,
                "training": {
                    "epochs": 1,
                    "learning_rate": 0.001,
                    "max_frames": 8,
                    "max_grad_norm": 1.0,
                    "max_target_tokens": 16,
                    "seed": 97,
                    "weight_decay": 0.01,
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    private = cryptography.Ed25519PrivateKey.generate()
    raw_private = private.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    private_path = tmp_path / "publisher.private"
    private_path.write_bytes(raw_private)

    assets = tmp_path / "speech-assets"
    assert main(
        [
            "--checkpoint", str(checkpoint),
            "--tokenizer", str(tokenizer_path),
            "--private-key", str(private_path),
            "--key-id", "publisher.main",
            "--capability-version", "3",
            "--source-origin", "vn97-speech-training",
            "--source-license", "proprietary",
            "--assets-dir", str(assets),
            "--validation-input", str(validation),
            "--validation-format", "text",
            "--validation-sequence-length", "16",
            "--validation-batch-size", "1",
            "--min-validation-target-tokens", "1",
            "--max-validation-loss", "100",
            "--speech-training-report", str(speech_training_report),
            "--speech-validation-input", str(speech_validation),
            "--speech-validation-max-examples", "4",
            "--speech-max-frames", "8",
            "--speech-max-target-tokens", "16",
            "--min-speech-validation-target-tokens", "1",
            "--min-speech-validation-examples", "1",
            "--max-speech-validation-loss", "100",
            "--tile-rows", "4",
            "--tile-cols", "4",
        ]
    ) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == "VN97BOOTREL3"
    assert report["speech_enabled"] is True
    assert report["speech_validation"]["examples"] == 1
    assert report["speech_validation"]["target_tokens"] > 0
    assert len(report["speech_validation"]["dataset_sha256"]) == 64

    package = (assets / "model.vn97cap1").read_bytes()
    # The signed capability contains the speech-enabled VN97MI1 as its only data section.
    assert b"VN97MI1\0" in package


def test_release_cli_accepts_lowercase_hex_private_key(tmp_path):
    checkpoint, tokenizer, validation = _checkpoint_and_tokenizer(tmp_path)
    raw_private = bytes(range(32))
    private_path = tmp_path / "publisher.hex"
    private_path.write_text(raw_private.hex(), encoding="ascii")

    assets = tmp_path / "assets"
    assert main(
        [
            "--checkpoint", str(checkpoint),
            "--tokenizer", str(tokenizer),
            "--private-key", str(private_path),
            "--key-id", "publisher.main",
            "--capability-version", "2",
            "--source-origin", "vn97-training",
            "--source-license", "proprietary",
            "--assets-dir", str(assets),
            "--validation-input", str(validation),
            "--validation-format", "text",
            "--validation-sequence-length", "32",
            "--validation-batch-size", "1",
            "--min-validation-target-tokens", "1",
            "--max-validation-loss", "100",
        ]
    ) == 0
    assert (assets / "model.vn97cap1").is_file()


def test_release_cli_rejects_private_key_symlink(tmp_path):
    checkpoint, tokenizer, validation = _checkpoint_and_tokenizer(tmp_path)
    key = tmp_path / "real-key"
    key.write_bytes(bytes(range(32)))
    link = tmp_path / "key-link"
    link.symlink_to(key)

    with pytest.raises(ValueError, match="opened safely"):
        main(
            [
                "--checkpoint", str(checkpoint),
                "--tokenizer", str(tokenizer),
                "--private-key", str(link),
                "--key-id", "publisher.main",
                "--capability-version", "1",
                "--source-origin", "vn97-training",
                "--source-license", "proprietary",
                "--assets-dir", str(tmp_path / "assets"),
                "--validation-input", str(validation),
                "--validation-format", "text",
                "--validation-sequence-length", "32",
                "--validation-batch-size", "1",
                "--min-validation-target-tokens", "1",
                "--max-validation-loss", "100",
            ]
        )


def test_release_quality_gate_runs_before_private_key_read(tmp_path):
    checkpoint, tokenizer, validation = _checkpoint_and_tokenizer(tmp_path)
    assets = tmp_path / "assets"

    with pytest.raises(VN97ReleaseQualityError, match="loss"):
        main(
            [
                "--checkpoint", str(checkpoint),
                "--tokenizer", str(tokenizer),
                "--private-key", str(tmp_path / "missing-private-key"),
                "--key-id", "publisher.main",
                "--capability-version", "1",
                "--source-origin", "vn97-training",
                "--source-license", "proprietary",
                "--assets-dir", str(assets),
                "--validation-input", str(validation),
                "--validation-format", "text",
                "--validation-sequence-length", "32",
                "--validation-batch-size", "1",
                "--min-validation-target-tokens", "1",
                "--max-validation-loss", "0.000001",
            ]
        )
    assert not assets.exists()
