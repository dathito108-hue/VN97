import hashlib
import json
import math
import struct
import wave

import pytest
import torch

from vn97 import (
    VN97Config,
    VN97LanguageCore,
    VN97SpeechEvaluationResult,
    VN97SpeechTrainingResult,
    VN97TokenizerPackage,
    build_deployment_checkpoint,
)
from vn97 import production_campaign_cli


def _write_wav(path, phase):
    samples = [
        int(4000 * math.sin(index * 0.05 + phase))
        for index in range(640)
    ]
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(
            b"".join(struct.pack("<h", value) for value in samples)
        )


def _campaign_dir(tmp_path):
    root = tmp_path / "language"
    root.mkdir()
    tokenizer = VN97TokenizerPackage()
    model = VN97LanguageCore(
        VN97Config(
            vocab_size=tokenizer.vocab_size,
            d_model=8,
            n_layers=1,
            d_state=2,
        )
    ).eval()
    checkpoint = build_deployment_checkpoint(model)
    (root / "model.vn97ck1").write_bytes(checkpoint)
    tokenizer_bytes = tokenizer.to_bytes()
    (root / "tokenizer.vn97tk1").write_bytes(tokenizer_bytes)
    report = {
        "schema": "VN97CAMP2",
        "selected_candidate_id": "0123456789abcdef",
        "selected_checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
    }
    (root / "campaign-report.json").write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return root


def _speech_manifest(tmp_path, name, wav_name, text):
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {"audio": wav_name, "text": text},
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_production_campaign_finalizes_one_speech_enabled_winner(
    tmp_path,
    monkeypatch,
):
    language = _campaign_dir(tmp_path)
    _write_wav(tmp_path / "train.wav", 0.0)
    _write_wav(tmp_path / "validation.wav", 0.3)
    _write_wav(tmp_path / "release.wav", 0.6)
    train = _speech_manifest(
        tmp_path, "train.jsonl", "train.wav", "alpha"
    )
    validation = _speech_manifest(
        tmp_path, "validation.jsonl", "validation.wav", "beta"
    )
    release = _speech_manifest(
        tmp_path, "release.jsonl", "release.wav", "gamma"
    )

    monkeypatch.setattr(
        production_campaign_cli,
        "train_vn97_speech_adapter",
        lambda *args, **kwargs: VN97SpeechTrainingResult(
            steps=1,
            examples=1,
            target_tokens=5,
            mean_loss=1.0,
            final_loss=0.9,
        ),
    )
    evaluations = iter(
        (
            VN97SpeechEvaluationResult(
                examples=1,
                target_tokens=5,
                mean_loss=1.0,
                top1_accuracy=0.2,
            ),
            VN97SpeechEvaluationResult(
                examples=1,
                target_tokens=5,
                mean_loss=1.1,
                top1_accuracy=0.2,
            ),
        )
    )
    monkeypatch.setattr(
        production_campaign_cli,
        "evaluate_vn97_speech",
        lambda *args, **kwargs: next(evaluations),
    )

    output = tmp_path / "out"
    assert production_campaign_cli.main(
        [
            "--language-campaign-dir", str(language),
            "--speech-input", str(train),
            "--speech-validation-input", str(validation),
            "--speech-release-input", str(release),
            "--output-dir", str(output),
            "--speech-max-examples", "4",
            "--speech-epochs", "1",
            "--speech-max-frames", "8",
            "--speech-max-target-tokens", "16",
            "--max-speech-validation-loss", "10",
            "--min-speech-validation-target-tokens", "1",
            "--release-max-speech-loss", "10",
            "--release-min-speech-target-tokens", "1",
            "--device", "cpu",
            "--tile-rows", "4",
            "--tile-cols", "4",
        ]
    ) == 0

    report = json.loads(
        (output / "production-campaign-report.json").read_text("utf-8")
    )
    speech_report = json.loads(
        (output / "speech-training-report.json").read_text("utf-8")
    )
    assert report["schema"] == "VN97PRODCAMP1"
    assert report["selected_candidate_id"] == "0123456789abcdef"
    assert len(report["model_image_sha256"]) == 64
    assert len(report["unified_checkpoint_sha256"]) == 64
    assert report["speech_validation"]["dataset_sha256"]
    assert report["speech_release"]["dataset_sha256"]
    assert speech_report["schema"] == "VN97SPEECHTRAIN1"
    assert (
        speech_report["checkpoint_sha256"]
        == report["unified_checkpoint_sha256"]
    )
    assert (output / "model.vn97ck1").is_file()
    assert (output / "tokenizer.vn97tk1").is_file()


def test_production_campaign_rejects_overlapping_speech_split(
    tmp_path,
):
    language = _campaign_dir(tmp_path)
    _write_wav(tmp_path / "same.wav", 0.0)
    train = _speech_manifest(
        tmp_path, "train.jsonl", "same.wav", "same"
    )
    validation = _speech_manifest(
        tmp_path, "validation.jsonl", "same.wav", "same"
    )
    _write_wav(tmp_path / "release.wav", 0.7)
    release = _speech_manifest(
        tmp_path, "release.jsonl", "release.wav", "release"
    )

    with pytest.raises(ValueError, match="speech records overlap"):
        production_campaign_cli.main(
            [
                "--language-campaign-dir", str(language),
                "--speech-input", str(train),
                "--speech-validation-input", str(validation),
                "--speech-release-input", str(release),
                "--output-dir", str(tmp_path / "out"),
                "--max-speech-validation-loss", "10",
                "--min-speech-validation-target-tokens", "1",
                "--speech-max-frames", "8",
                "--speech-max-target-tokens", "16",
                "--device", "cpu",
            ]
        )
