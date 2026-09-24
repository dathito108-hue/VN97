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
    VN97ReleaseCandidateDeviceEvidence,
    VN97ReleaseCandidateManifest,
    VN97TokenizerPackage,
    build_model_image,
    load_deployment_checkpoint_file,
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
    assert report["schema"] == "VN97BOOTREL6"
    assert report["device_evidence"] is None
    assert report["speech_enabled"] is False
    assert report["speech_validation"] is None
    assert report["publisher_key_id"] == "publisher.main"
    assert len(report["checkpoint_sha256"]) == 64
    assert len(report["package_sha256"]) == 64
    assert report["validation"]["target_tokens"] > 0
    assert report["validation"]["mean_loss"] <= 100
    assert len(report["validation"]["dataset_sha256"]) == 64


def test_release_cli_can_require_matching_device_evidence(
    tmp_path,
    capsys,
):
    checkpoint, tokenizer_path, validation = _checkpoint_and_tokenizer(tmp_path)
    loaded = load_deployment_checkpoint_file(checkpoint)
    tokenizer = VN97TokenizerPackage.from_bytes(tokenizer_path.read_bytes())
    preview = build_model_image(
        loaded.model,
        tokenizer=tokenizer,
        tile_rows=4,
        tile_cols=4,
    )
    model_sha = hashlib.sha256(preview.data).hexdigest()
    production_report = tmp_path / "production-campaign-report.json"
    production_report.write_text(
        json.dumps(
            {
                "model_image_sha256": model_sha,
                "schema": "VN97PRODCAMP1",
                "tokenizer_sha256": hashlib.sha256(
                    tokenizer_path.read_bytes()
                ).hexdigest(),
                "unified_checkpoint_sha256": loaded.checkpoint_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    evidence_path = tmp_path / "vn97-mobile-evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "battery_energy_counter_delta_nwh": 100,
                "device": {
                    "abi": "arm64-v8a",
                    "manufacturer": "fixture",
                    "model": "fixture-phone",
                    "sdk_int": 37,
                },
                "model_image_sha256": model_sha,
                "peak_pss_kib": 120000,
                "runs": 5,
                "schema": "VN97MOBEVID1",
                "speech_prefill": None,
                "text_decode_per_token": {
                    "p50_ms": 5.0,
                    "p95_ms": 6.0,
                },
                "text_prefill": {
                    "p50_ms": 10.0,
                    "p95_ms": 12.0,
                },
                "thermal_status_max": 2,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    private = cryptography.Ed25519PrivateKey.generate()
    private_path = tmp_path / "publisher.private"
    private_path.write_bytes(
        private.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    assets = tmp_path / "evidence-assets"
    assert main(
        [
            "--checkpoint", str(checkpoint),
            "--tokenizer", str(tokenizer_path),
            "--private-key", str(private_path),
            "--key-id", "publisher.main",
            "--capability-version", "11",
            "--source-origin", "vn97-production-campaign",
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
            "--production-campaign-report", str(production_report),
            "--require-production-campaign-report",
            "--device-evidence", str(evidence_path),
            "--require-device-evidence",
            "--max-text-prefill-p95-ms", "20",
            "--max-text-decode-p95-ms-per-token", "10",
            "--max-device-peak-pss-kib", "200000",
            "--max-device-thermal-status", "3",
        ]
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["device_evidence"]["evidence_sha256"] == hashlib.sha256(
        evidence_path.read_bytes()
    ).hexdigest()
    assert report["production_campaign_report_sha256"] == hashlib.sha256(
        production_report.read_bytes()
    ).hexdigest()
    assert report["device_evidence"]["model"] == "fixture-phone"
    assert report["model_image_sha256"] == model_sha


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
    assert report["schema"] == "VN97BOOTREL6"
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


def _release_candidate_fixture(tmp_path):
    checkpoint, tokenizer_path, validation = (
        _checkpoint_and_tokenizer(tmp_path)
    )
    loaded = load_deployment_checkpoint_file(
        checkpoint
    )
    tokenizer_bytes = tokenizer_path.read_bytes()
    tokenizer = VN97TokenizerPackage.from_bytes(
        tokenizer_bytes
    )
    tokenizer_sha = hashlib.sha256(
        tokenizer_bytes
    ).hexdigest()

    preview = build_model_image(
        loaded.model,
        tokenizer=tokenizer,
        tile_rows=4,
        tile_cols=4,
    )
    model_sha = hashlib.sha256(
        preview.data
    ).hexdigest()
    selected_id = "0123456789abcdef"

    production = {
        "model_image_sha256": model_sha,
        "schema": "VN97PRODCAMP1",
        "selected_candidate_id": selected_id,
        "tokenizer_sha256": tokenizer_sha,
        "unified_checkpoint_sha256":
            loaded.checkpoint_sha256,
    }
    production_bytes = json.dumps(
        production,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    evidence = {
        "battery_energy_counter_delta_nwh": 100,
        "device": {
            "abi": "arm64-v8a",
            "manufacturer": "fixture",
            "model": "fixture-phone",
            "sdk_int": 37,
        },
        "model_image_sha256": model_sha,
        "peak_pss_kib": 120000,
        "runs": 5,
        "schema": "VN97MOBEVID1",
        "speech_prefill": None,
        "text_decode_per_token": {
            "p50_ms": 5.0,
            "p95_ms": 6.0,
        },
        "text_prefill": {
            "p50_ms": 10.0,
            "p95_ms": 12.0,
        },
        "thermal_status_max": 2,
    }
    evidence_bytes = json.dumps(
        evidence,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    evidence_sha = hashlib.sha256(
        evidence_bytes
    ).hexdigest()

    candidate = tmp_path / "release-candidate"
    candidate.mkdir()
    (
        candidate /
        "model.vn97ck1"
    ).write_bytes(checkpoint.read_bytes())
    (
        candidate /
        "tokenizer.vn97tk1"
    ).write_bytes(tokenizer_bytes)
    (
        candidate /
        "production-campaign-report.json"
    ).write_bytes(production_bytes)

    evidence_dir = candidate / "device-evidence"
    evidence_dir.mkdir()
    (
        evidence_dir /
        f"001-{evidence_sha}.json"
    ).write_bytes(evidence_bytes)

    manifest = VN97ReleaseCandidateManifest(
        selected_candidate_id=selected_id,
        checkpoint_sha256=
            loaded.checkpoint_sha256,
        checkpoint_bytes=
            checkpoint.stat().st_size,
        tokenizer_sha256=tokenizer_sha,
        tokenizer_bytes=len(tokenizer_bytes),
        production_campaign_report_sha256=
            hashlib.sha256(
                production_bytes
            ).hexdigest(),
        model_image_sha256=model_sha,
        tile_rows=4,
        tile_cols=4,
        speech_enabled=False,
        vision_enabled=False,
        speech_training_report_sha256=None,
        vision_training_report_sha256=None,
        device_evidence=(
            VN97ReleaseCandidateDeviceEvidence(
                evidence_sha256=evidence_sha,
                manufacturer="fixture",
                model="fixture-phone",
                sdk_int=37,
                abi="arm64-v8a",
                runs=5,
                text_prefill_p95_ms=12.0,
                text_decode_p95_ms_per_token=6.0,
                speech_prefill_p95_ms=None,
                peak_pss_kib=120000,
                thermal_status_max=2,
                battery_energy_counter_delta_nwh=100,
            ),
        ),
    )
    manifest_bytes = manifest.to_bytes()
    (
        candidate /
        "release-candidate.vn97rc1"
    ).write_bytes(manifest_bytes)
    return (
        candidate,
        validation,
        hashlib.sha256(
            manifest_bytes
        ).hexdigest(),
    )


def test_release_cli_signs_verified_vn97rc1_candidate(
    tmp_path,
    capsys,
):
    candidate, validation, manifest_sha = (
        _release_candidate_fixture(
            tmp_path
        )
    )
    private = cryptography.Ed25519PrivateKey.generate()
    private_path = tmp_path / "candidate.private"
    private_path.write_bytes(
        private.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=
                serialization.NoEncryption(),
        )
    )
    assets = tmp_path / "candidate-assets"

    assert main(
        [
            "--release-candidate-dir",
            str(candidate),
            "--private-key",
            str(private_path),
            "--key-id",
            "publisher.main",
            "--capability-version",
            "12",
            "--source-origin",
            "vn97-production-campaign",
            "--source-license",
            "proprietary",
            "--assets-dir",
            str(assets),
            "--validation-input",
            str(validation),
            "--validation-format",
            "text",
            "--validation-sequence-length",
            "32",
            "--validation-batch-size",
            "1",
            "--min-validation-target-tokens",
            "1",
            "--max-validation-loss",
            "100",
        ]
    ) == 0

    report = json.loads(
        capsys.readouterr().out
    )
    assert report["schema"] == "VN97BOOTREL6"
    assert report["release_candidate"][
        "manifest_sha256"
    ] == manifest_sha
    assert report["release_candidate"][
        "selected_candidate_id"
    ] == "0123456789abcdef"
    assert report["release_candidate"][
        "device_evidence_sha256"
    ]
    assert {
        path.name
        for path in assets.iterdir()
    } == {
        "model.vn97cap1",
        "model.vn97sig1",
        "publisher.ed25519",
    }


def test_release_candidate_tamper_fails_before_private_key_read(
    tmp_path,
):
    candidate, validation, _ = (
        _release_candidate_fixture(
            tmp_path
        )
    )
    checkpoint = (
        candidate /
        "model.vn97ck1"
    )
    checkpoint.write_bytes(
        checkpoint.read_bytes() + b"x"
    )
    assets = tmp_path / "tampered-assets"

    with pytest.raises(
        Exception,
        match="checkpoint.*VN97RC1|SHA-256",
    ):
        main(
            [
                "--release-candidate-dir",
                str(candidate),
                "--private-key",
                str(
                    tmp_path /
                    "missing-private-key"
                ),
                "--key-id",
                "publisher.main",
                "--capability-version",
                "12",
                "--source-origin",
                "vn97-production-campaign",
                "--source-license",
                "proprietary",
                "--assets-dir",
                str(assets),
                "--validation-input",
                str(validation),
                "--validation-format",
                "text",
                "--validation-sequence-length",
                "32",
                "--validation-batch-size",
                "1",
                "--min-validation-target-tokens",
                "1",
                "--max-validation-loss",
                "100",
            ]
        )
    assert not assets.exists()
