from __future__ import annotations

import hashlib
import json

import pytest
import torch

from vn97 import (
    AudioAdapterConfig,
    AudioFrameAdapter,
    VN97Config,
    VN97LanguageCore,
    VN97TokenizerPackage,
    build_model_image,
    estimate_vn97_mobile_footprint,
    save_deployment_checkpoint,
)
from vn97.release_candidate import (
    parse_release_candidate_manifest,
)
from vn97.release_candidate_cli import main


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _fixture(tmp_path):
    tokenizer = VN97TokenizerPackage(
        (b" the", b"ing", b"VN97")
    )
    tokenizer_path = tmp_path / "tokenizer.vn97tk1"
    tokenizer_bytes = tokenizer.to_bytes()
    tokenizer_path.write_bytes(tokenizer_bytes)
    tokenizer_sha = hashlib.sha256(
        tokenizer_bytes
    ).hexdigest()

    torch.manual_seed(97)
    model = VN97LanguageCore(
        VN97Config(
            vocab_size=tokenizer.vocab_size,
            d_model=8,
            n_layers=1,
            d_state=2,
        )
    ).eval()
    audio = AudioFrameAdapter(
        model.config.d_model,
        ternary_threshold=
            model.config.ternary_threshold,
        config=AudioAdapterConfig(),
        rms_eps=model.config.rms_eps,
    ).eval()
    checkpoint_path = tmp_path / "model.vn97ck1"
    save_deployment_checkpoint(
        model,
        checkpoint_path,
        audio_adapter=audio,
    )
    checkpoint_sha = hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()

    tile_rows = 4
    tile_cols = 4
    preview = build_model_image(
        model,
        tokenizer=tokenizer,
        audio_adapter=audio,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
    )
    model_sha = hashlib.sha256(
        preview.data
    ).hexdigest()
    footprint = estimate_vn97_mobile_footprint(
        model.config,
        tokenizer_nbytes=len(tokenizer_bytes),
        audio_frame_size=audio.projection.in_features,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        batch_size=1,
    )
    production = {
        "deployment": {
            "tile_cols": tile_cols,
            "tile_rows": tile_rows,
        },
        "language_campaign_report_sha256":
            "10" * 32,
        "language_checkpoint_sha256":
            "20" * 32,
        "mobile_budget": {
            "max_model_image_bytes":
                512 * 1024 * 1024,
            "max_recurrent_state_bytes":
                512 * 1024 * 1024,
        },
        "mobile_footprint":
            footprint.canonical_object(),
        "model_image_sha256": model_sha,
        "schema": "VN97PRODCAMP1",
        "selected_candidate_id":
            "0123456789abcdef",
        "speech_release": {
            "dataset_sha256": "30" * 32,
            "examples": 10,
            "max_loss": 5.0,
            "mean_loss": 1.0,
            "min_target_tokens": 5,
            "min_top1_accuracy": 0.1,
            "target_tokens": 20,
            "top1_accuracy": 0.5,
        },
        "speech_training_dataset_sha256":
            "40" * 32,
        "speech_validation": {
            "dataset_sha256": "50" * 32,
            "examples": 10,
            "max_loss": 5.0,
            "mean_loss": 1.0,
            "min_target_tokens": 5,
            "min_top1_accuracy": 0.1,
            "target_tokens": 20,
            "top1_accuracy": 0.5,
        },
        "tokenizer_sha256": tokenizer_sha,
        "unified_checkpoint_sha256":
            checkpoint_sha,
    }
    production_path = (
        tmp_path /
        "production-campaign-report.json"
    )
    production_path.write_bytes(
        _canonical(production)
    )

    speech_report = {
        "base_checkpoint_sha256": "60" * 32,
        "checkpoint_sha256": checkpoint_sha,
        "dataset_sha256": "40" * 32,
        "examples": 10,
        "final_loss": 1.0,
        "mean_loss": 1.1,
        "schema": "VN97SPEECHTRAIN1",
        "steps": 5,
        "target_tokens": 20,
        "tokenizer_sha256": tokenizer_sha,
        "training": {
            "epochs": 1,
        },
    }
    speech_path = (
        tmp_path /
        "speech-training-report.json"
    )
    speech_path.write_bytes(
        _canonical(speech_report)
    )

    evidence = {
        "battery_energy_counter_delta_nwh":
            100,
        "device": {
            "abi": "arm64-v8a",
            "manufacturer": "fixture",
            "model": "phone",
            "sdk_int": 37,
        },
        "model_image_sha256": model_sha,
        "peak_pss_kib": 120000,
        "runs": 5,
        "schema": "VN97MOBEVID1",
        "speech_prefill": {
            "p50_ms": 10.0,
            "p95_ms": 12.0,
        },
        "text_decode_per_token": {
            "p50_ms": 4.0,
            "p95_ms": 6.0,
        },
        "text_prefill": {
            "p50_ms": 8.0,
            "p95_ms": 10.0,
        },
        "thermal_status_max": 2,
    }
    evidence_path = (
        tmp_path /
        "vn97-mobile-evidence.json"
    )
    evidence_path.write_bytes(
        _canonical(evidence)
    )
    return {
        "checkpoint": checkpoint_path,
        "tokenizer": tokenizer_path,
        "production": production_path,
        "speech": speech_path,
        "evidence": evidence_path,
        "model_sha": model_sha,
        "checkpoint_sha": checkpoint_sha,
    }


def test_release_candidate_qualifies_and_assembles(
    tmp_path,
):
    fixture = _fixture(tmp_path)
    output = tmp_path / "candidate"
    assert main(
        [
            "--checkpoint",
            str(fixture["checkpoint"]),
            "--tokenizer",
            str(fixture["tokenizer"]),
            "--production-campaign-report",
            str(fixture["production"]),
            "--speech-training-report",
            str(fixture["speech"]),
            "--device-evidence",
            str(fixture["evidence"]),
            "--output-dir",
            str(output),
            "--max-speech-prefill-p95-ms",
            "100",
        ]
    ) == 0

    manifest = parse_release_candidate_manifest(
        (
            output /
            "release-candidate.vn97rc1"
        ).read_bytes()
    )
    assert (
        manifest.checkpoint_sha256
        == fixture["checkpoint_sha"]
    )
    assert (
        manifest.model_image_sha256
        == fixture["model_sha"]
    )
    assert manifest.speech_enabled is True
    assert manifest.vision_enabled is False
    assert len(manifest.device_evidence) == 1
    assert (
        output /
        "model.vn97ck1"
    ).read_bytes() == (
        fixture["checkpoint"].read_bytes()
    )
    assert (
        output /
        "tokenizer.vn97tk1"
    ).read_bytes() == (
        fixture["tokenizer"].read_bytes()
    )


def test_release_candidate_rejects_mismatched_device_model(
    tmp_path,
):
    fixture = _fixture(tmp_path)
    evidence = json.loads(
        fixture["evidence"].read_text(
            encoding="utf-8"
        )
    )
    evidence["model_image_sha256"] = "ff" * 32
    fixture["evidence"].write_bytes(
        _canonical(evidence)
    )

    with pytest.raises(
        Exception,
        match="model image identity mismatch",
    ):
        main(
            [
                "--checkpoint",
                str(fixture["checkpoint"]),
                "--tokenizer",
                str(fixture["tokenizer"]),
                "--production-campaign-report",
                str(fixture["production"]),
                "--speech-training-report",
                str(fixture["speech"]),
                "--device-evidence",
                str(fixture["evidence"]),
                "--output-dir",
                str(tmp_path / "candidate"),
            ]
        )


def test_release_candidate_requires_speech_report(
    tmp_path,
):
    fixture = _fixture(tmp_path)
    with pytest.raises(
        ValueError,
        match="speech-enabled checkpoint requires",
    ):
        main(
            [
                "--checkpoint",
                str(fixture["checkpoint"]),
                "--tokenizer",
                str(fixture["tokenizer"]),
                "--production-campaign-report",
                str(fixture["production"]),
                "--device-evidence",
                str(fixture["evidence"]),
                "--output-dir",
                str(tmp_path / "candidate"),
            ]
        )
