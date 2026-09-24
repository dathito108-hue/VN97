from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "vn97"

pkg = types.ModuleType("vn97")
pkg.__path__ = [str(SRC)]
sys.modules["vn97"] = pkg


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name,
        SRC / filename,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"could not load {filename}"
        )
    module = importlib.util.module_from_spec(
        spec
    )
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DEVICE = load_module(
    "vn97.device_evidence",
    "device_evidence.py",
)
RC = load_module(
    "vn97.release_candidate",
    "release_candidate.py",
)
INTAKE = load_module(
    "vn97.production_intake",
    "production_intake.py",
)


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_sources(base: Path):
    language = base / "language"
    language.mkdir()
    language_checkpoint = (
        b"VN97CK1\0" + b"l" * 96
    )
    tokenizer = (
        b"VN97TK1\0" + b"t" * 64
    )
    language_checkpoint_sha = sha(
        language_checkpoint
    )
    tokenizer_sha = sha(tokenizer)
    candidate_id = "0123456789abcdef"

    language_report = {
        "candidates": [
            {
                "candidate": {},
                "candidate_id": candidate_id,
                "checkpoint_sha256":
                    language_checkpoint_sha,
                "evaluation": {},
                "mobile_footprint": {},
                "parameter_count": 1,
                "status": "ELIGIBLE",
                "training": {},
            }
        ],
        "criteria": {},
        "release_dataset_sha256":
            "11" * 32,
        "release_evaluation": {},
        "schema": "VN97CAMP2",
        "selected_candidate_id":
            candidate_id,
        "selected_checkpoint_sha256":
            language_checkpoint_sha,
        "tokenizer_sha256":
            tokenizer_sha,
        "training_dataset_sha256":
            "22" * 32,
        "validation_dataset_sha256":
            "33" * 32,
    }
    language_report_bytes = canonical(
        language_report
    )
    (language / "model.vn97ck1").write_bytes(
        language_checkpoint
    )
    (language / "tokenizer.vn97tk1").write_bytes(
        tokenizer
    )
    (
        language /
        "campaign-report.json"
    ).write_bytes(
        language_report_bytes
    )

    production = base / "production"
    production.mkdir()
    unified_checkpoint = (
        b"VN97CK1\0" + b"u" * 144
    )
    unified_sha = sha(
        unified_checkpoint
    )
    model_image_sha = "55" * 32
    speech_dataset_sha = "66" * 32
    speech_report = {
        "base_checkpoint_sha256":
            language_checkpoint_sha,
        "checkpoint_sha256":
            unified_sha,
        "dataset_sha256":
            speech_dataset_sha,
        "examples": 8,
        "final_loss": 1.0,
        "mean_loss": 1.1,
        "schema": "VN97SPEECHTRAIN1",
        "steps": 4,
        "target_tokens": 32,
        "tokenizer_sha256":
            tokenizer_sha,
        "training": {
            "epochs": 1,
        },
    }
    speech_bytes = canonical(
        speech_report
    )
    production_report = {
        "deployment": {
            "tile_cols": 16,
            "tile_rows": 16,
        },
        "language_campaign_report_sha256":
            sha(language_report_bytes),
        "language_checkpoint_sha256":
            language_checkpoint_sha,
        "mobile_budget": {
            "max_model_image_bytes": 1024,
            "max_recurrent_state_bytes": 1024,
        },
        "mobile_footprint": {
            "float_parameter_bytes": 1,
            "model_image_bytes": 1,
            "packed_ternary_bytes": 1,
            "recurrent_state_bytes": 1,
            "section_count": 1,
            "tokenizer_bytes": len(
                tokenizer
            ),
        },
        "model_image_sha256":
            model_image_sha,
        "schema": "VN97PRODCAMP1",
        "selected_candidate_id":
            candidate_id,
        "speech_release": {},
        "speech_training_dataset_sha256":
            speech_dataset_sha,
        "speech_validation": {},
        "tokenizer_sha256":
            tokenizer_sha,
        "unified_checkpoint_sha256":
            unified_sha,
    }
    production_report_bytes = canonical(
        production_report
    )
    (
        production /
        "model.vn97ck1"
    ).write_bytes(
        unified_checkpoint
    )
    (
        production /
        "tokenizer.vn97tk1"
    ).write_bytes(tokenizer)
    (
        production /
        "speech-training-report.json"
    ).write_bytes(speech_bytes)
    (
        production /
        "production-campaign-report.json"
    ).write_bytes(
        production_report_bytes
    )

    evidence_path = base / "evidence.json"
    evidence_payload = {
        "battery_energy_counter_delta_nwh":
            100,
        "device": {
            "abi": "arm64-v8a",
            "manufacturer":
                "fixture",
            "model": "phone",
            "sdk_int": 37,
        },
        "model_image_sha256":
            model_image_sha,
        "peak_pss_kib": 120000,
        "runs": 5,
        "schema": "VN97MOBEVID1",
        "speech_prefill": {
            "p50_ms": 10.0,
            "p95_ms": 12.0,
        },
        "text_decode_per_token": {
            "p50_ms": 5.0,
            "p95_ms": 6.0,
        },
        "text_prefill": {
            "p50_ms": 8.0,
            "p95_ms": 10.0,
        },
        "thermal_status_max": 2,
    }
    evidence_path.write_bytes(
        canonical(evidence_payload)
    )
    return (
        language,
        production,
        evidence_path,
    )


def fake_release_candidate_main(
    argv: list[str],
) -> int:
    def one(flag: str) -> str:
        index = argv.index(flag)
        return argv[index + 1]

    def repeated(flag: str) -> list[str]:
        result: list[str] = []
        for index, value in enumerate(argv):
            if value == flag:
                result.append(
                    argv[index + 1]
                )
        return result

    checkpoint_path = Path(
        one("--checkpoint")
    )
    tokenizer_path = Path(
        one("--tokenizer")
    )
    production_report_path = Path(
        one(
            "--production-campaign-report"
        )
    )
    speech_report_path = Path(
        one("--speech-training-report")
    )
    output = Path(
        one("--output-dir")
    )
    evidence_paths = [
        Path(value)
        for value in repeated(
            "--device-evidence"
        )
    ]

    production_report = json.loads(
        production_report_path.read_text(
            encoding="utf-8"
        )
    )
    checkpoint_bytes = (
        checkpoint_path.read_bytes()
    )
    tokenizer_bytes = (
        tokenizer_path.read_bytes()
    )
    production_bytes = (
        production_report_path
        .read_bytes()
    )
    speech_bytes = (
        speech_report_path.read_bytes()
    )

    evidence_items = [
        DEVICE.load_device_evidence(
            path
        )
        for path in evidence_paths
    ]
    manifest_evidence = tuple(
        RC.VN97ReleaseCandidateDeviceEvidence(
            evidence_sha256=
                item.evidence_sha256,
            manufacturer=
                item.manufacturer,
            model=item.model,
            sdk_int=item.sdk_int,
            abi=item.abi,
            runs=item.runs,
            text_prefill_p95_ms=
                item.text_prefill.p95_ms,
            text_decode_p95_ms_per_token=
                item
                .text_decode_per_token
                .p95_ms,
            speech_prefill_p95_ms=(
                None
                if item.speech_prefill
                is None
                else item
                .speech_prefill
                .p95_ms
            ),
            peak_pss_kib=
                item.peak_pss_kib,
            thermal_status_max=
                item.thermal_status_max,
            battery_energy_counter_delta_nwh=
                item
                .battery_energy_counter_delta_nwh,
        )
        for item in sorted(
            evidence_items,
            key=lambda item:
                item.evidence_sha256,
        )
    )
    manifest = RC.VN97ReleaseCandidateManifest(
        selected_candidate_id=
            production_report[
                "selected_candidate_id"
            ],
        checkpoint_sha256=
            sha(checkpoint_bytes),
        checkpoint_bytes=
            len(checkpoint_bytes),
        tokenizer_sha256=
            sha(tokenizer_bytes),
        tokenizer_bytes=
            len(tokenizer_bytes),
        production_campaign_report_sha256=
            sha(production_bytes),
        model_image_sha256=
            production_report[
                "model_image_sha256"
            ],
        tile_rows=
            production_report[
                "deployment"
            ]["tile_rows"],
        tile_cols=
            production_report[
                "deployment"
            ]["tile_cols"],
        speech_enabled=True,
        vision_enabled=False,
        speech_training_report_sha256=
            sha(speech_bytes),
        vision_training_report_sha256=None,
        device_evidence=
            manifest_evidence,
    )

    output.mkdir()
    shutil.copyfile(
        checkpoint_path,
        output / "model.vn97ck1",
    )
    shutil.copyfile(
        tokenizer_path,
        output / "tokenizer.vn97tk1",
    )
    shutil.copyfile(
        production_report_path,
        output /
            "production-campaign-report.json",
    )
    shutil.copyfile(
        speech_report_path,
        output /
            "speech-training-report.json",
    )
    evidence_root = (
        output /
        "device-evidence"
    )
    evidence_root.mkdir()
    for index, item in enumerate(
        sorted(
            evidence_items,
            key=lambda item:
                item.evidence_sha256,
        ),
        start=1,
    ):
        source = next(
            path
            for path in evidence_paths
            if (
                DEVICE.load_device_evidence(
                    path
                ).evidence_sha256
                == item.evidence_sha256
            )
        )
        shutil.copyfile(
            source,
            evidence_root /
                (
                    f"{index:03d}-"
                    f"{item.evidence_sha256}.json"
                ),
        )
    (
        output /
        "release-candidate.vn97rc1"
    ).write_bytes(
        manifest.to_bytes()
    )
    return 0


fake = types.ModuleType(
    "vn97.release_candidate_cli"
)
fake.main = fake_release_candidate_main
sys.modules[
    "vn97.release_candidate_cli"
] = fake

CLI = load_module(
    "vn97.production_intake_cli",
    "production_intake_cli.py",
)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        (
            language,
            production,
            evidence,
        ) = build_sources(base)
        output = base / "intake"

        result = CLI.main(
            [
                "--language-campaign-dir",
                str(language),
                "--production-campaign-dir",
                str(production),
                "--device-evidence",
                str(evidence),
                "--output-dir",
                str(output),
            ]
        )
        assert result == 0
        assert {
            entry.name
            for entry in output.iterdir()
        } == {
            "language-campaign-report.json",
            "production-intake.vn97intake1",
            "release-candidate",
        }

        report = (
            INTAKE
            .parse_production_intake_report(
                (
                    output /
                    "production-intake.vn97intake1"
                ).read_bytes()
            )
        )
        candidate = (
            RC.load_release_candidate_directory(
                output /
                "release-candidate"
            )
        )
        assert (
            report
            .release_candidate_manifest_sha256
            == candidate.manifest_sha256
        )
        assert (
            report.model_image_sha256
            == candidate
            .manifest
            .model_image_sha256
        )
        assert (
            sha(
                (
                    output /
                    "language-campaign-report.json"
                ).read_bytes()
            )
            == report
            .language_campaign_report_sha256
        )

    print(
        "M19F intake transaction contracts: PASS"
    )


if __name__ == "__main__":
    main()
