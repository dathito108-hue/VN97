from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
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
INTAKE = load_module(
    "vn97.production_intake",
    "production_intake.py",
)

inspect_language = (
    INTAKE.inspect_language_campaign_directory
)
inspect_production = (
    INTAKE.inspect_production_campaign_directory
)
inspect_evidence = (
    INTAKE.inspect_device_evidence_files
)
Report = INTAKE.VN97ProductionIntakeReport
parse_report = (
    INTAKE.parse_production_intake_report
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


def expect_failure(label: str, fn) -> None:
    try:
        fn()
    except Exception:
        return
    raise AssertionError(
        f"expected failure: {label}"
    )


def build_language(root: Path):
    root.mkdir()
    checkpoint = (
        b"VN97CK1\0"
        + b"l" * 128
    )
    tokenizer = (
        b"VN97TK1\0"
        + b"t" * 64
    )
    checkpoint_sha = sha(checkpoint)
    tokenizer_sha = sha(tokenizer)
    candidate = "0123456789abcdef"
    report = {
        "candidates": [
            {
                "candidate": {
                    "d_model": 8,
                },
                "candidate_id": candidate,
                "checkpoint_sha256":
                    checkpoint_sha,
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
            candidate,
        "selected_checkpoint_sha256":
            checkpoint_sha,
        "tokenizer_sha256":
            tokenizer_sha,
        "training_dataset_sha256":
            "22" * 32,
        "validation_dataset_sha256":
            "33" * 32,
    }
    report_bytes = canonical(report)
    (
        root /
        "model.vn97ck1"
    ).write_bytes(checkpoint)
    (
        root /
        "tokenizer.vn97tk1"
    ).write_bytes(tokenizer)
    (
        root /
        "campaign-report.json"
    ).write_bytes(report_bytes)
    return {
        "candidate": candidate,
        "checkpoint": checkpoint,
        "checkpoint_sha": checkpoint_sha,
        "tokenizer": tokenizer,
        "tokenizer_sha": tokenizer_sha,
        "report_sha": sha(report_bytes),
    }


def build_production(
    root: Path,
    language: dict[str, object],
):
    root.mkdir()
    checkpoint = (
        b"VN97CK1\0"
        + b"p" * 196
    )
    checkpoint_sha = sha(checkpoint)
    model_image_sha = "55" * 32
    speech_dataset_sha = "66" * 32
    speech_report = {
        "base_checkpoint_sha256":
            language["checkpoint_sha"],
        "checkpoint_sha256":
            checkpoint_sha,
        "dataset_sha256":
            speech_dataset_sha,
        "examples": 8,
        "final_loss": 1.0,
        "mean_loss": 1.2,
        "schema": "VN97SPEECHTRAIN1",
        "steps": 4,
        "target_tokens": 32,
        "tokenizer_sha256":
            language["tokenizer_sha"],
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
            language["report_sha"],
        "language_checkpoint_sha256":
            language["checkpoint_sha"],
        "mobile_budget": {
            "max_model_image_bytes":
                1024,
            "max_recurrent_state_bytes":
                1024,
        },
        "mobile_footprint": {
            "float_parameter_bytes": 1,
            "model_image_bytes": 1,
            "packed_ternary_bytes": 1,
            "recurrent_state_bytes": 1,
            "section_count": 1,
            "tokenizer_bytes": len(
                language["tokenizer"]
            ),
        },
        "model_image_sha256":
            model_image_sha,
        "schema": "VN97PRODCAMP1",
        "selected_candidate_id":
            language["candidate"],
        "speech_release": {},
        "speech_training_dataset_sha256":
            speech_dataset_sha,
        "speech_validation": {},
        "tokenizer_sha256":
            language["tokenizer_sha"],
        "unified_checkpoint_sha256":
            checkpoint_sha,
    }
    production_bytes = canonical(
        production_report
    )
    (
        root /
        "model.vn97ck1"
    ).write_bytes(checkpoint)
    (
        root /
        "tokenizer.vn97tk1"
    ).write_bytes(
        language["tokenizer"]
    )
    (
        root /
        "speech-training-report.json"
    ).write_bytes(speech_bytes)
    (
        root /
        "production-campaign-report.json"
    ).write_bytes(production_bytes)
    return {
        "checkpoint_sha": checkpoint_sha,
        "model_image_sha":
            model_image_sha,
        "report_sha":
            sha(production_bytes),
        "speech_report_sha":
            sha(speech_bytes),
    }


def build_evidence(
    path: Path,
    *,
    model_image_sha: str,
    manufacturer: str,
    model: str,
) -> str:
    payload = {
        "battery_energy_counter_delta_nwh":
            100,
        "device": {
            "abi": "arm64-v8a",
            "manufacturer":
                manufacturer,
            "model": model,
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
    data = canonical(payload)
    path.write_bytes(data)
    return sha(data)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        language_dir = base / "language"
        language_fixture = (
            build_language(
                language_dir
            )
        )
        language = inspect_language(
            language_dir
        )
        assert (
            language.report_sha256
            == language_fixture[
                "report_sha"
            ]
        )

        production_dir = (
            base /
            "production"
        )
        production_fixture = (
            build_production(
                production_dir,
                language_fixture,
            )
        )
        production = inspect_production(
            production_dir,
            language=language,
        )
        assert (
            production.report_sha256
            == production_fixture[
                "report_sha"
            ]
        )
        assert (
            production.model_image_sha256
            == production_fixture[
                "model_image_sha"
            ]
        )

        evidence_a = base / "phone-a.json"
        evidence_b = base / "phone-b.json"
        evidence_sha_a = build_evidence(
            evidence_a,
            model_image_sha=
                production
                .model_image_sha256,
            manufacturer="fixture-a",
            model="phone-a",
        )
        evidence_sha_b = build_evidence(
            evidence_b,
            model_image_sha=
                production
                .model_image_sha256,
            manufacturer="fixture-b",
            model="phone-b",
        )
        evidence = inspect_evidence(
            [
                evidence_b,
                evidence_a,
            ],
            expected_model_image_sha256=
                production
                .model_image_sha256,
        )
        assert tuple(
            item.evidence_sha256
            for item in evidence
        ) == tuple(
            sorted(
                (
                    evidence_sha_a,
                    evidence_sha_b,
                )
            )
        )

        report = Report(
            selected_candidate_id=
                production
                .selected_candidate_id,
            language_campaign_report_sha256=
                language.report_sha256,
            language_checkpoint_sha256=
                language
                .selected_checkpoint_sha256,
            production_campaign_report_sha256=
                production.report_sha256,
            speech_training_report_sha256=
                production
                .speech_training_report_sha256,
            unified_checkpoint_sha256=
                production
                .unified_checkpoint_sha256,
            tokenizer_sha256=
                production.tokenizer_sha256,
            model_image_sha256=
                production.model_image_sha256,
            device_evidence_sha256=
                tuple(
                    item.evidence_sha256
                    for item in evidence
                ),
            distinct_device_profiles=2,
            release_candidate_manifest_sha256=
                "77" * 32,
        )
        encoded = report.to_bytes()
        assert (
            parse_report(encoded)
            == report
        )

        duplicate = inspect_evidence
        expect_failure(
            "duplicate evidence",
            lambda: duplicate(
                [
                    evidence_a,
                    evidence_a,
                ],
                expected_model_image_sha256=
                    production
                    .model_image_sha256,
            ),
        )

        tampered_checkpoint = (
            language_dir /
            "model.vn97ck1"
        )
        tampered_checkpoint.write_bytes(
            tampered_checkpoint
            .read_bytes()
            + b"x"
        )
        expect_failure(
            "language checkpoint tamper",
            lambda:
                inspect_language(
                    language_dir
                ),
        )

        wrong_language_dir = (
            base /
            "language-clean"
        )
        wrong_fixture = build_language(
            wrong_language_dir
        )
        wrong_language = (
            inspect_language(
                wrong_language_dir
            )
        )
        production_report_path = (
            production_dir /
            "production-campaign-report.json"
        )
        production_report = json.loads(
            production_report_path
            .read_text(
                encoding="utf-8"
            )
        )
        production_report[
            "language_campaign_report_sha256"
        ] = "aa" * 32
        production_report_path.write_bytes(
            canonical(
                production_report
            )
        )
        expect_failure(
            "production to language chain mismatch",
            lambda:
                inspect_production(
                    production_dir,
                    language=
                        wrong_language,
                ),
        )
        assert wrong_fixture

    print(
        "M19F real campaign/evidence intake contracts: PASS"
    )


if __name__ == "__main__":
    main()
