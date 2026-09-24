from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
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
RC = load_module(
    "vn97.release_candidate",
    "release_candidate.py",
)
RUN = load_module(
    "vn97.production_run_manifest",
    "production_run_manifest.py",
)
CLI = load_module(
    "vn97.production_run_cli",
    "production_run_cli.py",
)

RunFile = RUN.VN97RunFile
Speech = RUN.VN97SpeechRunInput
Manifest = RUN.VN97ProductionRunManifest
parse_manifest = (
    RUN.parse_production_run_manifest
)
verify_inputs = (
    RUN.verify_production_run_inputs
)


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def bound_file(
    workspace: Path,
    relative: str,
) -> RunFile:
    data = (
        workspace /
        relative
    ).read_bytes()
    return RunFile(
        path=relative,
        bytes=len(data),
        sha256=
            hashlib.sha256(
                data
            ).hexdigest(),
    )


def write(
    workspace: Path,
    relative: str,
    data: bytes,
) -> None:
    target = (
        workspace /
        relative
    )
    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    target.write_bytes(data)


def speech_input(
    workspace: Path,
    manifest_path: str,
    audio_path: str,
    *,
    transcript: str,
) -> Speech:
    audio_name = Path(
        audio_path
    ).name
    write(
        workspace,
        audio_path,
        (
            b"RIFF-fixture-"
            + transcript.encode(
                "utf-8"
            )
        ),
    )
    write(
        workspace,
        manifest_path,
        (
            json.dumps(
                {
                    "audio":
                        audio_name,
                    "text":
                        transcript,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8"),
    )
    return Speech(
        manifest=bound_file(
            workspace,
            manifest_path,
        ),
        audio=(
            bound_file(
                workspace,
                audio_path,
            ),
        ),
    )


def build_manifest(
    workspace: Path,
    commit: str,
) -> Manifest:
    for relative, text in (
        (
            "data/train.jsonl",
            '{"text":"train"}\n',
        ),
        (
            "data/validation.jsonl",
            '{"text":"validation"}\n',
        ),
        (
            "data/release.jsonl",
            '{"text":"release"}\n',
        ),
    ):
        write(
            workspace,
            relative,
            text.encode("utf-8"),
        )

    campaign = {
        "candidates": [
            {
                "d_model": 8,
                "d_state": 2,
                "embedding_rank": 4,
                "learning_rate":
                    0.001,
                "n_layers": 1,
                "seed": 0,
            }
        ],
        "schema": "VN97CAMPDEF1",
    }
    write(
        workspace,
        "config/campaign.json",
        canonical(campaign),
    )

    speech_train = speech_input(
        workspace,
        "speech/train/train.jsonl",
        "speech/train/train.wav",
        transcript="train speech",
    )
    speech_validation = speech_input(
        workspace,
        "speech/validation/validation.jsonl",
        "speech/validation/validation.wav",
        transcript="validation speech",
    )
    speech_release = speech_input(
        workspace,
        "speech/release/release.jsonl",
        "speech/release/release.wav",
        transcript="release speech",
    )

    return Manifest(
        repository_commit=commit,
        campaign_definition=
            bound_file(
                workspace,
                "config/campaign.json",
            ),
        language_training=(
            bound_file(
                workspace,
                "data/train.jsonl",
            ),
        ),
        language_validation=(
            bound_file(
                workspace,
                "data/validation.jsonl",
            ),
        ),
        language_release=(
            bound_file(
                workspace,
                "data/release.jsonl",
            ),
        ),
        speech_training=
            speech_train,
        speech_validation=
            speech_validation,
        speech_release=
            speech_release,
        device_evidence_dir=
            "device-evidence",
        language_options=(
            (
                "device",
                "cpu",
            ),
            (
                "max_parameters",
                1_000_000,
            ),
            (
                "max_validation_loss",
                10.0,
            ),
        ),
        speech_options=(
            (
                "device",
                "cpu",
            ),
            (
                "max_speech_validation_loss",
                10.0,
            ),
        ),
        intake_options=tuple(),
        language_output_dir=
            "out/language",
        production_output_dir=
            "out/production",
        intake_output_dir=
            "out/intake",
    )


def expect_failure(label: str, fn) -> None:
    try:
        fn()
    except Exception:
        return
    raise AssertionError(
        f"expected failure: {label}"
    )


def evidence_bytes() -> bytes:
    return canonical(
        {
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
                "55" * 32,
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
    )


def main() -> None:
    commit = subprocess.run(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip().lower()

    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        manifest = build_manifest(
            workspace,
            commit,
        )
        encoded = manifest.to_bytes()
        parsed = parse_manifest(
            encoded
        )
        assert parsed == manifest
        assert (
            parsed.manifest_sha256
            == hashlib.sha256(
                encoded
            ).hexdigest()
        )

        resolved = verify_inputs(
            parsed,
            workspace_root=
                workspace,
        )
        assert (
            resolved
            .language_output_dir
            == workspace /
            "out/language"
        )
        assert not (
            workspace /
            "out"
        ).exists()

        manifest_path = (
            workspace /
            "production-run.vn97run1"
        )
        manifest_path.write_bytes(
            encoded
        )
        receipt = (
            workspace /
            "verify.vn97runexec1"
        )
        assert CLI.main(
            [
                "--manifest",
                str(manifest_path),
                "--workspace-root",
                str(workspace),
                "--repository-root",
                str(ROOT),
                "--stage",
                "verify",
                "--receipt",
                str(receipt),
            ]
        ) == 0
        verify_receipt = json.loads(
            receipt.read_text(
                encoding="utf-8"
            )
        )
        assert (
            verify_receipt["schema"]
            == "VN97RUNEXEC1"
        )
        assert (
            verify_receipt[
                "manifest_sha256"
            ]
            == manifest
            .manifest_sha256
        )
        assert (
            verify_receipt[
                "device_evidence_ready"
            ]
            is False
        )

        expect_failure(
            "all stage without physical evidence",
            lambda: CLI.main(
                [
                    "--manifest",
                    str(manifest_path),
                    "--workspace-root",
                    str(workspace),
                    "--repository-root",
                    str(ROOT),
                    "--stage",
                    "all",
                ]
            ),
        )
        assert not (
            workspace /
            "out"
        ).exists()

        evidence_dir = (
            workspace /
            "device-evidence"
        )
        evidence_dir.mkdir()
        (
            evidence_dir /
            "phone.json"
        ).write_bytes(
            evidence_bytes()
        )
        ready_receipt = (
            workspace /
            "verify-evidence.vn97runexec1"
        )
        assert CLI.main(
            [
                "--manifest",
                str(manifest_path),
                "--workspace-root",
                str(workspace),
                "--repository-root",
                str(ROOT),
                "--stage",
                "verify",
                "--receipt",
                str(ready_receipt),
            ]
        ) == 0
        assert (
            json.loads(
                ready_receipt
                .read_text(
                    encoding="utf-8"
                )
            )[
                "device_evidence_ready"
            ]
            is True
        )

        audio = (
            workspace /
            "speech/train/train.wav"
        )
        original = (
            audio.read_bytes()
        )
        audio.write_bytes(
            original + b"x"
        )
        expect_failure(
            "bound speech audio tamper",
            lambda: verify_inputs(
                manifest,
                workspace_root=
                    workspace,
            ),
        )
        audio.write_bytes(
            original
        )

        raw = manifest.canonical_object()
        raw["language"][
            "options"
        ]["device"] = "auto"
        expect_failure(
            "auto production device",
            lambda: parse_manifest(
                canonical(raw)
            ),
        )

        duplicate = manifest.canonical_object()
        duplicate[
            "inputs"
        ][
            "speech_validation"
        ][
            "audio"
        ] = duplicate[
            "inputs"
        ][
            "speech_training"
        ][
            "audio"
        ]
        expect_failure(
            "cross-split speech audio reuse",
            lambda: parse_manifest(
                canonical(
                    duplicate
                )
            ),
        )

        overlap = manifest.canonical_object()
        overlap[
            "intake"
        ][
            "device_evidence_dir"
        ] = "out/intake/evidence"
        expect_failure(
            "evidence/output overlap",
            lambda: parse_manifest(
                canonical(overlap)
            ),
        )

    print(
        "M19G reproducible production run contracts: PASS"
    )


if __name__ == "__main__":
    main()
