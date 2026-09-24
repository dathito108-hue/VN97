from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
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


RUN = load_module(
    "vn97.production_run_manifest",
    "production_run_manifest.py",
)
PREFLIGHT = load_module(
    "vn97.production_preflight",
    "production_preflight.py",
)

Candidate = PREFLIGHT.VN97PreflightCandidate
Blocker = PREFLIGHT.VN97PreflightBlocker
Report = PREFLIGHT.VN97ProductionPreflightReport
parse_report = (
    PREFLIGHT.parse_production_preflight_report
)
Receipt = RUN.VN97ProductionRunReceipt
parse_receipt = (
    RUN.parse_production_run_receipt
)


def expect_failure(label: str, fn) -> None:
    try:
        fn()
    except Exception:
        return
    raise AssertionError(
        f"expected failure: {label}"
    )


def ready_report() -> Report:
    candidate = Candidate(
        candidate_id="0123456789abcdef",
        parameter_count=12345,
        language_model_image_bytes=100000,
        language_recurrent_state_bytes=4096,
        speech_model_image_bytes=110000,
        speech_recurrent_state_bytes=4096,
        status="ADMITTED",
    )
    return Report(
        manifest_sha256="11" * 32,
        status="READY",
        tokenizer_sha256="22" * 32,
        tokenizer_bytes=1024,
        vocab_size=512,
        training_dataset_sha256="33" * 32,
        validation_dataset_sha256="44" * 32,
        release_dataset_sha256="55" * 32,
        training_records=100,
        validation_records=20,
        release_records=20,
        training_windows=120,
        validation_windows=20,
        release_windows=20,
        validation_target_tokens=1000,
        release_target_tokens=900,
        speech_training_dataset_sha256="66" * 32,
        speech_validation_dataset_sha256="77" * 32,
        speech_release_dataset_sha256="88" * 32,
        speech_training_examples=50,
        speech_validation_examples=10,
        speech_release_examples=10,
        speech_validation_target_tokens=400,
        speech_release_target_tokens=350,
        speech_max_observed_frames=100,
        candidates=(candidate,),
        blockers=tuple(),
    )


def main() -> None:
    ready = ready_report()
    encoded = ready.to_bytes()
    assert (
        hashlib.sha256(encoded)
        .hexdigest()
    )
    assert parse_report(encoded) == ready
    assert ready.ready

    blocked_candidate = Candidate(
        candidate_id="fedcba9876543210",
        parameter_count=22222,
        language_model_image_bytes=100000,
        language_recurrent_state_bytes=4096,
        speech_model_image_bytes=999999,
        speech_recurrent_state_bytes=4096,
        status=
            "REJECTED_SPEECH_MODEL_IMAGE_BUDGET",
    )
    blocked = Report(
        manifest_sha256="aa" * 32,
        status="BLOCKED",
        tokenizer_sha256="bb" * 32,
        tokenizer_bytes=2048,
        vocab_size=768,
        training_dataset_sha256="cc" * 32,
        validation_dataset_sha256="dd" * 32,
        release_dataset_sha256="ee" * 32,
        training_records=80,
        validation_records=10,
        release_records=10,
        training_windows=100,
        validation_windows=12,
        release_windows=12,
        validation_target_tokens=600,
        release_target_tokens=550,
        speech_training_dataset_sha256="12" * 32,
        speech_validation_dataset_sha256="23" * 32,
        speech_release_dataset_sha256="34" * 32,
        speech_training_examples=30,
        speech_validation_examples=8,
        speech_release_examples=8,
        speech_validation_target_tokens=250,
        speech_release_target_tokens=240,
        speech_max_observed_frames=90,
        candidates=(blocked_candidate,),
        blockers=(
            Blocker(
                code=
                    "candidate.language_winner_may_fail_speech_budget",
                message=
                    "candidate cannot fit final speech-enabled budget",
            ),
        ),
    )
    assert not blocked.ready
    assert (
        parse_report(
            blocked.to_bytes()
        )
        == blocked
    )

    expect_failure(
        "ready report with blocker",
        lambda: Report(
            **{
                **ready.__dict__,
                "blockers": (
                    Blocker(
                        code="x",
                        message="y",
                    ),
                ),
            }
        ),
    )

    preflight_sha = hashlib.sha256(
        encoded
    ).hexdigest()
    preflight_receipt = Receipt(
        manifest_sha256="11" * 32,
        repository_commit="ab" * 20,
        stage="preflight",
        environment_ready=True,
        device_evidence_ready=False,
        preflight_report_sha256=
            preflight_sha,
        language_campaign_report_sha256=None,
        production_campaign_report_sha256=None,
        intake_report_sha256=None,
        release_candidate_manifest_sha256=None,
    )
    assert (
        parse_receipt(
            preflight_receipt.to_bytes()
        )
        == preflight_receipt
    )

    language_receipt = Receipt(
        manifest_sha256="11" * 32,
        repository_commit="ab" * 20,
        stage="language",
        environment_ready=True,
        device_evidence_ready=False,
        preflight_report_sha256=
            preflight_sha,
        language_campaign_report_sha256=
            "91" * 32,
        production_campaign_report_sha256=None,
        intake_report_sha256=None,
        release_candidate_manifest_sha256=None,
    )
    assert (
        parse_receipt(
            language_receipt.to_bytes()
        )
        == language_receipt
    )

    intake_receipt = Receipt(
        manifest_sha256="11" * 32,
        repository_commit="ab" * 20,
        stage="intake",
        environment_ready=True,
        device_evidence_ready=True,
        preflight_report_sha256=None,
        language_campaign_report_sha256=
            "91" * 32,
        production_campaign_report_sha256=
            "92" * 32,
        intake_report_sha256=
            "93" * 32,
        release_candidate_manifest_sha256=
            "94" * 32,
    )
    assert (
        parse_receipt(
            intake_receipt.to_bytes()
        )
        == intake_receipt
    )

    expect_failure(
        "language receipt without preflight",
        lambda: Receipt(
            manifest_sha256="11" * 32,
            repository_commit="ab" * 20,
            stage="language",
            environment_ready=True,
            device_evidence_ready=False,
            preflight_report_sha256=None,
            language_campaign_report_sha256=
                "91" * 32,
            production_campaign_report_sha256=None,
            intake_report_sha256=None,
            release_candidate_manifest_sha256=None,
        ),
    )

    runner = (
        SRC /
        "production_run_cli.py"
    ).read_text(
        encoding="utf-8"
    )
    main_start = runner.index(
        "def main("
    )
    body = runner[main_start:]
    preflight_call = body.index(
        "preflight = run_production_preflight("
    )
    blocked_check = body.index(
        "if not preflight.ready:"
    )
    first_training = body.index(
        "_run_module("
    )
    assert (
        preflight_call
        < blocked_check
        < first_training
    )
    assert '"preflight",' in body

    print(
        "M19I zero-compute preflight contracts: PASS"
    )


if __name__ == "__main__":
    main()
