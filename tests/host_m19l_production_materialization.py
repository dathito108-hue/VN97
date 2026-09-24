from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import types
import zipfile


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


MAT = load_module(
    "vn97.production_materialization",
    "production_materialization.py",
)

Receipt = MAT.VN97FinalReleaseReceipt
Validation = MAT.VN97FinalValidationConfig
parse_receipt = MAT.parse_final_release_receipt


def expect_failure(label: str, fn) -> None:
    try:
        fn()
    except Exception:
        return
    raise AssertionError(
        f"expected failure: {label}"
    )


def main() -> None:
    receipt = Receipt(
        manifest_sha256="11" * 32,
        repository_commit="ab" * 20,
        closure_report_sha256="22" * 32,
        readiness_report_sha256="33" * 32,
        release_candidate_manifest_sha256=
            "44" * 32,
        release_attestation_sha256=
            "55" * 32,
        bootstrap_release_report_sha256=
            "66" * 32,
        release_manifest_sha256=
            "77" * 32,
        application_id="ai.vn97.app",
        version_code=190100,
        version_name="1.0.0-rc1",
        apk_bytes=123456,
        apk_sha256="88" * 32,
        signer_certificate_sha256=(
            "99" * 32,
        ),
    )
    assert (
        parse_receipt(
            receipt.to_bytes()
        )
        == receipt
    )
    assert (
        b'"schema":"VN97FINAL1"'
        in receipt.to_bytes()
    )
    assert (
        b'"status":"MATERIALIZED"'
        in receipt.to_bytes()
    )

    expect_failure(
        "trailing final receipt data",
        lambda: parse_receipt(
            receipt.to_bytes()
            + b"\n"
        ),
    )
    expect_failure(
        "unsorted signers",
        lambda: Receipt(
            manifest_sha256="11" * 32,
            repository_commit="ab" * 20,
            closure_report_sha256=
                "22" * 32,
            readiness_report_sha256=
                "33" * 32,
            release_candidate_manifest_sha256=
                "44" * 32,
            release_attestation_sha256=
                "55" * 32,
            bootstrap_release_report_sha256=
                "66" * 32,
            release_manifest_sha256=
                "77" * 32,
            application_id=
                "ai.vn97.app",
            version_code=190100,
            version_name="1.0.0-rc1",
            apk_bytes=123456,
            apk_sha256="88" * 32,
            signer_certificate_sha256=(
                "bb" * 32,
                "aa" * 32,
            ),
        ),
    )

    Validation()
    expect_failure(
        "invalid accuracy",
        lambda: Validation(
            min_validation_accuracy=1.1
        ),
    )
    expect_failure(
        "invalid sequence length",
        lambda: Validation(
            validation_sequence_length=0
        ),
    )

    manifest = SimpleNamespace(
        intake_options=(
            (
                "max_device_peak_pss_kib",
                111111,
            ),
            (
                "max_device_thermal_status",
                4,
            ),
            (
                "max_speech_prefill_p95_ms",
                321.0,
            ),
            (
                "max_text_decode_p95_ms_per_token",
                123.0,
            ),
            (
                "max_text_prefill_p95_ms",
                456.0,
            ),
            (
                "min_device_runs",
                7,
            ),
            (
                "require_energy_counter",
                True,
            ),
        ),
        speech_options=(
            (
                "max_model_image_bytes",
                222222,
            ),
            (
                "max_recurrent_state_bytes",
                333333,
            ),
        ),
    )
    policy = MAT._release_policy_argv(
        manifest
    )
    joined = " ".join(policy)
    for expected in (
        "--device-evidence-min-runs 7",
        "--max-text-prefill-p95-ms 456.0",
        "--max-text-decode-p95-ms-per-token 123.0",
        "--max-device-peak-pss-kib 111111",
        "--max-device-thermal-status 4",
        "--max-speech-prefill-p95-ms 321.0",
        "--require-device-energy-counter",
        "--max-model-image-bytes 222222",
        "--max-recurrent-state-bytes 333333",
    ):
        assert expected in joined

    inputs = SimpleNamespace(
        publisher_private_key=
            Path("/outside/key"),
        validation_inputs=(
            Path("/fresh/lang.jsonl"),
        ),
        speech_validation_input=
            Path("/fresh/speech.jsonl"),
        vision_validation_input=None,
        output_dir=
            Path("/out/release"),
        key_id="prod-key",
        capability_version=9,
        source_origin="vn97:production",
        source_license="internal",
        max_validation_loss=2.5,
        max_speech_validation_loss=3.5,
        max_vision_validation_loss=None,
        gradle="gradle",
        apksigner="/sdk/apksigner",
        aapt="/sdk/aapt",
    )
    args = MAT.build_release_argv(
        manifest=manifest,
        repository_root=Path("/repo"),
        candidate_dir=
            Path("/workspace/rc"),
        readiness_inputs=inputs,
        validation=Validation(),
    )
    text = " ".join(args)
    for expected in (
        "--repository-root /repo",
        "--release-candidate-dir /workspace/rc",
        "--private-key /outside/key",
        "--validation-input /fresh/lang.jsonl",
        "--speech-validation-input /fresh/speech.jsonl",
        "--max-validation-loss 2.5",
        "--max-speech-validation-loss 3.5",
        "--output-dir /out/release",
        "--apksigner /sdk/apksigner",
        "--aapt /sdk/aapt",
    ):
        assert expected in text

    with tempfile.TemporaryDirectory() as tmp:
        target = (
            Path(tmp)
            / "final.vn97final1"
        )
        MAT._atomic_create(
            target,
            receipt.to_bytes(),
        )
        assert parse_receipt(
            target.read_bytes()
        ) == receipt
        expect_failure(
            "final receipt overwrite",
            lambda: MAT._atomic_create(
                target,
                receipt.to_bytes(),
            ),
        )

    source = (
        SRC /
        "production_materialization.py"
    ).read_text(
        encoding="utf-8"
    )
    run_start = source.index(
        "def materialize_final_release("
    )
    body = source[run_start:]
    closure = body.index(
        "closure = run_production_closure("
    )
    ready_gate = body.index(
        'closure.phase != (\n        "READY_TO_RELEASE"'
    )
    release_call = body.index(
        "result = release_runner(args)"
    )
    verify = body.index(
        "final = verify_materialized_release("
    )
    receipt_write = body.index(
        "_atomic_create(",
        verify,
    )
    assert (
        closure
        < ready_gate
        < release_call
        < verify
        < receipt_write
    )

    cli = (
        SRC /
        "production_materialization_cli.py"
    ).read_text(
        encoding="utf-8"
    )
    assert (
        "VN97ProductionMaterializationBlocked"
        in cli
    )
    assert (
        "return 2"
        in cli
    )
    assert (
        "return 0"
        in cli
    )

    print(
        "M19L final turnkey materialization contracts: PASS"
    )


if __name__ == "__main__":
    main()
