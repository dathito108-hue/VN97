from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
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


CLOSURE = load_module(
    "vn97.production_closure",
    "production_closure.py",
)

Report = CLOSURE.VN97ProductionClosureReport
Observation = CLOSURE.VN97ClosureObservation
parse_report = CLOSURE.parse_production_closure_report
phase = CLOSURE.closure_phase


def expect_failure(label: str, fn) -> None:
    try:
        fn()
    except Exception:
        return
    raise AssertionError(
        f"expected failure: {label}"
    )


def blocked(*codes: str):
    return SimpleNamespace(
        ready=False,
        status="BLOCKED",
        blockers=tuple(
            SimpleNamespace(code=code)
            for code in codes
        ),
    )


def ready():
    return SimpleNamespace(
        ready=True,
        status="READY",
        blockers=tuple(),
    )


def main() -> None:
    base = dict(
        manifest_sha256="11" * 32,
        repository_commit="ab" * 20,
        environment_ready=True,
        language_campaign_report_sha256=
            "22" * 32,
        production_campaign_report_sha256=
            "33" * 32,
        device_evidence_sha256=(
            "44" * 32,
            "55" * 32,
        ),
        intake_report_sha256=
            "66" * 32,
        release_candidate_manifest_sha256=
            "77" * 32,
    )

    complete = Report(
        **base,
        phase="READY_TO_RELEASE",
        readiness_report_sha256=
            "88" * 32,
        readiness_status="READY",
        readiness_blocker_codes=tuple(),
    )
    assert (
        parse_report(
            complete.to_bytes()
        )
        == complete
    )

    needs_phone = Report(
        manifest_sha256="11" * 32,
        repository_commit="ab" * 20,
        phase="NEEDS_PHYSICAL_EVIDENCE",
        environment_ready=True,
        language_campaign_report_sha256=
            "22" * 32,
        production_campaign_report_sha256=
            "33" * 32,
        device_evidence_sha256=tuple(),
        intake_report_sha256=None,
        release_candidate_manifest_sha256=None,
        readiness_report_sha256=None,
        readiness_status=None,
        readiness_blocker_codes=tuple(),
    )
    assert (
        parse_report(
            needs_phone.to_bytes()
        )
        == needs_phone
    )

    needs_signing = Report(
        **base,
        phase="NEEDS_SIGNING",
        readiness_report_sha256=
            "99" * 32,
        readiness_status="BLOCKED",
        readiness_blocker_codes=(
            "android_keystore.missing",
            "publisher_private_key.missing",
        ),
    )
    assert (
        parse_report(
            needs_signing.to_bytes()
        )
        == needs_signing
    )

    expect_failure(
        "ready with blocker",
        lambda: Report(
            **base,
            phase="READY_TO_RELEASE",
            readiness_report_sha256=
                "88" * 32,
            readiness_status="READY",
            readiness_blocker_codes=(
                "x",
            ),
        ),
    )
    expect_failure(
        "blocked without blocker",
        lambda: Report(
            **base,
            phase="READINESS_BLOCKED",
            readiness_report_sha256=
                "88" * 32,
            readiness_status="BLOCKED",
            readiness_blocker_codes=tuple(),
        ),
    )
    expect_failure(
        "noncanonical JSON",
        lambda: parse_report(
            complete
            .to_bytes()
            + b"\n"
        ),
    )

    assert phase(
        Observation(
            False,
            False,
            False,
            False,
            None,
        )
    ) == "NEEDS_TRAINING"
    assert phase(
        Observation(
            True,
            True,
            False,
            False,
            None,
        )
    ) == "NEEDS_PHYSICAL_EVIDENCE"
    assert phase(
        Observation(
            True,
            True,
            True,
            False,
            None,
        )
    ) == "NEEDS_INTAKE"
    assert phase(
        Observation(
            True,
            True,
            True,
            True,
            blocked(
                "language_validation.missing",
            ),
        )
    ) == "NEEDS_RELEASE_INPUTS"
    assert phase(
        Observation(
            True,
            True,
            True,
            True,
            blocked(
                "publisher_private_key.missing",
                "android_keystore.missing",
            ),
        )
    ) == "NEEDS_SIGNING"
    assert phase(
        Observation(
            True,
            True,
            True,
            True,
            blocked(
                "gradle.missing",
            ),
        )
    ) == "READINESS_BLOCKED"
    assert phase(
        Observation(
            True,
            True,
            True,
            True,
            ready(),
        )
    ) == "READY_TO_RELEASE"

    expect_failure(
        "intake before evidence",
        lambda: Observation(
            True,
            True,
            False,
            True,
            None,
        ),
    )
    expect_failure(
        "production before language",
        lambda: Observation(
            False,
            True,
            False,
            False,
            None,
        ),
    )

    with tempfile.TemporaryDirectory() as tmp:
        target = (
            Path(tmp) /
            "closure.vn97close1"
        )
        CLOSURE._atomic_write_replace(
            target,
            needs_phone.to_bytes(),
        )
        assert (
            parse_report(
                target.read_bytes()
            )
            == needs_phone
        )
        CLOSURE._atomic_write_replace(
            target,
            complete.to_bytes(),
        )
        assert (
            parse_report(
                target.read_bytes()
            )
            == complete
        )

    source = (
        SRC /
        "production_closure.py"
    ).read_text(
        encoding="utf-8"
    )
    run_start = source.index(
        "def run_production_closure("
    )
    body = source[run_start:]
    training = body.index(
        '"--stage",\n                "train"'
    )
    evidence = body.index(
        "evidence_call("
    )
    intake = body.index(
        '"--stage",\n                "intake"'
    )
    readiness = body.index(
        "readiness = _readiness_report("
    )
    assert (
        training
        < evidence
        < intake
        < readiness
    )

    cli = (
        SRC /
        "production_closure_cli.py"
    ).read_text(
        encoding="utf-8"
    )
    assert (
        "vn97-production-close"
        in cli
    )
    assert (
        '== "READY_TO_RELEASE"'
        in cli
    )

    print(
        "M19K production closure contracts: PASS"
    )


if __name__ == "__main__":
    main()
