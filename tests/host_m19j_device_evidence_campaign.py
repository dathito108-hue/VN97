from __future__ import annotations

import hashlib
import importlib.util
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
RUN = load_module(
    "vn97.production_run_manifest",
    "production_run_manifest.py",
)
CAMPAIGN = load_module(
    "vn97.device_evidence_campaign",
    "device_evidence_campaign.py",
)

Config = CAMPAIGN.VN97PhysicalEvidenceConfig
Assets = CAMPAIGN.VN97EvidenceAssets
Collected = CAMPAIGN.VN97CollectedEvidence
AdbClient = CAMPAIGN.VN97AdbClient
Criteria = DEVICE.VN97DeviceEvidenceCriteria


def evidence_bytes(
    *,
    model_sha: str = "a" * 64,
) -> bytes:
    import json

    return json.dumps(
        {
            "battery_energy_counter_delta_nwh":
                100,
            "device": {
                "abi": "arm64-v8a",
                "manufacturer":
                    "VN97Fixture",
                "model":
                    "PhysicalPhone",
                "sdk_int": 37,
            },
            "model_image_sha256":
                model_sha,
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
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def expect_failure(label: str, fn) -> None:
    try:
        fn()
    except Exception:
        return
    raise AssertionError(
        f"expected failure: {label}"
    )


class FakeExecutor:
    def __init__(
        self,
        evidence: bytes,
        *,
        emulator: bool = False,
        installed: bool = False,
    ) -> None:
        self.evidence = evidence
        self.emulator = emulator
        self.installed = installed
        self.triggered = False
        self.commands: list[
            tuple[list[str], bytes | None]
        ] = []

    def __call__(
        self,
        command: list[str],
        input_bytes: bytes | None,
        timeout: float | None,
    ) -> subprocess.CompletedProcess[bytes]:
        self.commands.append(
            (
                list(command),
                input_bytes,
            )
        )
        text = " ".join(command)
        if " get-state" in text:
            return subprocess.CompletedProcess(
                command,
                0,
                b"device\n",
                b"",
            )
        if "getprop ro.kernel.qemu" in text:
            return subprocess.CompletedProcess(
                command,
                0,
                (
                    b"1\n"
                    if self.emulator
                    else b"0\n"
                ),
                b"",
            )
        if "getprop ro.boot.qemu" in text:
            return subprocess.CompletedProcess(
                command,
                0,
                b"0\n",
                b"",
            )
        if (
            "pm path ai.vn97.app"
            in text
        ):
            return subprocess.CompletedProcess(
                command,
                0,
                (
                    b"package:/data/app/base.apk\n"
                    if self.installed
                    else b""
                ),
                b"",
            )
        if " install " in (
            " " + text + " "
        ):
            return subprocess.CompletedProcess(
                command,
                0,
                b"Success\n",
                b"",
            )
        if (
            "am start" in text
            and CAMPAIGN.ANDROID_ACTION
            in text
        ):
            self.triggered = True
            return subprocess.CompletedProcess(
                command,
                0,
                b"Status: ok\n",
                b"",
            )
        if (
            "exec-out run-as"
            in text
            and CAMPAIGN.EVIDENCE_FILE
            in text
        ):
            return subprocess.CompletedProcess(
                command,
                (
                    0
                    if self.triggered
                    else 1
                ),
                (
                    self.evidence
                    if self.triggered
                    else b""
                ),
                b"",
            )
        if (
            "exec-out run-as"
            in text
            and CAMPAIGN.ERROR_FILE
            in text
        ):
            return subprocess.CompletedProcess(
                command,
                1,
                b"",
                b"missing",
            )
        return subprocess.CompletedProcess(
            command,
            0,
            b"",
            b"",
        )


def main() -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip().lower()
    verified = CAMPAIGN._verify_repository(
        ROOT,
        expected_commit=commit,
    )
    assert verified == ROOT.resolve()

    config = Config(
        warmup_runs=1,
        measured_runs=5,
        decode_tokens=16,
        speech_frames=8,
        timeout_seconds=30,
    )
    expect_failure(
        "measured runs lower bound",
        lambda: Config(
            measured_runs=2,
        ),
    )

    raw = evidence_bytes()
    parsed = DEVICE.parse_device_evidence(
        raw
    )
    assets = Assets(
        capability_package=b"package",
        signature_envelope=b"signature",
        publisher_public_key=b"k" * 32,
        model_image_sha256="a" * 64,
    )
    criteria = Criteria(
        min_runs=5,
        max_text_prefill_p95_ms=20.0,
        max_text_decode_p95_ms_per_token=10.0,
        max_peak_pss_kib=200000,
        max_thermal_status=3,
        max_speech_prefill_p95_ms=20.0,
    )

    with tempfile.TemporaryDirectory() as tmp:
        apk = Path(tmp) / "app-debug.apk"
        apk.write_bytes(b"apk")

        fake = FakeExecutor(raw)
        adb = AdbClient(
            "adb",
            executor=fake,
        )
        result = CAMPAIGN._collect_one(
            adb=adb,
            serial="physical-001",
            apk=apk,
            assets=assets,
            config=config,
            criteria=criteria,
            speech_enabled=True,
        )
        assert (
            result.evidence
            .model_image_sha256
            == "a" * 64
        )
        assert (
            result.canonical_bytes
            == raw
        )
        flattened = [
            " ".join(command)
            for command, _ in fake.commands
        ]
        assert any(
            CAMPAIGN.ANDROID_ACTION
            in item
            for item in flattened
        )
        assert any(
            "uninstall ai.vn97.app"
            in item
            for item in flattened
        )
        staged_payloads = [
            payload
            for command, payload
            in fake.commands
            if (
                payload is not None
                and "run-as ai.vn97.app sh -c"
                in " ".join(command)
            )
        ]
        assert assets.capability_package in staged_payloads
        assert assets.signature_envelope in staged_payloads
        assert assets.publisher_public_key in staged_payloads

        emulator = FakeExecutor(
            raw,
            emulator=True,
        )
        expect_failure(
            "emulator evidence",
            lambda:
                CAMPAIGN._collect_one(
                    adb=AdbClient(
                        "adb",
                        executor=emulator,
                    ),
                    serial="emulator-001",
                    apk=apk,
                    assets=assets,
                    config=config,
                    criteria=criteria,
                    speech_enabled=True,
                ),
        )
        assert not any(
            " install " in
            (" " + " ".join(command) + " ")
            for command, _
            in emulator.commands
        )

        existing = FakeExecutor(
            raw,
            installed=True,
        )
        expect_failure(
            "existing app state",
            lambda:
                CAMPAIGN._collect_one(
                    adb=AdbClient(
                        "adb",
                        executor=existing,
                    ),
                    serial="physical-existing",
                    apk=apk,
                    assets=assets,
                    config=config,
                    criteria=criteria,
                    speech_enabled=True,
                ),
        )

    with tempfile.TemporaryDirectory() as tmp:
        evidence_dir = (
            Path(tmp) /
            "device-evidence"
        )
        evidence_dir.mkdir()
        record = Collected(
            canonical_bytes=raw,
            evidence=parsed,
        )
        CAMPAIGN._publish_evidence_atomic(
            evidence_dir,
            (record,),
        )
        files = list(
            evidence_dir.iterdir()
        )
        assert len(files) == 1
        assert files[0].suffix == ".json"
        loaded = DEVICE.load_device_evidence(
            files[0]
        )
        assert (
            loaded.evidence_sha256
            == parsed.evidence_sha256
        )

    with tempfile.TemporaryDirectory() as tmp:
        evidence_dir = (
            Path(tmp) /
            "device-evidence"
        )
        evidence_dir.mkdir()
        duplicate = Collected(
            canonical_bytes=raw,
            evidence=parsed,
        )
        expect_failure(
            "duplicate atomic publish",
            lambda:
                CAMPAIGN._publish_evidence_atomic(
                    evidence_dir,
                    (
                        duplicate,
                        duplicate,
                    ),
                ),
        )
        assert not any(
            evidence_dir.iterdir()
        )

    activity = (
        ROOT /
        "android/app/src/main/java/ai/vn97/app/VN97MainActivity.kt"
    ).read_text(
        encoding="utf-8"
    )
    handler = activity.index(
        "private fun handleM19JDeveloperEvidenceIntent"
    )
    turnkey = activity.index(
        "BuildConfig.VN97_TURNKEY_REQUIRED",
        handler,
    )
    provision = activity.index(
        "app.provisioner.review(",
        handler,
    )
    collect = activity.index(
        ".collectMobileEvidence(",
        handler,
    )
    assert handler < turnkey < provision < collect
    assert CAMPAIGN.ANDROID_ACTION in activity

    print(
        "M19J physical-device evidence orchestrator contracts: PASS"
    )


if __name__ == "__main__":
    main()
